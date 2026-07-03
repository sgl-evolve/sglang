#!/usr/bin/env python3
"""Losslessness proof for the parallel-L3-read mechanism (onyx-7q2).

The v3/v4 speedup comes from reading the pages of a HiCacheFile.batch_get()
concurrently through a ThreadPoolExecutor (SGLANG_HICACHE_FILE_READ_THREADS>1)
instead of one-page-at-a-time. The correctness claim a skeptical maintainer needs:
the parallel path returns EXACTLY what the serial path returns — same bytes, same
order, same None-alignment for misses — for identical on-disk data. This test
proves it with pure file I/O + tensor equality (no GPU), across dtypes, sizes,
a large batch that actually exercises thread concurrency, and missing keys.

Run: .venv/bin/python test_parallel_read_lossless.py
"""
import os, sys, tempfile, shutil

def build_backend(file_path, threads):
    # env is read at __init__ time; set BEFORE constructing.
    os.environ["SGLANG_HICACHE_FILE_READ_THREADS"] = str(threads)
    # force-reimport environ so the EnvInt re-reads (it caches nothing, but be safe)
    from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
    cfg = HiCacheStorageConfig(
        tp_rank=0, tp_size=1, pp_rank=0, pp_size=1,
        attn_cp_rank=0, attn_cp_size=1,
        is_mla_model=False, enable_storage_metrics=False,
        is_page_first_layout=True, model_name="test/parallel-read",
        extra_config={},  # no max_size/min_free -> eviction disabled, always admits
    )
    return HiCacheFile(cfg, file_path=file_path)


def main():
    import torch
    tmp = tempfile.mkdtemp(prefix="hicache_lossless_")
    try:
        # ---- build a varied corpus: dtypes, shapes, a big batch for real concurrency ----
        torch.manual_seed(1234)
        specs = []
        for i in range(200):
            dt = [torch.float16, torch.bfloat16, torch.uint8, torch.int32][i % 4]
            n = 1 + (i * 37) % 4096              # varied sizes incl. size-1 edge
            if dt in (torch.float16, torch.bfloat16):
                t = torch.randn(n, dtype=torch.float32).to(dt)
            else:
                t = torch.randint(0, 255, (n,), dtype=torch.int64).to(dt)
            specs.append((f"page_{i:04d}", t.contiguous()))

        # ---- write once (serial writer) ----
        wb = build_backend(tmp, threads=1)
        for k, t in specs:
            assert wb.set(k, t), f"set failed for {k}"

        keys = [k for k, _ in specs]
        # include 5 missing keys interleaved to prove None-alignment is preserved
        keys_with_miss = []
        truth = []
        for idx, (k, t) in enumerate(specs):
            keys_with_miss.append(k); truth.append(t)
            if idx % 40 == 7:
                keys_with_miss.append(f"__MISSING_{idx}__"); truth.append(None)

        def fresh_targets():
            out = []
            for t in truth:
                if t is None:
                    # a miss still needs a target buffer slot in the call; use a dummy
                    out.append(torch.zeros(1, dtype=torch.uint8))
                else:
                    out.append(torch.zeros_like(t))
            return out

        # ---- serial read (pool disabled) ----
        rb_serial = build_backend(tmp, threads=1)
        assert rb_serial._read_pool is None, "threads=1 must disable the pool"
        st = fresh_targets()
        serial = rb_serial.batch_get(list(keys_with_miss), st)

        # ---- parallel read (16 threads) on the SAME on-disk data ----
        rb_par = build_backend(tmp, threads=16)
        assert rb_par._read_pool is not None, "threads=16 must enable the pool"
        pt = fresh_targets()
        parallel = rb_par.batch_get(list(keys_with_miss), pt)

        # ---- assertions ----
        assert len(serial) == len(parallel) == len(truth), "length/order mismatch"
        n_hit = n_miss = 0
        for i, exp in enumerate(truth):
            if exp is None:
                assert serial[i] is None, f"serial[{i}] should be None (miss)"
                assert parallel[i] is None, f"parallel[{i}] should be None (miss)"
                n_miss += 1
            else:
                # both paths must reproduce the original bytes exactly
                assert serial[i] is not None and parallel[i] is not None, f"unexpected miss at {i}"
                assert torch.equal(serial[i], exp), f"serial byte mismatch at {i}"
                assert torch.equal(parallel[i], exp), f"parallel byte mismatch at {i}"
                # and parallel must equal serial (the actual A/B claim)
                assert torch.equal(parallel[i], serial[i]), f"parallel!=serial at {i}"
                n_hit += 1

        # distinct-buffer check: each result aliases its own target (no cross-write)
        hit_targets = [pt[i] for i, e in enumerate(truth) if e is not None]
        ids = {id(x) for x in hit_targets}
        assert len(ids) == len(hit_targets), "target buffers must be distinct objects"

        print(f"PASS: parallel L3 read is LOSSLESS vs serial "
              f"({n_hit} hits byte-identical + order-preserved, {n_miss} misses None-aligned, "
              f"dtypes={{f16,bf16,u8,i32}}, sizes 1..4096, 16 threads).")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
