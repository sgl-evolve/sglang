#!/usr/bin/env python3
"""Send an [AutoEvolve][kv-onyx-mf76] email. Usage: send_email.py <subject> <body_file_or_-->
Reads APP_PASSWORD from env (.env). From chengjunyan2@gmail.com -> chengjunyan0@gmail.com.
"""
import os, smtplib, sys
from email.mime.text import MIMEText

NAME = "kv-onyx-mf76"
SENDER = "chengjunyan2@gmail.com"
RECIPIENT = "chengjunyan0@gmail.com"


def main():
    subject = sys.argv[1]
    body_arg = sys.argv[2] if len(sys.argv) > 2 else "-"
    if body_arg == "-":
        body = sys.stdin.read()
    else:
        with open(body_arg) as f:
            body = f.read()
    pw = (os.environ.get("APP_PASSWORD") or "").replace(" ", "")
    if not pw:
        print("ERROR: APP_PASSWORD not set"); sys.exit(2)
    msg = MIMEText(body)
    msg["Subject"] = f"[AutoEvolve][{NAME}] {subject}"
    msg["From"] = SENDER
    msg["To"] = RECIPIENT
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(SENDER, pw)
        s.sendmail(SENDER, [RECIPIENT], msg.as_string())
    print("EMAIL SENT:", msg["Subject"])


if __name__ == "__main__":
    main()
