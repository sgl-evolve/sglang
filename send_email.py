#!/usr/bin/env python3
"""Send an [AutoEvolve][kv-heron-e29] email. Usage: send_email.py "<subject>" "<body>" """
import os, sys, ssl, smtplib
from email.mime.text import MIMEText

SENDER = "chengjunyan2@gmail.com"
TO = "chengjunyan0@gmail.com"
NAME = "kv-heron-e29"

def main():
    subj = sys.argv[1]
    body = sys.argv[2]
    pw = os.environ["APP_PASSWORD"].replace(" ", "")
    msg = MIMEText(body)
    msg["Subject"] = f"[AutoEvolve][{NAME}] {subj}"
    msg["From"] = SENDER
    msg["To"] = TO
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as s:
        s.login(SENDER, pw)
        s.sendmail(SENDER, [TO], msg.as_string())
    print("sent:", subj)

if __name__ == "__main__":
    main()
