#!/usr/bin/env python3
"""Send an [AutoEvolve][onyx-7q2] email. Usage: send_email.py "<subject>" <body_file>"""
import os, sys, smtplib
from email.mime.text import MIMEText

SENDER = "chengjunyan2@gmail.com"
RECIP = "chengjunyan0@gmail.com"
NAME = "onyx-7q2"

def main():
    subject = sys.argv[1]
    body = open(sys.argv[2]).read() if len(sys.argv) > 2 else sys.stdin.read()
    pw = os.environ["APP_PASSWORD"].replace(" ", "")
    msg = MIMEText(body)
    msg["Subject"] = f"[AutoEvolve][{NAME}] {subject}"
    msg["From"] = SENDER
    msg["To"] = RECIP
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(SENDER, pw)
        s.sendmail(SENDER, [RECIP], msg.as_string())
    print(f"sent: {msg['Subject']}")

if __name__ == "__main__":
    main()
