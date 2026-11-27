"""Email: SMTP locally (MailHog or Gmail), Amazon SES in cloud mode."""

from __future__ import annotations

import smtplib
from email.message import EmailMessage
from typing import Protocol

from app.config import AppMode, Settings


class Mailer(Protocol):
    def send(self, subject: str, html: str, to: list[str] | None = None) -> None: ...


class SmtpMailer:
    def __init__(self, settings: Settings) -> None:
        self.s = settings

    def send(self, subject: str, html: str, to: list[str] | None = None) -> None:
        recipients = to or [a.strip() for a in self.s.email_to.split(",") if a.strip()]
        if not recipients:
            raise RuntimeError("EMAIL_TO is not set")
        msg = EmailMessage()
        msg["Subject"], msg["From"], msg["To"] = subject, self.s.email_from, ", ".join(recipients)
        msg.set_content("This email needs an HTML-capable client.")
        msg.add_alternative(html, subtype="html")
        with smtplib.SMTP(self.s.smtp_host, self.s.smtp_port, timeout=20) as smtp:
            if self.s.smtp_user and self.s.smtp_password:
                smtp.starttls()
                smtp.login(self.s.smtp_user, self.s.smtp_password.get_secret_value())
            smtp.send_message(msg)


class SesMailer:
    def __init__(self, settings: Settings) -> None:
        import boto3

        self.s = settings
        self.client = boto3.client("sesv2", region_name=settings.aws_region)

    def send(self, subject: str, html: str, to: list[str] | None = None) -> None:
        recipients = to or [a.strip() for a in self.s.email_to.split(",") if a.strip()]
        self.client.send_email(
            FromEmailAddress=self.s.email_from,
            Destination={"ToAddresses": recipients},
            Content={"Simple": {"Subject": {"Data": subject}, "Body": {"Html": {"Data": html}}}},
        )


def get_mailer(settings: Settings) -> Mailer:
    return SesMailer(settings) if settings.app_mode is AppMode.CLOUD else SmtpMailer(settings)
