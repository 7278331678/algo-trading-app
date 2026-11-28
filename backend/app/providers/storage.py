"""Object storage: MinIO locally, S3 in cloud. Both speak the S3 API."""

from __future__ import annotations

from app.config import AppMode, Settings


class ObjectStore:
    def __init__(self, settings: Settings) -> None:
        import boto3

        kwargs: dict = {"region_name": settings.aws_region}
        if settings.app_mode is AppMode.LOCAL:
            kwargs.update(
                endpoint_url=settings.s3_endpoint_url,
                aws_access_key_id=settings.s3_access_key,
                aws_secret_access_key=(
                    settings.s3_secret_key.get_secret_value() if settings.s3_secret_key else None
                ),
            )
        # In cloud mode credentials come from the task's IAM role, never from env vars.
        self.client = boto3.client("s3", **kwargs)
        self.bucket = settings.s3_bucket

    def ensure_bucket(self) -> None:
        names = {b["Name"] for b in self.client.list_buckets().get("Buckets", [])}
        if self.bucket not in names:
            self.client.create_bucket(Bucket=self.bucket)

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)

    def get_bytes(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def list_keys(self, prefix: str) -> list[str]:
        keys: list[str] = []
        for page in self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix):
            keys += [o["Key"] for o in page.get("Contents", [])]
        return keys

    def move(self, src: str, dst: str) -> None:
        self.client.copy_object(Bucket=self.bucket, Key=dst, CopySource={"Bucket": self.bucket, "Key": src})
        self.client.delete_object(Bucket=self.bucket, Key=src)
