"""Application settings, read from environment variables (and .env locally).

APP_MODE picks the infrastructure (local vs AWS). TRADING_MODE picks paper vs
live orders. They are independent: you can paper-trade in cloud mode.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from urllib.parse import quote_plus

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppMode(StrEnum):
    LOCAL = "local"
    CLOUD = "cloud"


class TradingMode(StrEnum):
    PAPER = "paper"
    LIVE = "live"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_mode: AppMode = AppMode.LOCAL
    trading_mode: TradingMode = TradingMode.PAPER
    live_trading_confirm: str = ""  # must equal "I-UNDERSTAND-THE-RISK" to allow live orders
    log_level: str = "INFO"

    # Zerodha Kite Connect
    kite_api_key: str = ""
    kite_api_secret: SecretStr = SecretStr("")
    kite_redirect_url: str = "http://127.0.0.1:8100/auth/kite/callback"

    # Data stores. DATABASE_URL wins if set; otherwise the URL is built from the parts
    # below with the password URL-encoded, so characters like @ : / # are safe.
    database_url: str | None = None
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_user: str = "trading"
    postgres_db: str = "trading"
    postgres_password: SecretStr = SecretStr("")
    redis_url: str = "redis://redis:6379/0"

    # Object storage (MinIO locally, S3 in cloud)
    s3_endpoint_url: str | None = "http://minio:9000"
    s3_bucket: str = "trading-app"
    s3_access_key: str | None = None
    s3_secret_key: SecretStr | None = None
    aws_region: str = "ap-south-1"

    # Experiment tracking (in-cluster MLflow locally; unset = tracking off)
    mlflow_tracking_uri: str | None = None

    # LLM
    ollama_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "llama3.1:8b"
    bedrock_model_id: str = ""

    # Email
    smtp_host: str = "mailhog"
    smtp_port: int = 1025
    smtp_user: str | None = None
    smtp_password: SecretStr | None = None
    email_from: str = "trading-app@localhost"
    email_to: str = ""

    # Market data (Phase 1)
    universe_index: str = "NIFTY 100"
    universe_csv_url: str = "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv"
    kite_hist_rate_per_sec: float = Field(2.5, gt=0, le=3)  # Kite allows ~3 historical calls/sec
    retention_minute_days: int = Field(90, ge=1)
    retention_5minute_days: int = Field(730, ge=1)
    retention_15minute_days: int = Field(730, ge=1)
    retention_day_days: int = Field(1095, ge=1)

    # Risk defaults (overridable later from the dashboard)
    capital: float = Field(100_000, gt=0)
    risk_per_trade_pct: float = Field(0.5, gt=0, le=2)
    daily_loss_cap_pct: float = Field(2.0, gt=0, le=10)
    max_trades_per_day: int = Field(5, ge=0, le=10)

    @model_validator(mode="after")
    def _guard_live(self) -> Settings:
        if self.trading_mode is TradingMode.LIVE and self.live_trading_confirm != "I-UNDERSTAND-THE-RISK":
            raise ValueError(
                "TRADING_MODE=live requires LIVE_TRADING_CONFIRM=I-UNDERSTAND-THE-RISK"
            )
        return self

    @property
    def db_url(self) -> str:
        if self.database_url:
            return self.database_url
        user = quote_plus(self.postgres_user)
        pwd = quote_plus(self.postgres_password.get_secret_value())
        return (
            f"postgresql+psycopg://{user}:{pwd}@{self.postgres_host}:{self.postgres_port}/"
            f"{quote_plus(self.postgres_db)}"
        )

    @property
    def retention_days(self) -> dict[str, int]:
        return {
            "minute": self.retention_minute_days,
            "5minute": self.retention_5minute_days,
            "15minute": self.retention_15minute_days,
            "day": self.retention_day_days,
        }

    @property
    def kite_configured(self) -> bool:
        return bool(self.kite_api_key and self.kite_api_secret.get_secret_value())


@lru_cache
def get_settings() -> Settings:
    return Settings()
