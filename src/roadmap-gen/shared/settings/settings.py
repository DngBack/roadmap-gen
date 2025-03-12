from __future__ import annotations

from dotenv import find_dotenv
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

from .models import OpenAISettings

# test in locals
load_dotenv(find_dotenv(".env"))


class Settings(BaseSettings):
    openai: OpenAISettings

    class Config:
        env_nested_delimiter = "__"


def load_settings() -> Settings:
    return Settings()
