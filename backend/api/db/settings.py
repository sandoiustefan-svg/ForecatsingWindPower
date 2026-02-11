from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import ValidationInfo, field_validator
import os
from dotenv import dotenv_values

class Settings(BaseSettings):
    app_name: str = "WindAi_Predcition"

    BACKEND_CORS_ORIGINS: List[str] = ["*"]

    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str
    DB_HOST: str
    DB_PORT: str
    
    # Database URI can be provided or built from other variables
    DATABASE_URI: str = ""

    @field_validator("DATABASE_URI", mode="after")
    def assemble_db_connection(cls, uri: str, info: ValidationInfo) -> str:
        """Builds the MariaDB connector if it is not provided."""

        if uri != "":
            return uri
        
        return 'mariadb+mariadbconnector://%s:%s@%s:%s/%s' % (
            info.data.get("DB_USER"),
            info.data.get("DB_PASSWORD"),
            info.data.get("DB_HOST"),
            info.data.get("DB_PORT"),
            info.data.get('DB_NAME'),
        )

    @classmethod
    def from_env(cls):
        """
        Tries to load from .env first if it exists, otherwise from the environment variables.
        Allows running the app locally (outside docker) without having to write env variables.
        """

        raw_vars: dict[str, Optional[str]] = {
            **dotenv_values("../.env"),
            **os.environ
        }
        env_vars = {k: v for k, v in raw_vars.items() if v is not None}

        return cls(
            DATABASE_URI=env_vars.get("DATABASE_URI") or "",
            DB_NAME=env_vars["DATABASE_NAME"],
            DB_USER=env_vars["DATABASE_USER"],
            DB_PASSWORD=env_vars["DATABASE_PASSWORD"],
            DB_HOST=env_vars["DATABASE_HOST"],
            DB_PORT=env_vars["DATABASE_PORT"],
        )

    class Config:
        case_sensitive = True

settings = Settings.from_env()
print(settings)
