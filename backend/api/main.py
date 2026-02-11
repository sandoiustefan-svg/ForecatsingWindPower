from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers.predictions import router as predictions_router
from api.db.settings import Settings


def create_app() -> FastAPI:
    settings = Settings.from_env()

    app = FastAPI(
        title="Wind Power Forecast API",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=getattr(settings, "CORS_ORIGINS", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(predictions_router)

app = create_app()
