"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from aerodiagnosis.bootstrap import bootstrap
from aerodiagnosis.config import RuntimeSettings
from aerodiagnosis.version import __version__

from .rate_limit import RateLimiter
from .routers import router


def create_app(settings: RuntimeSettings | None = None) -> FastAPI:
    configured = settings or RuntimeSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.application = bootstrap(configured)
        app.state.rate_limiter = (
            RateLimiter(
                requests_per_minute=configured.rate_limit_diagnosis_per_minute
            )
            if configured.rate_limit_enabled
            else None
        )
        yield

    application = FastAPI(
        title="AeroDiagnosis",
        version=__version__,
        description="Evidence-grounded diagnostic application runtime.",
        lifespan=lifespan,
    )
    application.include_router(router)
    static_root = Path(__file__).parent.parent / "web" / "static"
    application.mount("/assets", StaticFiles(directory=static_root), name="assets")

    @application.get("/", include_in_schema=False)
    def frontend() -> FileResponse:
        return FileResponse(static_root / "index.html")

    return application


app = create_app()


def main() -> None:
    settings = RuntimeSettings.from_env()
    uvicorn.run(create_app(settings), host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
