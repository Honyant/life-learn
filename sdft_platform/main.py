from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sdft_platform.db import SessionLocal, init_db
from sdft_platform.routers.auth import router as auth_router
from sdft_platform.routers.chats import router as chats_router
from sdft_platform.routers.models import router as models_router
from sdft_platform.routers.orgs import router as orgs_router
from sdft_platform.services.model_runtime import OrgModelRuntime
from sdft_platform.services.training_service import TrainingCoordinator
from sdft_platform.settings import get_settings


settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.1.0")

base_dir = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(base_dir / "templates"))
app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")

app.include_router(auth_router)
app.include_router(orgs_router)
app.include_router(chats_router)
app.include_router(models_router)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    settings.model_storage_dir.mkdir(parents=True, exist_ok=True)
    app.state.model_runtime = OrgModelRuntime(settings)
    app.state.training_coordinator = TrainingCoordinator(
        settings=settings,
        session_factory=SessionLocal,
        model_runtime=app.state.model_runtime,
    )


@app.on_event("shutdown")
def on_shutdown() -> None:
    runtime = getattr(app.state, "model_runtime", None)
    if runtime is not None:
        runtime.shutdown()
    coordinator = getattr(app.state, "training_coordinator", None)
    if coordinator is not None:
        coordinator.shutdown()


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "app_name": settings.app_name,
            "inference_backend": settings.inference_backend,
        },
    )


@app.get("/invite/{token}", response_class=HTMLResponse)
def invite_landing(request: Request, token: str) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "app_name": settings.app_name,
            "inference_backend": settings.inference_backend,
            "invite_token": token,
        },
    )
