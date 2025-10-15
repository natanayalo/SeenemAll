import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.health import router as health_router
from api.db.session import init_engine, get_sessionmaker, get_db
from api.routes.user import router as user_router
from api.routes.recommend import router as recommend_router
from api.routes.watch import router as watch_router
from api.routes.feedback import router as feedback_router
from api.config import TMDB_API_KEY
from etl.tmdb_client import TMDBClient

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# Set specific loggers
logging.getLogger("api.core.reranker").setLevel(logging.DEBUG)
logging.getLogger("api.core.intent_parser").setLevel(logging.DEBUG)
# Reduce noise from other modules
logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
logging.getLogger("urllib3").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.INFO)


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    _initialise_application(app)
    tmdb_client = TMDBClient(TMDB_API_KEY) if TMDB_API_KEY else None
    app.state.tmdb_client = tmdb_client
    yield
    client = getattr(app.state, "tmdb_client", None)
    if client is not None:
        await client.aclose()
        app.state.tmdb_client = None


app = FastAPI(title="Seen'emAll", version="0.1.0", lifespan=app_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(health_router, prefix="")
app.include_router(
    user_router,
)
app.include_router(recommend_router)
app.include_router(watch_router)
app.include_router(feedback_router)


def _initialise_application(app: FastAPI) -> None:
    # When tests override get_db we skip touching the real database.
    if get_db in app.dependency_overrides:
        return
    init_engine()
    # Ensure we can get a sessionmaker without error
    get_sessionmaker()
    if not hasattr(app.state, "tmdb_client"):
        app.state.tmdb_client = None


def on_startup() -> None:
    """Backward-compatible startup hook kept for existing tests."""
    _initialise_application(app)
