import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.health import router as health_router
from api.db.session import init_engine, get_sessionmaker, get_db
from api.routes.user import router as user_router
from api.routes.recommend import router as recommend_router
from api.routes.watch import router as watch_router
from api.routes.feedback import router as feedback_router

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
logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("httpcore").setLevel(logging.INFO)

app = FastAPI(title="Seen'emAll", version="0.1.0")

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


@app.on_event("startup")
def on_startup():
    # When tests override get_db we skip touching the real database.
    if get_db in app.dependency_overrides:
        return
    init_engine()
    # Ensure we can get a sessionmaker without error
    get_sessionmaker()
