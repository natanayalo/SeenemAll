from fastapi import APIRouter
from api.core.metrics import METRICS

router = APIRouter()


@router.get("/healthz")
def healthz():
    return {"status": "ok"}


@router.get("/healthz/metrics")
def healthz_metrics():
    return METRICS.snapshot()
