from fastapi import APIRouter
from fastapi.responses import JSONResponse
from api.core.metrics import METRICS

router = APIRouter()


@router.get("/healthz")
def healthz():
    from api.core.fast_intent_parser import FastIntentParser

    if FastIntentParser.get_instance().gliner_failed:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "component": "intent_parser",
                "reason": "required neural runtime load or inference failed",
            },
        )
    return {"status": "ok"}


@router.get("/healthz/metrics")
def healthz_metrics():
    return METRICS.snapshot()
