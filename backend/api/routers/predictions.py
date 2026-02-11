from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from api.dependencies import get_db
from api.schemas.prediction import PredictionCreateRequest, PredictionOut
from api.services.inference_service import predict_h61
from api.services.prediction_service import (
    create_prediction_group,
    delete_group,
    get_group,
    list_predictions,
)

router = APIRouter(prefix="/predictions", tags=["predictions"])


def wants_csv(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    return "text/csv" in accept.lower()


def to_csv(rows, filename: str) -> StreamingResponse:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "run_id", "model_id", "region", "step", "time", "y_pred", "created_at"])

    for r in rows:
        w.writerow(
            [
                r.id,
                r.run_id,
                r.model_id,
                r.region,
                r.step,
                r.time.isoformat(),
                r.y_pred,
                r.created_at.isoformat(),
            ]
        )

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("", response_model=list[PredictionOut])
def get_predictions(
    request: Request,
    db: Session = Depends(get_db),
    run_id: Optional[str] = None,
    region: Optional[str] = None,
    model_id: Optional[str] = None,
    time_from: Optional[datetime] = None,
    time_to: Optional[datetime] = None,
    limit: int = Query(5000, ge=1, le=200000),
):
    rows = list_predictions(
        db,
        run_id=run_id,
        region=region,
        model_id=model_id,
        time_from=time_from,
        time_to=time_to,
        limit=limit,
    )

    if wants_csv(request):
        return to_csv(rows, filename="predictions.csv")

    return rows


@router.get("/run/{run_id}", response_model=list[PredictionOut])
def get_prediction_group(
    run_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    rows = get_group(db, run_id)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No predictions found for run_id={run_id}")

    if wants_csv(request):
        return to_csv(rows, filename=f"predictions_{run_id}.csv")

    return rows


@router.post("")
def create_predictions(
    request: Request,
    payload: PredictionCreateRequest,
    db: Session = Depends(get_db),
):
    # 1) inference (in memory)
    y_pred = predict_h61(model_id=payload.model_id, x_window=payload.x_window)

    # 2) store (61 rows)
    run_id = create_prediction_group(
        db,
        model_id=payload.model_id,
        region=payload.region,
        t0=payload.t0,
        y_pred=y_pred,
    )

    # 3) return stored rows (so client can fetch immediately)
    rows = get_group(db, run_id)

    if wants_csv(request):
        return to_csv(rows, filename=f"predictions_{run_id}.csv")

    return {
        "run_id": run_id,
        "count": len(rows),
        "predictions": [PredictionOut.model_validate(r) for r in rows],
    }


@router.delete("/run/{run_id}")
def delete_prediction_group(
    run_id: str,
    db: Session = Depends(get_db),
):
    deleted = delete_group(db, run_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"No predictions found for run_id={run_id}")
    return {"run_id": run_id, "deleted_rows": deleted}
