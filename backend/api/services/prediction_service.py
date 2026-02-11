from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from api.models.prediction import Prediction


def list_predictions(
    db: Session,
    *,
    run_id: Optional[str] = None,
    region: Optional[str] = None,
    model_id: Optional[str] = None,
    time_from: Optional[datetime] = None,
    time_to: Optional[datetime] = None,
    limit: int = 5000,
) -> list[Prediction]:
    stmt = select(Prediction)

    if run_id:
        stmt = stmt.where(Prediction.run_id == run_id)
    if region:
        stmt = stmt.where(Prediction.region == region)
    if model_id:
        stmt = stmt.where(Prediction.model_id == model_id)
    if time_from:
        stmt = stmt.where(Prediction.time >= time_from)
    if time_to:
        stmt = stmt.where(Prediction.time <= time_to)

    stmt = stmt.order_by(Prediction.time.asc()).limit(limit)
    return db.execute(stmt).scalars().all()


def create_prediction_group(
    db: Session,
    *,
    model_id: str,
    region: str,
    t0: datetime,
    y_pred: list[float],
) -> str:
    if len(y_pred) != 61:
        raise ValueError(f"Expected 61 predictions, got {len(y_pred)}")

    run_id = str(uuid.uuid4())
    created_at = datetime.utcnow()

    rows = [
        Prediction(
            run_id=run_id,
            model_id=model_id,
            region=region,
            step=step,
            time=t0 + timedelta(hours=step),
            y_pred=float(val),
            created_at=created_at,
        )
        for step, val in enumerate(y_pred, start=1)
    ]

    db.add_all(rows)
    db.commit()
    return run_id


def get_group(db: Session, run_id: str) -> list[Prediction]:
    stmt = (
        select(Prediction)
        .where(Prediction.run_id == run_id)
        .order_by(Prediction.step.asc())
    )
    return db.execute(stmt).scalars().all()


def delete_group(db: Session, run_id: str) -> int:
    stmt = delete(Prediction).where(Prediction.run_id == run_id)
    res = db.execute(stmt)
    db.commit()
    return int(res.rowcount or 0)
