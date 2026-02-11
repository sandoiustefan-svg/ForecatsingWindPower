from datetime import datetime
from pydantic import BaseModel
from typing import Optional


class PredictionOut(BaseModel):
    id: int
    run_id: str

    model_id: str
    region: str

    step: int
    time: datetime

    y_pred: float

    created_at: datetime

    class Config:
        from_attributes = True

class PredictionRequest(BaseModel):
    model_id: str
    region: str
    t0: datetime
    x_window: list[list[float]]
