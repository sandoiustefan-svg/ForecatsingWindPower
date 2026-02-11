from pydantic import BaseModel
from datetime import datetime
from typing import List

from prediction import PredictionOut

class ForecastOut(BaseModel):
    run_id: str
    model_id: str
    region: str

    created_at: datetime

    predictions: List[PredictionOut]