import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from api.db.session import Base


class Prediction(Base):
    """
    One row = one predicted hour.

    A forecast of 61 hours will generate 61 rows,
    all sharing the same run_id.
    """

    __tablename__ = "predictions"

    # Primary key (unique row)
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    # Groups the 61 predicted rows together
    run_id: Mapped[str] = mapped_column(
        String(36),
        index=True,
        nullable=False,
        default=lambda: str(uuid.uuid4())
    )

    # Which trained model produced this forecast
    model_id: Mapped[str] = mapped_column(
        String(255),
        index=True,
        nullable=False
    )

    # Region (NO1, NO2, etc.)
    region: Mapped[str] = mapped_column(
        String(32),
        index=True,
        nullable=False
    )

    # Forecast horizon step: 1..61
    step: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    # The timestamp that this prediction represents
    time: Mapped[datetime] = mapped_column(
        DateTime,
        index=True,
        nullable=False
    )

    # Predicted value at that hour
    y_pred: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )

    # When the prediction was generated/stored
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )
