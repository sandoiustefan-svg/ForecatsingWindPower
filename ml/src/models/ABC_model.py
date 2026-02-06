from abc import ABC, abstractmethod
from pathlib import Path
import matplotlib.pyplot as plt


class DLModel(ABC):
    """
    Base class for all deep learning models.

    Every model MUST implement:
      - fit()
      - predict()
      - save_weights()
    """

    def __init__(self, name="model", save_dir="artifacts"):
        self.name = name
        self.save_dir = Path(save_dir) / name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.history = None

    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val, **kwargs):
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, X):
        """Return predictions"""
        pass

    @abstractmethod
    def save_weights(self):
        """Save model weights"""
        pass

    def plot_learning_curves(self):
        """Plot loss + val_loss after training"""

        if self.history is None:
            print("No training history found.")
            return

        plt.figure()
        plt.plot(self.history["loss"], label="train loss")
        plt.plot(self.history["val_loss"], label="val loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Learning Curves")
        plt.legend()
        plt.savefig(self.save_dir / "learning_curves.png")
        plt.show()

    def plot_predictions(self, time, y_true, y_pred):
        """Plot ground truth vs prediction over time"""

        plt.figure()
        plt.plot(time, y_true, label="Ground Truth")
        plt.plot(time, y_pred, label="Prediction")

        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("Truth vs Prediction")
        plt.legend()
        plt.savefig(self.save_dir / "predictions.png")
        plt.show()
