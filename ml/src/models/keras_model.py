import yaml
import tensorflow as tf
from .ABC_model import DLModel


class KerasModel(DLModel):
    def __init__(self, config_path, name="keras_model"):
        super().__init__(name)

        # Load YAML config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Build model automatically
        self.model = self.build_model()

    def build_model(self):
        cfg = self.config["model"]
        model_type = cfg["type"]

        input_dim = cfg["input_dim"]
        horizon = cfg["horizon"]

        hidden_units = cfg["hidden_units"]
        dropout = cfg["dropout"]

        # Input layer
        inputs = tf.keras.layers.Input(shape=(None, input_dim))

        x = inputs

        # GRU architecture
        if model_type == "GRU":
            for units in hidden_units:
                x = tf.keras.layers.GRU(units, return_sequences=True)(x)
                x = tf.keras.layers.Dropout(dropout)(x)

            x = tf.keras.layers.GRU(hidden_units[-1])(x)

        # LSTM architecture
        elif model_type == "LSTM":
            for units in hidden_units:
                x = tf.keras.layers.LSTM(units, return_sequences=True)(x)
                x = tf.keras.layers.Dropout(dropout)(x)

            x = tf.keras.layers.LSTM(hidden_units[-1])(x)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Output layer (forecast horizon)
        outputs = tf.keras.layers.Dense(horizon)(x)

        model = tf.keras.Model(inputs, outputs)

        # Compile model
        lr = self.config["training"]["lr"]
        model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")

        return model

    def fit(self, X_train, y_train, X_val, y_val, **kwargs):
        hist = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), **kwargs)

        self.history = hist.history

    def predict(self, X):
        return self.model.predict(X)

    def save_weights(self):
        path = self.save_dir / "weights.h5"
        self.model.save_weights(path)
        print("Weights saved to:", path)
