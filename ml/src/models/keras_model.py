import yaml
import tensorflow as tf
from .ABC_model import DLModel


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = int(d_model)

    def call(self, x):
        # x: (B, T, D)
        x = tf.cast(x, tf.float32)
        seq_len = tf.shape(x)[1]
        d_model = self.d_model

        position = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)  # (T, 1)
        div_term = tf.exp(
            tf.cast(tf.range(0, d_model, 2), tf.float32)
            * (-tf.math.log(10000.0) / tf.cast(d_model, tf.float32))
        )  # (ceil(D/2),)

        sin_part = tf.sin(position * div_term)
        cos_part = tf.cos(position * div_term)

        pe = tf.reshape(tf.stack([sin_part, cos_part], axis=-1), (seq_len, -1))  # (T, 2*ceil(D/2))
        pe = pe[:, :d_model]                      # (T, D)
        pe = pe[tf.newaxis, ...]                  # (1, T, D)
        return x + pe


class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float, reg=None):
        super().__init__()
        d_model = int(d_model)
        num_heads = int(num_heads)
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        head_dim = d_model // num_heads

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=head_dim,
            dropout=float(dropout),
        )

        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(int(ff_dim), activation="relu", kernel_regularizer=reg),
                tf.keras.layers.Dropout(float(dropout)),
                tf.keras.layers.Dense(d_model, kernel_regularizer=reg),
            ]
        )

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(float(dropout))
        self.drop2 = tf.keras.layers.Dropout(float(dropout))

    def call(self, x, training=False):
        attn = self.mha(x, x, training=training)
        x = self.norm1(x + self.drop1(attn, training=training))
        ffn_out = self.ffn(x, training=training)
        x = self.norm2(x + self.drop2(ffn_out, training=training))
        return x



class KerasModel(DLModel):
    def __init__(self, config_path, name="keras_model"):
        super().__init__(name)

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model = self.build_model()

    def _build_loss(self, loss_cfg):
        """
        loss_cfg can be:
          - "mse" / "mae" / "huber"
          - dict like {"name": "huber", "delta": 0.05}
          - dict like {"name": "weighted_mse", "alpha": 3.0}  (emphasize peaks)
        """
        if loss_cfg is None:
            return "mse"

        # string loss
        if isinstance(loss_cfg, str):
            name = loss_cfg.lower()
            if name in {"mse", "mae"}:
                return name
            if name == "huber":
                return tf.keras.losses.Huber()
            raise ValueError(f"Unknown loss string: {loss_cfg}")

        # dict loss
        name = str(loss_cfg.get("name", "mse")).lower()

        if name in {"mse", "mae"}:
            return name

        if name == "huber":
            delta = float(loss_cfg.get("delta", 0.05))
            return tf.keras.losses.Huber(delta=delta)

        if name == "weighted_mse":
            # weight = 1 + alpha * y_true (assumes y_true is scaled reasonably, e.g. 0..1)
            alpha = float(loss_cfg.get("alpha", 3.0))

            def weighted_mse(y_true, y_pred):
                w = 1.0 + alpha * tf.cast(y_true, tf.float32)
                return tf.reduce_mean(w * tf.square(tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32)))

            return weighted_mse

        raise ValueError(f"Unknown loss config: {loss_cfg}")

    def build_model(self):
        cfg = self.config["model"]
        model_type = cfg["type"]

        input_dim = int(cfg["input_dim"])
        horizon = int(cfg["horizon"])

        dropout = float(cfg.get("dropout", 0.0))
        l2v = float(cfg.get("l2", 0.0))
        use_ln = bool(cfg.get("layer_norm", False))

        # Optional: enforce non-negative outputs (use only if your target can't be negative)
        output_activation = cfg.get("output_activation", None)  # e.g. "relu" or null

        reg = tf.keras.regularizers.l2(l2v) if l2v > 0 else None

        inputs = tf.keras.layers.Input(shape=(None, input_dim))
        x = tf.keras.layers.LayerNormalization()(inputs)

        if model_type == "GRU":
            hidden_units = cfg["hidden_units"]

            for units in hidden_units[:-1]:
                x = tf.keras.layers.GRU(
                    int(units),
                    return_sequences=True,
                    dropout=dropout,
                    kernel_regularizer=reg,
                    recurrent_regularizer=reg,
                )(x)
                if use_ln:
                    x = tf.keras.layers.LayerNormalization()(x)

            x = tf.keras.layers.GRU(
                int(hidden_units[-1]),
                return_sequences=False,
                dropout=dropout,
                kernel_regularizer=reg,
                recurrent_regularizer=reg,
            )(x)

        elif model_type == "LSTM":
            hidden_units = cfg["hidden_units"]

            for units in hidden_units[:-1]:
                x = tf.keras.layers.LSTM(
                    int(units),
                    return_sequences=True,
                    dropout=dropout,
                    kernel_regularizer=reg,
                    recurrent_regularizer=reg,
                )(x)
                if use_ln:
                    x = tf.keras.layers.LayerNormalization()(x)

            x = tf.keras.layers.LSTM(
                int(hidden_units[-1]),
                return_sequences=False,
                dropout=dropout,
                kernel_regularizer=reg,
                recurrent_regularizer=reg,
            )(x)

        elif model_type == "Transformer":
            # Transformer-specific params (with sane defaults)
            d_model = int(cfg.get("d_model", 128))
            num_heads = int(cfg.get("num_heads", 4))
            ff_dim = int(cfg.get("ff_dim", 256))
            num_layers = int(cfg.get("num_layers", 4))

            # Project features -> d_model
            x = tf.keras.layers.Dense(d_model, kernel_regularizer=reg)(x)

            # Positional encoding
            x = PositionalEncoding(d_model)(x)
            x = tf.keras.layers.Dropout(dropout)(x)

            # Encoder stack
            for _ in range(num_layers):
                x = TransformerEncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    reg=reg,
                )(x)

            # Pool over time -> fixed vector
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            if use_ln:
                x = tf.keras.layers.LayerNormalization()(x)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        outputs = tf.keras.layers.Dense(horizon, activation=output_activation, kernel_regularizer=reg)(x)
        model = tf.keras.Model(inputs, outputs)

        # compile
        tcfg = self.config.get("training", {})
        lr = float(tcfg.get("lr", 1e-3))
        clipnorm = tcfg.get("clipnorm", None)
        if clipnorm is not None:
            clipnorm = float(clipnorm)

        opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)

        loss = self._build_loss(tcfg.get("loss", "mse"))

        model.compile(
            optimizer=opt,
            loss=loss,
            metrics=[
                tf.keras.metrics.MeanAbsoluteError(name="mae"),
                tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
        )

        return model

    def fit(self, X_train, y_train, X_val, y_val, **kwargs):
        hist = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), **kwargs)
        self.history = hist.history

    def predict(self, X):
        return self.model.predict(X)

    def save_weights(self):
        path = self.save_dir / "model.weights.h5"
        self.model.save_weights(path)
        print("Weights saved to:", path)
