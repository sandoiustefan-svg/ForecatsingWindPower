import json
from pathlib import Path

import numpy as np
import yaml
import tensorflow as tf

from .keras_model import KerasModel


def _cast_callback_params(params: dict) -> dict:
    float_keys = {"min_lr", "factor", "min_delta", "threshold"}
    int_keys = {"patience", "cooldown", "verbose"}

    for k in list(params.keys()):
        v = params[k]
        if v is None:
            continue

        if k in float_keys:
            params[k] = float(v)
        elif k in int_keys:
            params[k] = int(v)

    return params


def build_callbacks(fit_cfg: dict, run_dir: Path):
    cb_cfgs = fit_cfg.pop("callbacks", [])
    callbacks = []

    for cb in cb_cfgs:
        name = cb["name"]
        params = {k: v for k, v in cb.items() if k != "name"}
        params = _cast_callback_params(params)

        if name == "EarlyStopping":
            callbacks.append(tf.keras.callbacks.EarlyStopping(**params))
        elif name == "ReduceLROnPlateau":
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(**params))
        elif name == "ModelCheckpoint":
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(**params))
        else:
            raise ValueError(f"Unknown callback: {name}")

    callbacks.append(tf.keras.callbacks.CSVLogger(str(run_dir / "history.csv")))
    return callbacks


def load_npz(npz_path: str):
    data = np.load(npz_path)
    return (
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        data["X_test"],
        data["y_test"],
    )


def save_yaml(path: Path, cfg: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def save_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def evaluate(model: KerasModel, X, y):
    res = model.model.evaluate(X, y, verbose=0, return_dict=True)
    return {k: float(v) for k, v in res.items()}


def main(
    npz_path="src/processed_data/pipeline_data/windows_h61.npz",
    multi_cfg_path="Transformer_config.yaml",
    artifacts_root="artifacts",
):
    X_train, y_train, X_val, y_val, X_test, y_test = load_npz(npz_path)

    with open(multi_cfg_path, "r") as f:
        multi_cfg = yaml.safe_load(f)

    archs = multi_cfg["architectures"]
    fit_defaults = multi_cfg["defaults"]["fit"]

    for arch_name, arch_cfg in archs.items():
        print("\n==============================")
        print(f"Training architecture: {arch_name}")
        print("==============================")

        run_dir = Path(artifacts_root) / arch_name
        run_dir.mkdir(parents=True, exist_ok=True)

        run_config_path = run_dir / "config_used.yaml"
        save_yaml(run_config_path, arch_cfg)

        model = KerasModel(config_path=str(run_config_path), name=arch_name)

        fit_cfg = dict(fit_defaults)  # copy so we don't mutate shared dict
        callbacks = build_callbacks(fit_cfg, run_dir)

        model.fit(
            X_train,
            y_train,
            X_val,
            y_val,
            callbacks=callbacks,
            **fit_cfg,
            verbose=2,
        )

        model.save_weights()
        model.plot_learning_curves()

        # Plot one example
        i = 0
        y_pred_one = model.predict(X_test[i : i + 1])[0]  # (61,)
        t = np.arange(y_test.shape[1])  # 0..60
        model.plot_predictions(t, y_test[i], y_pred_one)

        # ✅ Save a small preview of predictions for API usage
        N = 32
        y_pred_batch = model.predict(X_test[:N])  # (N, 61)
        y_true_batch = y_test[:N]                 # (N, 61)

        save_npz(
            run_dir / "predictions_preview.npz",
            t=t,
            y_true=y_true_batch,
            y_pred=y_pred_batch,
        )

        val_metrics = evaluate(model, X_val, y_val)
        test_metrics = evaluate(model, X_test, y_test)

        results = {
            "arch_name": arch_name,
            "fit_defaults": fit_defaults,
            "val": val_metrics,
            "test": test_metrics,
        }
        save_json(run_dir / "metrics.json", results)

        print("Saved run to:", run_dir)
        print("Val metrics:", val_metrics)
        print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
