import json
from pathlib import Path

import numpy as np
import yaml

from .keras_model import KerasModel


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


def evaluate(model: KerasModel, X, y):
    # returns dict with loss and metrics from compile (here: mse loss, maybe mae if you add it)
    res = model.model.evaluate(X, y, verbose=0, return_dict=True)
    return {k: float(v) for k, v in res.items()}


def main(
    npz_path="src/processed_data/pipeline_data/windows_h61.npz",
    multi_cfg_path="config.yaml",
    artifacts_root="artifacts",
):
    X_train, y_train, X_val, y_val, X_test, y_test = load_npz(npz_path)

    with open(multi_cfg_path, "r") as f:
        multi_cfg = yaml.safe_load(f)

    archs = multi_cfg["architectures"]
    fit_defaults = multi_cfg.get("defaults", {}).get("fit", {"epochs": 30, "batch_size": 128})

    for arch_name, arch_cfg in archs.items():
        print("\n==============================")
        print(f"Training architecture: {arch_name}")
        print("==============================")

        run_dir = Path(artifacts_root) / arch_name
        run_dir.mkdir(parents=True, exist_ok=True)

        run_config_path = run_dir / "config_used.yaml"
        save_yaml(run_config_path, arch_cfg)

        model = KerasModel(config_path=str(run_config_path), name=arch_name)

        model.fit(X_train, y_train, X_val, y_val, **fit_defaults, verbose=2)

        model.save_weights()

        model.plot_learning_curves()

        i = 0
        y_pred = model.predict(X_test[i : i + 1])[0]  # (61,)
        time = np.arange(y_test.shape[1])  # 0..60
        model.plot_predictions(time, y_test[i], y_pred)

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
