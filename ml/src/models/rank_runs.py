# src/models/rank_runs.py
import json
from pathlib import Path

ARTIFACTS = Path("artifacts")  # same as your training script output
OUT_TXT = Path("logs") / "ranking.txt"
OUT_JSON = Path("logs") / "ranking.json"

def main():
    rows = []
    for p in ARTIFACTS.glob("*/metrics.json"):
        arch = p.parent.name
        m = json.loads(p.read_text())
        val = m.get("val", {}) or {}
        test = m.get("test", {}) or {}
        rows.append(
            {
                "arch": arch,
                "val_rmse": val.get("rmse"),
                "val_mae": val.get("mae"),
                "val_loss": val.get("loss"),
                "test_rmse": test.get("rmse"),
                "test_mae": test.get("mae"),
                "test_loss": test.get("loss"),
                "path": str(p),
            }
        )

    if not rows:
        raise SystemExit("No metrics.json found under artifacts/*/metrics.json")

    # Compare across different losses: sort by val_rmse then val_mae
    rows.sort(key=lambda r: (r["val_rmse"] is None, r["val_rmse"], r["val_mae"]))

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text(
        "Top 15 by val_rmse (then val_mae)\n\n"
        + "\n".join(
            f"{i+1:02d}. {r['arch']:35s}  "
            f"val_rmse={r['val_rmse']:.6f}  val_mae={r['val_mae']:.6f}  "
            f"test_rmse={r['test_rmse']:.6f}  test_mae={r['test_mae']:.6f}"
            for i, r in enumerate(rows[:15])
        )
        + "\n"
    )

    OUT_JSON.write_text(json.dumps(rows, indent=2))
    print(OUT_TXT.read_text())
    print(f"\nSaved ranking to: {OUT_TXT}")
    print(f"Saved full ranking JSON to: {OUT_JSON}")

if __name__ == "__main__":
    main()
