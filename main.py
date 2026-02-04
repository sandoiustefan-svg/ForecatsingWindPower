#!/usr/bin/env python3
"""
Main CLI entrypoint for the ForecastingWindPower project.

ML Pipeline commands:
---------------------
python3 main.py model align        # build aligned dataset
python3 main.py model preprocess   # preprocess aligned dataset
python3 main.py model train        # train model
python3 main.py model test         # run unit tests
"""

import argparse
import subprocess
import sys

def run_backend():
    print("Starting backend...")
    subprocess.run(
        ["uvicorn", "app.main:app", "--reload"],
        cwd="backend",
        check=True,
    )


def run_frontend():
    print("Starting frontend...")
    subprocess.run(
        ["npm", "run", "dev"],
        cwd="frontend",
        check=True,
    )


def run_model_align():
    """
    Step 1: Align raw datasets:
    forecast + nowcast + metadata + power → aligned parquet
    """
    print("Running data alignment...")
    subprocess.run(
        ["python3", "src/features/region_dataset_builder.py"],
        cwd="ml",
        check=True,
    )


def run_model_preprocess():
    """
    Step 2: Preprocess aligned dataset:
    scaling, windowing, splits → model-ready tensors
    """
    print("Running preprocessing...")
    subprocess.run(
        ["python3", "src/features/preprocess.py"],
        cwd="ml",
        check=True,
    )


def run_model_train():
    print("Training model...")
    subprocess.run(
        ["python3", "src/models/train.py"],
        cwd="ml",
        check=True,
    )


def run_model_tests():
    print("Running ML tests...")
    subprocess.run(
        ["pytest", "-q"],
        cwd="ml",
        check=True,
    )

def main():
    parser = argparse.ArgumentParser(
        description="ForecastingWindPower Project CLI"
    )

    subparsers = parser.add_subparsers(dest="command")

    # Backend
    subparsers.add_parser("backend", help="Run backend server")

    # Frontend
    subparsers.add_parser("frontend", help="Run frontend dev server")

    # Model pipeline
    model_parser = subparsers.add_parser("model", help="ML pipeline commands")
    model_sub = model_parser.add_subparsers(dest="action")

    model_sub.add_parser("align", help="Run data alignment pipeline")
    model_sub.add_parser("preprocess", help="Run preprocessing pipeline")
    model_sub.add_parser("train", help="Train the forecasting model")
    model_sub.add_parser("test", help="Run ML unit tests")

    args = parser.parse_args()

    if args.command == "backend":
        run_backend()

    elif args.command == "frontend":
        run_frontend()

    elif args.command == "model":
        if args.action == "align":
            run_model_align()

        elif args.action == "preprocess":
            run_model_preprocess()

        elif args.action == "train":
            run_model_train()

        elif args.action == "test":
            run_model_tests()

        else:
            print("Missing model action.")
            print("Use: align / preprocess / train / test")
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
