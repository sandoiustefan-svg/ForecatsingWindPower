# Machine Learning

This directory contains the machine learning code used for wind power forecasting, including data preprocessing, feature engineering, and model training.  
The ML pipeline is designed to be reproducible, modular, and independent from the API and frontend layers.

## Project Structure

```text
ml/
├── notebooks/      # Exploration and experimentation
├── src/            # Core ML logic
├── config.yaml     # Experiment configuration
├── requirements.txt
└── README.md
└── test
```

## Setup
```bash
cd ForecastingWindPower/ml

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```