# ReliefRoute-Resources — Disaster Response Demand Forecasting

**Sector:** Disaster response  
**Task:** Hourly resource-demand regression

Forecasts synthetic demand intensity from an event-impact decay, road disruption, rainfall, calendar patterns, and recent demand. No real disaster or response records are included.

## Data and split

`reliefroute/data.py` contains a deterministic offline simulator. It uses no public or private operational data and does not download a dataset. Values and patterns are illustrative. Reported target units are **relative synthetic demand-index units**; they are generated for this simulator and are not calibrated real-world measurements. The holdout is chronological (first 75% train, last 25% test). Evaluation uses a rolling one-step forecast: actual observed history up to the row is available to the lag features. This is not recursive multi-step evaluation.

## Models and metrics

- **Baseline:** 24-hour seasonal naive (`lag_24`).
- **Stronger model:** histogram gradient-boosting regression fitted on the training period.
- **Metrics:** MAE, RMSE, R², sMAPE, 90th-percentile absolute error, and mean bias. Error summaries are emitted by the CLI; no performance is asserted in advance.

## Limitations

Disaster response needs involve human safety, equity, accessibility, local knowledge, and rapidly changing conditions. This toy series cannot allocate real resources; decisions need accountable responders and direct situational data.

## Run

    python3 -m venv .venv
    . .venv/bin/activate
    python -m pip install -e ".[test]"
    python -m reliefroute.train --seed 42 --samples 1200 --json
    python -m pytest -q

`make setup`, `make train`, and `make test` are shortcuts. Python 3.9 or newer is required.

## Layout

- `reliefroute/data.py`: domain simulator and lag features.
- `reliefroute/split.py`: forward-only split.
- `reliefroute/model.py`: seasonal baseline, stronger model, metrics.
- `reliefroute/train.py`: CLI.
- `tests/`: reproducibility, split, metrics, and CLI tests.

No code license has been selected; default copyright rules apply. Choose a license before public distribution.
