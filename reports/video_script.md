# Video script — approximately 2–3 minutes

Keep your camera on. Suggested screen sequence: README → data audit → validation results → code → December chart.

## 0:00–0:30 — Problem and data

Hi, I’m Naveen. This project predicts the total posted freight rate for twelve thousand future loads. I first audited the forty-eight thousand labeled examples before choosing a model. Training covers January through October, while the prediction set covers November and December. That made time-aware validation essential.

The data also contains eight cities absent from training, so I checked transfer to unfamiliar locations rather than assuming every route had already been seen.

## 0:30–1:00 — Quality and validation

I found three hundred missing weights, two hundred ninety-two nonpositive weights, and three hundred seventy-four missing market indices in training. I treat invalid weights as missing, retain indicators, and learn replacements inside each training fold. I keep all positive target extremes in evaluation.

I compare models on May–June and July–August, always training only on earlier dates. September–October is reserved for the frozen final evaluation. Early stopping uses a separate month inside each training period, preventing the evaluation window from controlling training.

## 1:00–1:40 — Models and measured results

Quote signal times distance is a strong baseline, so I compared it with calibration, ridge regression, LightGBM, and XGBoost, including robust-loss and residual models. I also tested a small fixed blend grid using only the tuning windows.

The selected enriched predictor is 25% equipment + 75% lgb_l1. On the locked evaluation, its root mean squared error is 633.77 dollars and its mean absolute error is 119.42 dollars. These are internal backtest results, not a hidden-test score.

The December chart lacks quote signal and market index. I therefore separately train and validate a core model using only information available to that chart.

## 1:40–2:30 — Code and deliverables

In the code, data.py checks schemas and IDs, features.py learns cleaning rules from historical rows, models.py implements the candidates, and run.py controls the splits, frozen selection, and final refit. Tests check unavailable features cannot leak into the core path and that unknown cities are handled.

The submission contains every requested ID. The supplied scorer validates both output files and generates this fixed December chart. Only the date changes; I do not invent special holiday adjustments.

The main limitations are unseen locations, rare expensive loads, limited seasonal history, and undocumented quote-signal provenance. The repository includes the reproducible workflow, complete metrics, report, and prediction files.
