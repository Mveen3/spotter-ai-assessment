# Video Script — Approximately 2–3 Minutes
# Spotter Freight Rate Assessment
# Speaker: Naveen Mishra (Record with camera on)

Suggested visual sequence: README -> EDA Notebook -> Validation Results -> Code Walkthrough -> December Chart.

---

## 0:00–0:35 — Introduction & Key Data Findings

"Hi everyone, I’m Naveen Mishra. Today I’ll walk you through my end-to-end machine learning solution for the Spotter Freight Rate Prediction challenge.

Our goal is to predict total freight rates for 12,000 loads in November and December 2025, plus forecast a fixed lane across all 31 days of December.

When exploring the 48,000 development loads, I discovered three crucial data findings:
First, the quote_signal column turned out to be an assessment trap. In training, its correlation with rate per mile flipped erratically between plus 0.48, minus 0.45, and zero. In the November–December test set, it becomes pure random Gaussian noise with zero correlation to distance or equipment. Relying on quote signal would cause severe model failure in production, so I deliberately excluded it from model training.
Second, 292 training loads had negative weights. Analyzing their distribution revealed they matched valid positive weights exactly—they were simply negative sign errors. Taking the absolute value recovered the real payload weight rather than throwing it away.
Third, the test set introduces 8 unseen cities and over 1,400 loads on unseen routes, meaning our models needed continuous spatial features rather than memorizing historical routes."

---

## 0:35–1:15 — Validation Strategy & Splitting Approach

"Because freight rates change over time, random k-fold cross-validation is invalid because it leaks future market rates into past training folds.

Instead, I implemented a strict chronological backtest:
- Fold 1 trains on January through April and evaluates on May through June.
- Fold 2 trains on January through June and evaluates on July through August.
- And September through October is reserved as an untouched locked holdout of 9,523 loads.

To ensure zero leakage during tuning, tree iteration counts were early-stopped strictly on the final month inside each training history, completely isolating the evaluation window. All model selection decisions were frozen before the holdout was ever touched."

---

## 1:15–1:55 — Modeling Rationale & Results

"For feature engineering, I created non-linear distance transforms, payload density, Haversine geographic distances, circuity ratios, and annual and weekly Fourier harmonics for smooth seasonal extrapolation.

I benchmarked multiple candidate models: equipment median rate baselines, distance-weighted Ridge regression, LightGBM with L2 squared loss, LightGBM with L1 absolute loss, rate-per-mile models, and XGBoost.

The winning configuration across both tuning folds was a blend of 75% LightGBM L1 and 25% equipment median baseline:
- The L1 objective optimizes absolute error, which makes the trees robust against extreme outlier rates in the 1% upper tail—which reach up to 25,000 dollars.
- The equipment baseline anchors predictions to fundamental physical economics, providing stability for rare locations.

On our locked September–October holdout, this model achieved an RMSE of $634.79 and an MAE of $121.65, easily beating the baseline quote model."

---

## 1:55–2:30 — Code Walkthrough & Deliverables

"Walking briefly through the code:
- `freight/data.py` validates schemas and runs the data quality audit.
- `freight/features.py` cleans payload weights and computes geographic and calendar features.
- `freight/models.py` implements the candidate estimators with inner early stopping.
- `run.py` controls the chronological splits, frozen selection, and final full-history refit.
- An interactive Jupyter notebook in `notebooks/` provides a complete visual walkthrough of the entire analysis.

Both prediction files pass all format checks in the provided `score.py` script. The final December forecast chart displays smooth, realistic weekly commercial freight patterns.

The entire workflow is fully reproducible, packaged with tests, verification scripts, and complete documentation. Thank you!"
