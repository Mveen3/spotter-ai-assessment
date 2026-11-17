# Freight rate prediction: evidence, plan, and revision

Status: initial plan, critique, full audit, local implementation, model experiments,
and final artifact generation completed. Sample observations remain labeled as samples.

## Completed results

Both paths selected 75% absolute-loss LightGBM plus 25% equipment median-rate
baseline using the declared tuning grid. On the locked 9,523-row Sep–Oct window,
enriched RMSE/MAE are $633.77/$119.42; core RMSE/MAE are $631.04/$113.77. The
quote-times-distance baseline has $711.80/$246.02. Core outperforming enriched on
the locked window did not trigger reselection. Final prediction files pass the
original scorer and preserve every input and template ID.

The first CUDA request fell back to CPU. An explicit CPU rerun, after correcting
backend metadata and adapting to the installed LightGBM API, produced identical
prediction files and locked metrics with unchanged model selections. It took
about 71 seconds, substantially less than the conservative initial estimate.
The complete initial evidence is retained in artifacts/initial_run_evidence.json.
PDF/DOCX reports and the video script are generated from actual saved results.

## Execution update after full audit and user authorization

The user authorized full execution on this PC, superseding the earlier local
runtime restriction for this task. Kaggle and Hugging Face are not used. Local
atomic checkpoints replace the remote checkpoint architecture below.

The full audit confirms 48,000 labeled rows covering January 1–October 31 and
12,000 prediction rows covering November 1–December 31. There are 64 training
cities, eight additional submission cities, and 1,461 submission loads on unseen
routes. Training contains 300 missing and 292 nonpositive weights, plus 374 missing
market indices. Submission contains 165 missing and 145 nonpositive weights, plus
249 missing market indices. No invalid targets or exact duplicate feature rows
were found. Each training city has one consistent supplied coordinate pair.

The frozen date splits are therefore retained. Add a deterministic city-exclusion
stress test after selection, with no reselection based on that diagnostic. Use
XGBoost CUDA as an additional candidate; use LightGBM CPU with six threads to
avoid installing a separate OpenCL/CUDA build. Compare fixed 25/50/75% pairwise
blends of the candidate predictions. The smooth core model uses annual and weekly
Fourier terms; a free trend is omitted to limit unsupported long-range growth.

Store source/data hashes and local completed-task checkpoints. The project is
organized into data/, freight/, scripts/, tests/, docs/, artifacts/, submission/,
and reports/. Publish to https://github.com/Mveen3/spotter-ai-assessment after checks.

## 1. Submission contract

- Preserve the supplied `score.py`.
- Produce exactly 12,000 unique `load_id,predicted_rate` records, joined to the
  provided template by ID, with finite positive predictions.
- Predict the 31 supplied December rows without changing their six input columns.
- Run the supplied scorer and include its `candidate_december.png` in a PDF report.
- Supply reproducible code, dependencies, GitHub run instructions, and a 2–3 minute
  video script. Hidden evaluation labels and its metric are not supplied.
- Original files had root-level hyphenated names, unlike the README's `data/` paths.
  Support both layouts explicitly. The PDF referenced by README is not present.

## 2. Initial proposal, before inspecting values

Audit schemas and labels; split development data into training and validation;
compare a simple mileage baseline with boosted trees; engineer route, equipment,
weight, geography, and calendar features; refit the selected model; predict both
output files and create the report.

## 3. Critical review against the actual supplied samples

| Evidence | Problem with the initial proposal | Revised decision |
| --- | --- | --- |
| First 1,200 labeled rows span only eight January days; byte-spaced samples reach late October. Submission samples span November–December 2025. | A random split and head-only EDA would be optimistic and unrepresentative. | Use date boundaries, expanding historical training windows, and two-month future validation horizons. Reserve September–October as a final untouched evaluation. |
| December input has only pickup, delivery, distance, equipment, weight, and date. | An enriched model would receive inputs it never learned to handle as entirely absent. | Train/select a separate core-feature model, evaluate with exactly the December schema, and use it for the chart. Recover city coordinates only from historical training rows. |
| `quote_signal * distance` has sample median absolute error $33.14, MAE $204.33, RMSE $626.26 on 900 distributed labeled rows. | A generic tree may underperform a simple signal-based estimator; rare large residuals matter. | Include the quote baseline, calibrated signal models, and residual boosting; compare squared and absolute loss. Keep an explicit no-signal ablation. These diagnostic errors are not a backtest. |
| Sample weights include negatives and missing values; market index has missing values. | Blind imputation can hide defects; taking absolute weights invents a correction. | Mark nonpositive physical values missing, preserve missing/invalid indicators, and learn imputations within each training fold. |
| Labels include large values and large deviations from quote estimates. | Automatically deleting extremes could remove genuine expensive loads and bias evaluation. | Exclude only invalid/nonpositive labels from training, report them, and retain all finite positive extremes in primary validation. Compare robust and squared-loss estimators. |
| Supplied coordinates do not reliably match named real-world cities. | Replacing them using public geocoding would silently change the dataset's semantics. | Treat supplied geography as dataset-specific; audit city-coordinate consistency and avoid external joins. |
| No November/December labeled sample has been observed. | Month categories and tree date splits cannot establish learned December seasonality. | Use cyclic and smooth time features, compare a regularized linear core model, and disclose extrapolation and calendar uncertainty. No invented December events or synthetic chart shape. |
| Scorer performs format checks and plotting only. | Passing it cannot demonstrate predictive accuracy. | Report independently computed chronological RMSE, MAE, WAPE, and R²; clearly separate these from unknown hidden-test performance. |

Sample inspection: 1,200 initial labeled rows plus 900 records drawn from five
bounded byte-offset windows; 899 similarly sampled submission rows. These samples
overlap and are not probability samples. Full counts, tails, duplication, drift,
and date ranges remain remote-audit tasks.

## 4. Execution sequence and decision gates

1. **Remote full audit.** Verify columns, row/ID uniqueness, exact template ID set,
   invalid fields, date coverage, duplicates excluding IDs, coordinate consistency,
   unseen cities/routes, numeric drift, and target distributions. Fail on structural
   problems; record cleanable field problems. Do not inspect hidden paths or secrets.
2. **Freeze validation.** If the full audit confirms January–October development:
   tune on Jan–Apr → May–Jun and Jan–Jun → Jul–Aug. Evaluate the frozen selection
   once on Jan–Aug → Sep–Oct. Keep complete days together. Exclude exact repeated
   feature records crossing boundaries from fitting, with counts recorded. An inner
   final-month split within each training period sets boosting iterations; outer
   validation never sets early stopping. Fall back to fixed rounds if an inner
   split is too small. Abort if expected date coverage is absent.
3. **Baseline and feature pipeline.** Mileage/equipment median-rate baseline;
   quote×distance baseline; fold-local imputations and city maps; categorical
   pickup/delivery/route/equipment; distance transforms; weight/route interactions;
   dataset-coordinate geometry; weekday and annual sine/cosine plus bounded trend.
   Exclude `load_id` and any target-derived whole-dataset feature.
4. **Small, purposeful model comparison.** Regularized linear models, LightGBM
   squared-loss and absolute-loss models, and quote-residual boosting. Train both
   enriched and core-only candidates. Prefer GPU when supported; bounded threads
   and CPU fallback are suitable for compact tabular data. No neural network or
   unbounded parameter sweep without evidence it is worthwhile.
5. **Selection.** Predeclare mean dollar RMSE across the two tuning folds as the
   provisional primary criterion because the assessment does not specify one.
   Also report MAE/WAPE/R² and equipment, mileage, month, seen/unseen-route,
   quote-availability, and target-tail slices. Compare a small fixed convex blend
   grid only using tuning predictions. Select core and enriched models independently.
   Disclose RMSE/MAE tradeoffs; do not claim an unknown official metric is optimized.
6. **Locked evaluation.** Run the chosen candidates/blends on Sep–Oct once. Save
   per-row predictions, segment errors, and residual/importance diagnostics. Do
   not tune again on this window. Quote/market provenance is undocumented: using
   supplied prediction-time values is an assumption, not proof of no leakage.
7. **Final refit and outputs.** Refit frozen selections on all valid labeled rows,
   predict submission rows with the enriched path (core fallback if signal missing),
   and the chart with the core path. Join by ID; assert complete coverage, exact
   schemas, finite positive values, and untouched chart inputs. Use scorer unchanged.
8. **Evidence-based deliverables.** Generate report PDF, readable Markdown report,
   and timed video script directly from actual audit and metric artifacts. Include
   the official chart and explicitly label limitations. No placeholder scores.

## 5. Architecture

```text
CSV files -> schema/quality audit -> frozen calendar folds
                                   |
                 fold-fitted cleaning + feature construction
                       /                         \
          enriched signal-aware candidates    core-only candidates
                       \                         /
                   tuning metrics + fixed blend comparison
                                   |
                       frozen Sep–Oct evaluation
                                   |
                        all-history final refit
                       /                         \
             12,000 ID-aligned rates          31 December rates
                       \                         /
                      original scorer -> chart
                                   |
                       report PDF + video script
```

Modules separate data contracts, preprocessing, estimators, evaluation, artifact
generation, and remote orchestration. Store configurations, source/data hashes,
selected candidates, metrics, models, and progress with the run.

## 6. Original remote execution proposal (superseded by authorization above)

Local work is limited to bounded sampling and tiny smoke tests under 30 seconds
inside conda `ml`. No full-data training or inference runs locally. Full work runs
on a private GPU-enabled, internet-enabled Kaggle kernel that clones the committed
GitHub source by HTTPS and mounts registered Kaggle datasets.

Use `/kaggle/working/ckpt/`, atomic writes, configuration/source/data hashes, and
`stage_done` markers. Mirror to a verified private Hugging Face repository after
each stage and during long training loops; pull matching state at startup. Store
progress with elapsed time, stage and total ETA, RAM/GPU information. Enforce an
11.5-hour deadline and a working-directory disk limit below 20 GB. Prediction uses
decreasing batches on OOM; tree training retries with reduced histogram bins and
CPU fallback. Never silently reduce the training population.

Original prerequisites were a GitHub destination, Kaggle dataset/kernel owner,
and private Hugging Face checkpoint repository. The user supplied the GitHub
destination and chose local execution, so no remote compute/checkpoint setup was
needed. Secret files are never included in artifacts or uploads.

## 7. References used for design

- Calendar-duration folds: https://scikit-learn.org/stable/modules/cross_validation
- LightGBM objectives and GPU options: https://lightgbm.readthedocs.io/en/stable/Parameters.html
- Checkpoint downloads: https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download
