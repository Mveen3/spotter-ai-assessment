# Freight Rate Prediction · Spotter Assessment

A complete local workflow: audit, chronological model comparison, locked evaluation,
final predictions, and an evidence-based report. The [plan and critique](docs/PLAN.md)
record the decisions made before modeling.

## Submission

- [12,000 load predictions (root)](validation_predictions.csv) · [submission copy](submission/validation_predictions.csv)
- [31 fixed December predictions](data/december_chart_inputs.csv) · [submission copy](submission/december_predictions.csv)
- [Interactive EDA & Modeling Notebook](notebooks/exploratory_data_analysis.ipynb)
- [PDF report](reports/freight_rate_report.pdf) · [editable DOCX](reports/freight_rate_report.docx) · [readable report](reports/freight_rate_report.md)
- [Required December chart](scorer_results/candidate_december.png) · [report chart](reports/candidate_december.png)
- [2–3 minute video script](reports/video_script.md) — record with camera on

The supplied `score.py` is unchanged. It checks CSV contracts and generates the
chart; **it does not calculate hidden-test accuracy**.

## Measured results

Selections use May–June and July–August chronological backtests. The following
results use 9,523 September–October loads, with no model reselection:

| Predictor | RMSE ($) | MAE ($) | WAPE | R² |
|---|---:|---:|---:|---:|
| Selected enriched model | 633.77 | 119.42 | 4.99% | 0.8275 |
| Selected core-only model | 631.04 | 113.77 | 4.76% | 0.8290 |
| Quote × distance baseline | 711.80 | 246.02 | 10.28% | 0.7824 |

Both selected recipes blend **75% absolute-loss LightGBM and 25% equipment-based
median rate-per-mile baseline**, fitted separately for their available inputs.
The enriched model predicts submission loads; the core model predicts December,
whose input lacks quote signal, market index, and coordinates. Missing/invalid
quote signals trigger the core fallback. The core model's better locked result
is reported without changing the preselected enriched submission recipe.

These are internal backtests, not an official Spotter score. The hidden ranking
metric is unspecified; mean tuning-fold dollar RMSE is the declared criterion.

## Project layout

```text
.
├── README.md                 # Results, setup, commands
├── requirements.txt          # Supported dependency ranges
├── requirements-lock.txt     # Verified package versions
├── validation_predictions.csv# 12,000 final load rate predictions
├── score.py                  # Unmodified supplied scorer
├── run.py                    # Complete experiment entry point
├── data/                     # Original CSVs and completed december_chart_inputs.csv
├── docs/                     # Plan, requirements, and setup instructions
├── freight/                  # Data, features, models, evaluation, checkpoints
├── notebooks/                # Interactive EDA and modeling Jupyter notebook
├── scripts/                  # Audit, inference, reporting, verification
├── tests/                    # Leakage, model, and split checks
├── artifacts/                # Audit, metrics, predictions, manifest, local models
│   └── checkpoints/          # Resumable state, not committed
├── scorer_results/           # Default scorer output (candidate_december.png)
├── submission/               # Final CSVs ready to submit
└── reports/                  # PDF, DOCX, Markdown, charts, video script
```

The loader supports both supplied hyphenated filenames and the underscore names
in the original instructions, in the root or `data/`. No machine-specific paths
are embedded in the implementation.

## Setup and complete run

Run all Python commands inside the existing Miniconda environment `ml`:

```bash
conda activate ml
OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 python run.py --device cpu --threads 6
python -m scripts.make_report
python -m scripts.verify_delivery
```

The verified environment uses Python 3.12. All libraries were already available;
no installation was needed. Exact directly used versions are in
`requirements-lock.txt`. If dependencies are missing, first try:

```bash
conda install numpy pandas scikit-learn lightgbm xgboost matplotlib threadpoolctl python-docx
```

For only unavailable packages, next try `conda install -c conda-forge <package>`.
Use `python -m pip install <package>` only when unavailable in both Conda sources;
in Jupyter, prefer `%pip` for that last resort. Do not create a virtualenv.

PDF rebuilding requires `pdflatex` with geometry, lmodern, graphicx, booktabs,
longtable, hyperref, xcolor, and xurl. Generated PDF and DOCX files are already included.

The measured modeling workflow took approximately **1–2 minutes**, excluding
development, setup, and report generation, with peak process memory below 1 GB.
The initial 20–45 minute estimate was conservative. Timings vary by environment.

`--device cuda` requests XGBoost GPU acceleration. The installed runtime fell back
to CPU despite the visible RTX 3050; the final reference run explicitly uses CPU.
LightGBM uses six CPU threads. GPU errors retry smaller histograms and CPU;
prediction retries smaller batches on OOM. No Kaggle/Hugging Face service is used.

Atomic checkpoints are keyed by source, data, configuration, and library versions.
Rerunning an identical command resumes matching tasks. `artifacts/progress.json`
records progress, timing estimates, peak RAM, and GPU memory. Early ETA estimates
can be optimistic. The local model bundle and bulky checkpoints are not committed;
the short training command regenerates them. Load pickle files only from trusted
sources.

## Additional commands

```bash
# Full audit without training
python -m scripts.audit_data

# Regenerate outputs from the trained bundle
python -m scripts.predict --model artifacts/model_bundle.pkl --output submission

# Run the unmodified scorer (original assessment instruction)
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv

# Run the unmodified scorer (submission folder layout)
python score.py --predictions submission/validation_predictions.csv \
  --december-predictions submission/december_predictions.csv --output-dir reports

# Re-run or rebuild the interactive EDA notebook
python -m scripts.build_eda_notebook

# Run small regression tests
python -m unittest discover -s tests -v
```

## Validation and architecture

1. Audit all 48,000 labeled and 12,000 prediction rows, schemas, IDs, dates,
   physical values, repeated features, cities, routes, and supplied coordinates.
2. Tune on Jan–Apr → May–Jun and Jan–Jun → Jul–Aug. Early stopping uses the last
   month inside each training history, followed by a full-history refit. Learned
   preprocessing and category vocabularies remain inside each training partition.
3. Compare equipment/quote baselines, affine calibration, weighted ridge,
   LightGBM squared/absolute loss, residual/rate models, and XGBoost. Select among
   individual models and fixed 25/50/75% pairwise blends by mean fold RMSE.
4. Freeze before Jan–Aug → Sep–Oct evaluation. Report RMSE, MAE, WAPE, R², bias,
   and equipment/month/distance/route/city/tail slices.
5. Run deterministic city-exclusion stress tests without reselection.
6. Refit on all valid labels, align predictions by ID, and run the supplied scorer.

`artifacts/initial_run_evidence.json` preserves selections, scores, and prediction
hashes from the initial CUDA-requested/CPU-fallback run. Delivery verification
checks that the explicit CPU rerun produced identical files and locked metrics.

## Findings and limitations

- Training spans January–October; prediction spans November–December 2025.
- Training has 300 missing weights, 292 nonpositive weights, and 374 missing
  market indices. Invalid weights become missing with indicators and fold-fitted
  imputation; their sign is not silently reversed.
- All targets are finite and positive. Extremes up to $25,533 remain in evaluation
  and heavily influence RMSE.
- Prediction contains eight unseen cities and 1,461 unseen-route rows. Categories
  support unknown values and numerical geography supports transfer.
- Coordinates are internally consistent but dataset-specific. External geocoding
  is not used to overwrite their meaning.
- The core path ignores privileged fields and derives geography from historical
  city maps. December seasonality is extrapolated; there is no labeled December
  or exact-lane December ground truth.
- Quote signal is supplied for prediction, but its provenance and availability
  before the realized rate are undocumented. Deployment requires confirmation.
  Core-only results provide the no-signal comparison.
- Hidden labels and the official ranking metric are unavailable. Passing the
  scorer certifies structure, not accuracy. No globally optimal model, hidden-score,
  or uncertainty-band claims are made.

See the [report](reports/freight_rate_report.pdf) for full evidence and references.
