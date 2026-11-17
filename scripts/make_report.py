"""Generate evidence-based Markdown, DOCX, PDF, charts, and a short video script."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor


def load(folder, filename):
    return json.loads((folder / filename).read_text())


def money(value):
    return f'${value:,.2f}'


def blend_label(weights):
    return ' + '.join(f'{weight:.0%} {name}' for name, weight in weights.items())


def esc(value):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
                    '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(c, c) for c in str(value)).replace('–', '--').replace('—', '---').replace('×', r'$\times$').replace('→', r'$\rightarrow$')


def prose_tex(text):
    """Allow line breaks in source paths and clickable documentation URLs."""
    parts, offset = [], 0
    for match in re.finditer(r'https?://[^\s)]+|(?:[A-Za-z_][\w.-]*/)+[\w.-]+', text):
        token = match.group().rstrip('.,;')
        parts.append(esc(text[offset:match.start()]))
        parts.append((r'\url{' if token.startswith('http') else r'\path{') + token + '}')
        offset = match.start() + len(token)
    parts.append(esc(text[offset:]))
    return ''.join(parts)


def main(args):
    artifacts, report_dir, submission_dir = Path(args.artifacts), Path(args.reports), Path(args.submission)
    report_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)
    audit = load(artifacts, 'audit.json')
    metrics = load(artifacts, 'locked_holdout_metrics.json')
    selections = load(artifacts, 'selection.json')
    stress = load(artifacts, 'unseen_city_stress.json')
    complete = load(artifacts, 'run_complete.json')
    manifest = load(artifacts, 'manifest.json')
    slices = load(artifacts, 'locked_holdout_slices.json')
    tuning = pd.read_csv(artifacts / 'tuning_metrics.csv')
    holdout = pd.read_csv(artifacts / 'locked_holdout_predictions.csv')
    december = pd.read_csv(artifacts / 'december_predictions.csv')
    for name in ['validation_predictions.csv', 'december_predictions.csv']:
        shutil.copy2(artifacts / name, submission_dir / name)
    shutil.copy2(artifacts / 'scorer_results/candidate_december.png', report_dir / 'candidate_december.png')

    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    y, pred = holdout.posted_rate.to_numpy(), holdout.enriched_prediction.to_numpy()
    axes[0].hexbin(y, pred, gridsize=45, bins='log', cmap='Blues', mincnt=1)
    lim = max(float(y.max()), float(pred.max()))
    axes[0].plot([0, lim], [0, lim], '--', color='#CF6B3B', linewidth=1)
    axes[0].set(xlabel='Observed rate ($)', ylabel='Predicted rate ($)', title='Locked Sep–Oct: all targets retained')
    absolute = np.sort(np.abs(y - pred))
    axes[1].plot(absolute, np.arange(1, len(absolute) + 1) / len(absolute), color='#064A56')
    axes[1].set(xlabel='Absolute error ($), logarithmic scale', ylabel='Fraction of loads', title='Error distribution including the tail', xscale='log')
    axes[1].grid(alpha=.2)
    figure.savefig(report_dir / 'holdout_diagnostics.png', dpi=180)
    plt.close(figure)

    figure, ax = plt.subplots(figsize=(10, 4), layout='constrained')
    mean = tuning.groupby(['mode', 'model']).rmse.mean().unstack(0)
    mean.plot.bar(ax=ax, color=['#72A6A6', '#064A56'], rot=30)
    ax.set(ylabel='Mean chronological tuning RMSE ($)', xlabel='', title='Model comparison before locked evaluation')
    figure.savefig(report_dir / 'model_comparison.png', dpi=180)
    plt.close(figure)

    doc = Document()
    doc.styles['Normal'].font.name = 'Calibri'
    doc.styles['Normal'].font.size = Pt(10)
    doc.sections[0].header.paragraphs[0].text = 'SPOTTER  /  FREIGHT RATE ASSESSMENT'
    doc.add_heading('Freight Rate Prediction', 0)
    doc.add_paragraph('Validation design, measured results, and reproducible submission', style='Subtitle')
    md = ['# Freight Rate Prediction', '', 'Validation design, measured results, and reproducible submission', '']
    tex = [r'\documentclass[10pt]{article}', r'\usepackage[margin=0.85in]{geometry}',
           r'\usepackage[T1]{fontenc}', r'\usepackage{lmodern}', r'\usepackage{graphicx,booktabs,longtable,hyperref,xcolor,xurl}',
           r'\hypersetup{colorlinks=true,urlcolor=teal,linkcolor=teal,pdftitle={Freight Rate Prediction},pdfauthor={Naveen Mishra}}', r'\setlength{\parindent}{0pt}',
           r'\setlength{\parskip}{6pt}', r'\title{Freight Rate Prediction\\\large Validation design, measured results, and reproducible submission}',
           r'\author{Naveen Mishra}', r'\date{17 November 2026}', r'\begin{document}', r'\maketitle']

    def section(title):
        if title.startswith('7.'):
            doc.add_page_break()
            tex.append(r'\clearpage')
        doc.add_heading(title, 1)
        md.extend(['## ' + title, ''])
        tex.append(r'\section*{' + esc(title) + '}')

    def paragraph(text):
        doc.add_paragraph(text)
        md.extend([text, ''])
        tex.append(prose_tex(text) + '\n')

    def table(headers, rows):
        t = doc.add_table(rows=1, cols=len(headers))
        t.style = 'Light Shading Accent 1'
        for cell, text in zip(t.rows[0].cells, headers):
            cell.text = str(text)
        for values in rows:
            for cell, value in zip(t.add_row().cells, values):
                cell.text = str(value)
        md.extend(['| ' + ' | '.join(map(str, headers)) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'])
        md.extend('| ' + ' | '.join(map(str, values)) + ' |' for values in rows)
        md.append('')
        tex.append(r'\begin{center}\small\begin{tabular}{' + 'l' * len(headers) + r'}\toprule')
        tex.append(' & '.join(map(esc, headers)) + r' \\ \midrule')
        for values in rows:
            tex.append(' & '.join(map(esc, values)) + r' \\')
        tex.append(r'\bottomrule\end{tabular}\end{center}')

    def picture(filename, caption):
        doc.add_picture(str(report_dir / filename), width=Inches(6.2))
        doc.add_paragraph(caption, style='Caption')
        md.extend([f'![{caption}]({filename})', ''])
        tex.append(r'\begin{center}\includegraphics[width=\linewidth]{' + filename + r'}\\{\small ' + esc(caption) + r'}\end{center}')

    section('1. Result and submission contract')
    paragraph(f'The frozen enriched predictor achieved RMSE {money(metrics["enriched"]["rmse"])} and MAE '
              f'{money(metrics["enriched"]["mae"])} on {metrics["enriched"]["rows"]:,} held-out September–October loads. '
              'These are internal chronological evaluation results, not the unknown Spotter hidden-test score. '
              'The final outputs contain all 12,000 requested IDs and all 31 fixed December dates. The unmodified supplied scorer accepted both files.')
    paragraph('The submission uses the enriched model where quote signal is available. The December chart uses a separately trained and selected core model '
              'because its input omits quote signal, market index, and coordinates. This information difference is evaluated explicitly rather than hidden by filling privileged fields with arbitrary values.')

    section('2. Data exploration and quality decisions')
    table(['Property', 'Development', 'Prediction set'], [
        ['Rows', '48,000', '12,000'], ['Date coverage', 'Jan–Oct 2025', 'Nov–Dec 2025'],
        ['Cities', audit['train']['unique_cities'], audit['validation']['unique_cities']],
        ['Directed routes', audit['train']['unique_routes'], audit['validation']['unique_routes']],
        ['Missing weights', audit['train']['missing']['weight'], audit['validation']['missing']['weight']],
        ['Nonpositive weights', audit['train']['invalid_nonpositive_or_infinite']['weight'], audit['validation']['invalid_nonpositive_or_infinite']['weight']],
        ['Missing market indices', audit['train']['missing']['market_index'], audit['validation']['missing']['market_index']],
        ['Duplicate feature rows', audit['train']['feature_duplicates'], audit['validation']['feature_duplicates']],
    ])
    paragraph('All 48,000 development targets are finite and positive. Missing and nonpositive physical features become missing values with explicit indicators; '
              'imputation medians are fitted only on historical training rows. Negative weights are not converted to absolute values. No finite positive target is removed or winsorized. '
              'The target range is $57.22–$25,533.00; large errors must therefore be assessed rather than discarded.')
    paragraph('There are eight unseen prediction cities: ' + ', '.join(audit['submission_shift']['unseen_cities']) + '. '
              f'{audit["submission_shift"]["unseen_route_rows"]:,} prediction rows use routes absent from development. Categorical handling supports unknown values, '
              'and supplied numerical coordinates support geographic transfer. Historical city-coordinate pairs are internally consistent; external geocoding is not used because these are dataset-specific coordinates.')
    paragraph('The simple quote-signal-times-distance diagnostic has a median absolute error of '
              f'{money(audit["quote_diagnostic_not_backtest"]["median_absolute_error"])} but RMSE '
              f'{money(audit["quote_diagnostic_not_backtest"]["rmse"])} across development. This gap motivates strong signal baselines, residual learning, '
              'and comparisons of absolute and squared loss. This whole-development descriptive diagnostic is not a validation score.')

    section('3. Plan critique and validation architecture')
    paragraph('The initial generic baseline-plus-boosting plan was revised after inspecting actual date coverage and the December schema. '
              'A random split would mix future and past loads; a single enriched model would also be evaluated on inputs absent from the chart. '
              'The revised architecture separates enriched and core paths, uses complete calendar periods, and reserves a final untouched period. '
              'The initial sample, critique, full audit update, and runtime authorization are retained in docs/PLAN.md.')
    table(['Purpose', 'Training period', 'Evaluation period', 'Evaluation rows'], [
        ['Tune fold 1', 'Jan–Apr', 'May–Jun', int(tuning[tuning.fold == 'may_jun'].iloc[0]['rows'])],
        ['Tune fold 2', 'Jan–Jun', 'Jul–Aug', int(tuning[tuning.fold == 'jul_aug'].iloc[0]['rows'])],
        ['Locked evaluation', 'Jan–Aug', 'Sep–Oct', metrics['enriched']['rows']],
        ['Submission refit', 'Jan–Oct', 'Nov–Dec (unlabeled)', '12,000'],
    ])
    paragraph('Each boosting fit first selects its iteration count on the last month inside that fit’s training history, then refits on the full training history. '
              'Outer tuning or locked evaluation labels never set early-stopping rounds. Category vocabularies, city maps, medians, scalers, and encoders are fitted '
              'inside the corresponding training partition. Exact feature duplicates crossing a boundary are purged from fitting; none were found. Load IDs are never model features.')
    paragraph('Because the assessment does not specify the hidden ranking metric, the provisional primary criterion is the unweighted mean of the two tuning-fold RMSE values in dollars. '
              'MAE, WAPE, R-squared, and bias are also recorded. The blend grid contains only 25%, 50%, and 75% weights for each candidate pair, plus individual models. '
              'Selections are written before September–October predictions are evaluated. The final window is not used for model or blend reselection.')

    section('4. Features and model comparison')
    paragraph('Features include pickup and delivery categories, directed route, equipment, cleaned distance and weight, distance transforms, weight per mile, '
              'coordinate deltas and distance, and missing/invalid indicators. Calendar features include weekly and annual sine/cosine terms, weekend and month-boundary flags. '
              'The enriched path adds quote signal, quote times distance, market index, and simple interactions. The core path reconstructs coordinates solely from its fitted historical city map '
              'and ignores all extra input fields, even when they are supplied by the caller.')
    paragraph('Candidates include an equipment-specific median rate-per-mile baseline; quote times distance and affine quote calibration; a regularized, distance-weighted ridge rate-per-mile model; '
              'LightGBM with squared and absolute loss; quote-residual LightGBM; core rate-per-mile LightGBM; and XGBoost. '
              'The rate-per-mile models use squared-distance weights so their squared-error objective corresponds to dollar squared error. '
              'No unconstrained hyperparameter search, external weather data, unverified holiday effects, or target encoding is used.')
    grouped = tuning.groupby(['mode', 'model'])[['rmse', 'mae']].mean().reset_index()
    table(['Input mode / model', 'Mean RMSE', 'Mean MAE'], [
        [f'{row["mode"]} / {row["model"]}', money(row['rmse']), money(row['mae'])]
        for row in grouped.sort_values(['mode', 'rmse']).to_dict('records')
    ])
    paragraph('Frozen enriched selection: ' + blend_label(selections['enriched']['weights']) + '. '
              'Frozen core selection: ' + blend_label(selections['core']['weights']) + '. '
              'These are the best configurations in the declared candidate/blend search under the tuning criterion, not a claim of globally optimal accuracy.')
    picture('model_comparison.png', 'Mean tuning RMSE for individual candidates; selected blends are recorded separately in artifacts/selection.json.')

    section('5. Locked evaluation and transfer diagnostics')
    table(['Predictor', 'RMSE', 'MAE', 'WAPE', 'R-squared'], [
        [name, money(m['rmse']), money(m['mae']), f'{m["wape_percent"]:.2f}%', f'{m["r2"]:.4f}']
        for name, m in metrics.items()
    ])
    picture('holdout_diagnostics.png', 'Locked evaluation diagnostics. Extreme observed rates remain in both plots and all headline metrics.')
    rows = [s for s in slices['enriched'] if s['dimension'] in ['equipment', 'route_seen', 'target_tail']]
    table(['Segment', 'Loads', 'RMSE', 'MAE'], [[s['segment'], s['rows'], money(s['rmse']), money(s['mae'])] for s in rows])
    paragraph('A separate transfer stress test removes all training routes touching eight deterministically selected cities, then scores future loads touching those cities. '
              f'It retains {stress["training_rows"]:,} training rows and evaluates {stress["validation_rows"]:,} future rows. '
              f'Enriched RMSE is {money(stress["enriched"]["rmse"])}; core RMSE is {money(stress["core"]["rmse"])}. '
              'This is a sensitivity diagnostic, not an additional selection fold or an estimate of performance for the specific eight unseen submission cities.')

    section('6. Fixed December chart')
    paragraph('The chart below is generated by the supplied score.py without modifications. All inputs are fixed to Lexington → Fort Wayne, '
              '360 miles, Dry Van, and 32,000 lb; only the date varies across December 2025. Original input CSVs remain unchanged. '
              f'Predictions range from {money(december.predicted_rate.min())} to {money(december.predicted_rate.max())}. '
              'The same frozen core-model recipe evaluated above is refitted on all labeled history before generating these values.')
    picture('candidate_december.png', 'Required fixed December prediction chart, generated by the original assessment scorer.')
    paragraph('December has no labeled development observations. Smooth cyclic features permit extrapolation, but the chart is not proof that December seasonality was learned correctly. '
              'Core validation measures the information-restricted problem across many loads; it does not provide ground truth for this exact fixed lane. '
              'The chart is a point prediction, not a confidence interval or a fabricated holiday scenario.')

    section('7. Reproduction, architecture, and limitations')
    paragraph('freight/data.py validates schemas and data quality; freight/features.py owns fitted cleaning and features; freight/models.py implements candidates and inner early stopping; '
              'freight/evaluation.py scores folds, selects blends, and computes slices; run.py freezes the experiment sequence and writes predictions. '
              'scripts/predict.py regenerates both prediction files from the trusted local model bundle. scripts/make_report.py generates this report and the video script from saved evidence.')
    paragraph('Run in the existing conda ml environment. Execute python run.py --device cpu --threads 6, then python -m scripts.make_report. '
              'The default CUDA request has a CPU fallback; the actual backend is recorded. Checkpoints use atomic replacement, stage-completion markers, '
              'source/configuration/data hashes, and trusted local pickle files. Matching completed tasks can resume. No Kaggle or Hugging Face service is required for this local workflow.')
    paragraph(f'The recorded modeling pipeline took {complete["elapsed_seconds"]:.1f} seconds in this environment, excluding development, report creation, and setup. '
              'Earlier runtime estimates were conservative; this compact search and early stopping were substantially faster. '
              'GPU hardware was visible, but the installed XGBoost runtime fell back to CPU during the initial CUDA attempt. '
              'The finalized reproducible run uses CPU and six threads. Exact library versions and data/source hashes are saved in artifacts/manifest.json.')
    paragraph('Limitations: the hidden test labels and ranking metric are unavailable; there is only one year of partial history; future city and market distributions differ; '
              'rare high-rate loads dominate squared error; quote and market feature provenance is not documented; and no prediction intervals are claimed. '
              'Using quote signal assumes it is available at quote time, as it is in the provided prediction file. This assumption requires business confirmation before deployment. '
              'No score improvement on the hidden test set can be asserted.')
    paragraph('Submission files: submission/validation_predictions.csv and submission/december_predictions.csv. Report: reports/freight_rate_report.pdf or .docx. '
              'Video narration: reports/video_script.md. Repository: https://github.com/Mveen3/spotter-ai-assessment. '
              'The scorer validates output contracts and plots rates; its success is distinct from model accuracy.')
    paragraph('Method references: scikit-learn chronological validation documentation (https://scikit-learn.org/stable/modules/cross_validation.html); '
              'LightGBM parameters (https://lightgbm.readthedocs.io/en/stable/Parameters.html); '
              'XGBoost Python API (https://xgboost.readthedocs.io/en/stable/python/python_api.html).')
    doc.save(report_dir / 'freight_rate_report.docx')
    (report_dir / 'freight_rate_report.md').write_text('\n'.join(md))
    tex.append(r'\end{document}')
    (report_dir / 'freight_rate_report.tex').write_text('\n'.join(tex))
    for _ in range(2):
        subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'freight_rate_report.tex'],
                       cwd=report_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)

    script = f'''# Video script — approximately 2–3 minutes

Keep your camera on. Suggested screen sequence: README → data audit → validation results → code → December chart.

## 0:00–0:30 — Problem and data

Hi, I’m Naveen. This project predicts the total posted freight rate for twelve thousand future loads. I first audited the forty-eight thousand labeled examples before choosing a model. Training covers January through October, while the prediction set covers November and December. That made time-aware validation essential.

The data also contains eight cities absent from training, so I checked transfer to unfamiliar locations rather than assuming every route had already been seen.

## 0:30–1:00 — Quality and validation

I found three hundred missing weights, two hundred ninety-two nonpositive weights, and three hundred seventy-four missing market indices in training. I treat invalid weights as missing, retain indicators, and learn replacements inside each training fold. I keep all positive target extremes in evaluation.

I compare models on May–June and July–August, always training only on earlier dates. September–October is reserved for the frozen final evaluation. Early stopping uses a separate month inside each training period, preventing the evaluation window from controlling training.

## 1:00–1:40 — Models and measured results

Quote signal times distance is a strong baseline, so I compared it with calibration, ridge regression, LightGBM, and XGBoost, including robust-loss and residual models. I also tested a small fixed blend grid using only the tuning windows.

The selected enriched predictor is {blend_label(selections['enriched']['weights'])}. On the locked evaluation, its root mean squared error is {metrics['enriched']['rmse']:.2f} dollars and its mean absolute error is {metrics['enriched']['mae']:.2f} dollars. These are internal backtest results, not a hidden-test score.

The December chart lacks quote signal and market index. I therefore separately train and validate a core model using only information available to that chart.

## 1:40–2:30 — Code and deliverables

In the code, data.py checks schemas and IDs, features.py learns cleaning rules from historical rows, models.py implements the candidates, and run.py controls the splits, frozen selection, and final refit. Tests check unavailable features cannot leak into the core path and that unknown cities are handled.

The submission contains every requested ID. The supplied scorer validates both output files and generates this fixed December chart. Only the date changes; I do not invent special holiday adjustments.

The main limitations are unseen locations, rare expensive loads, limited seasonal history, and undocumented quote-signal provenance. The repository includes the reproducible workflow, complete metrics, report, and prediction files.
'''
    (report_dir / 'video_script.md').write_text(script)
    print(f'Generated PDF, DOCX, Markdown, charts, video script, and submission exports in {report_dir} and {submission_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', default='artifacts')
    parser.add_argument('--reports', default='reports')
    parser.add_argument('--submission', default='submission')
    main(parser.parse_args())
