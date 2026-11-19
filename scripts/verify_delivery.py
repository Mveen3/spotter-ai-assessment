"""Check output contracts, provenance, reproducibility, and report packaging."""
import json
from pathlib import Path
import zipfile

import pandas as pd

from freight.data import file_hash, read_inputs
from score import validate_december, validate_predictions


def main():
    root = Path(__file__).resolve().parents[1]
    artifacts = root / 'artifacts'
    frames, paths = read_inputs(root)
    audit = json.loads((artifacts / 'audit.json').read_text())
    for kind, path in paths.items():
        assert file_hash(path) == audit['files'][kind]['sha256'], f'Original input changed: {kind}'
    submission = pd.read_csv(root / 'submission/validation_predictions.csv')
    december = pd.read_csv(root / 'submission/december_predictions.csv')
    validate_predictions(submission)
    validate_december(december)
    assert submission.load_id.tolist() == frames['template'].load_id.tolist()
    pd.testing.assert_frame_equal(december.drop(columns='predicted_rate'), frames['december'].drop(columns='predicted_rate'))
    for filename in ['validation_predictions.csv', 'december_predictions.csv']:
        assert file_hash(root / 'submission' / filename) == file_hash(artifacts / filename)
        assert file_hash(root / filename) == file_hash(artifacts / filename)
    assert file_hash(root / 'data/december_chart_inputs.csv') == file_hash(artifacts / 'december_predictions.csv')
    assert file_hash(root / 'reports/candidate_december.png') == file_hash(artifacts / 'scorer_results/candidate_december.png')
    assert file_hash(root / 'scorer_results/candidate_december.png') == file_hash(artifacts / 'scorer_results/candidate_december.png')
    evidence_path = artifacts / 'initial_run_evidence.json'
    reproducibility = 'not_applicable'
    if evidence_path.exists():
        evidence = json.loads(evidence_path.read_text())
        current = json.loads((artifacts / 'selection.json').read_text())
        for mode in ['core', 'enriched']:
            assert evidence['selection'][mode]['weights'] == current[mode]['weights']
        for filename, digest in evidence['output_sha256'].items():
            assert file_hash(artifacts / filename) == digest, f'Output changed during reproducibility run: {filename}'
        locked = json.loads((artifacts / 'locked_holdout_metrics.json').read_text())
        assert locked == evidence['locked_metrics'], 'Locked metrics changed'
        reproducibility = 'identical_prediction_files_and_locked_metrics'
    manifest = json.loads((artifacts / 'manifest.json').read_text())
    for filename, digest in manifest['source'].items():
        path = root / filename if filename in ['run.py', 'score.py'] else root / 'freight' / filename
        assert file_hash(path) == digest, f'Source changed after run: {filename}'
    with zipfile.ZipFile(root / 'reports/freight_rate_report.docx') as archive:
        assert archive.testzip() is None
        assert 'word/document.xml' in archive.namelist()
        assert len([x for x in archive.namelist() if x.startswith('word/media/')]) == 3
    pdf = root / 'reports/freight_rate_report.pdf'
    assert pdf.read_bytes().startswith(b'%PDF-') and pdf.stat().st_size > 10000
    log = root / 'reports/freight_rate_report.log'
    if log.exists():
        text = log.read_text(errors='replace')
        assert 'Overfull' not in text, 'PDF layout has overfull boxes'
        assert 'LaTeX Error' not in text
    summary = {'submission_rows': len(submission), 'december_rows': len(december),
               'input_hashes_unchanged': True, 'template_order_preserved': True,
               'report_docx_valid': True, 'report_pdf_present': True,
               'reproducibility': reproducibility,
               'submission_sha256': file_hash(root / 'submission/validation_predictions.csv')}
    (artifacts / 'delivery_checks.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
