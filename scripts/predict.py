"""Regenerate submission files from a trusted local model bundle."""
import argparse
import pickle
from pathlib import Path

from freight.data import read_inputs
from run import write_outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='artifacts/model_bundle.pkl')
    parser.add_argument('--data-dir', default='.')
    parser.add_argument('--output', default='submission')
    args = parser.parse_args()
    # Pickle is appropriate for this locally trained artifact; never load untrusted files.
    with open(args.model, 'rb') as handle:
        bundle = pickle.load(handle)
    frames, _ = read_inputs(args.data_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    submission, december = write_outputs(bundle, frames, output)
    root = Path(args.data_dir).resolve()
    if output != root:
        import shutil
        shutil.copy2(output / 'validation_predictions.csv', root / 'validation_predictions.csv')
        shutil.copy2(output / 'december_predictions.csv', root / 'december_predictions.csv')
        shutil.copy2(output / 'december_predictions.csv', root / 'data/december_chart_inputs.csv')
    print(f'Created {len(submission):,} load predictions and {len(december)} December predictions in {output}')
