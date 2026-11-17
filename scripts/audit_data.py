"""Run structural validation and save dataset-wide, non-model audit facts."""
import argparse
import json
from pathlib import Path

from freight.data import atomic_json, audit, read_inputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='.')
    parser.add_argument('--output', default='artifacts/audit.json')
    args = parser.parse_args()
    frames, paths = read_inputs(args.data_dir)
    result = audit(frames, paths)
    atomic_json(Path(args.output), result)
    print(json.dumps(result, indent=2))
