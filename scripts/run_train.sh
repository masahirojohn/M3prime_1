#!/usr/bin/env bash
set -euo pipefail
python m3prime/train.py --config configs/default.yaml
python m3prime/eval.py  --config configs/default.yaml
cat out/metrics/metrics.json
