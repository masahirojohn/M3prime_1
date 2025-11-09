#!/usr/bin/env bash
set -euo pipefail
python m3prime/eval.py --config configs/default.yaml
cat out/metrics/metrics.json
