#!/usr/bin/env bash
set -euo pipefail

for EMB in 128 192; do
 for LAY in 2 3; do
  for FFN in 256 384; do
    yq e ".model.emb_dim=$EMB | .model.n_layers=$LAY | .model.ffn_dim=$FFN" configs/default.yaml > /tmp/cfg.yaml
    python m3prime/train.py --config /tmp/cfg.yaml
    python m3prime/eval.py  --config /tmp/cfg.yaml
    echo "---- result (emb=${EMB}, layers=${LAY}, ffn=${FFN}) ----"
    cat out/metrics/metrics.json || true
  done
 done
done
