#!/usr/bin/env bash
set -euo pipefail

echo "=== 0) Environment check & deps ==="
python -V || true
pip -q install -U pip
# NumPy 2系で統一（OpenCVはheadless推奨）
pip -q install "numpy>=2.0,<2.3" "pandas>=2.0" "scikit-learn>=1.6,<1.7" \
"matplotlib>=3.8" "scipy>=1.11" "PyYAML>=6.0" \
"opencv-python-headless==4.12.0.88" "torch==2.3.1" "pykakasi==2.2.1"

python - <<'PY'
import numpy as np, pandas as pd, sklearn, cv2, yaml, sys
print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("pandas:", pd.__version__)
print("sklearn:", sklearn.__version__)
print("OpenCV:", cv2.__version__)
PY

echo "=== 1) Clone repo ==="
# 作業ディレクトリはJulesの既定カレントを想定
if [ ! -d M3prime_1 ]; then
git clone --depth=1 https://github.com/masahirojohn/M3prime_1.git
fi
cd M3prime_1

echo "=== 2) Prepare IO ==="
mkdir -p in out/exp_smoke configs

# ▼ 入力CSVを配置（Julesの入力アーティファクト経由で受け取る想定）
# 例: 事前にアップした CSV を $JULES_INPUT_DIR からコピー
# すでにリポジトリにある場合はこの行は不要です。
if [ -n "${JULES_INPUT_DIR:-}" ] && [ -f "$JULES_INPUT_DIR/1_0_1.csv" ]; then
cp "$JULES_INPUT_DIR/1_0_1.csv" in/1_0_1.csv
fi
# 手元に無い場合はエラーにせず続行（あなたがin/1_0_1.csvを用意していればOK）
if [ ! -f in/1_0_1.csv ]; then
echo "WARN: in/1_0_1.csv が見つかりません。あなたのCSVを in/ に置いてください。"
fi
if [ ! -f in/transcript.json ]; then
echo "WARN: in/transcript.json が見つかりません。空のJSON配列を配置します。"
echo '[
  { "surface": "こん", "start_ms": 50, "end_ms": 150 },
  { "surface": "にち", "start_ms": 150, "end_ms": 250 },
  { "surface": "は", "start_ms": 250, "end_ms": 350, "punct": "。" }
]' > in/transcript.json
fi


echo "=== 3) Patch configs/default.yaml (ensure t_ms) ==="
# default.yaml が存在しない/未設定でも動くように安全パッチ
python - <<'PY'
from pathlib import Path
import yaml, sys, json

cfg_p = Path("configs/default.yaml")
if cfg_p.exists():
    cfg = yaml.safe_load(cfg_p.read_text(encoding="utf-8"))
else:
    cfg = {}

# 安全に既定値を埋める
cfg.setdefault("seed", 42)
cfg.setdefault("device", "cpu")
cfg.setdefault("step_ms", 40)
cfg.setdefault("paths", {})
cfg["paths"].setdefault("in_dir", "in")
cfg["paths"].setdefault("out_dir", "out/exp_smoke")
cfg["paths"].setdefault("transcript", "in/transcript.json")
cfg["paths"].setdefault("dlc_csv", "in/1_0_1.csv")
cfg["paths"].setdefault("pose_timeline", "out/exp_smoke/pose_timeline.json")
cfg["paths"].setdefault("train_samples", "out/exp_smoke/train_samples.jsonl")

cfg.setdefault("dlc_columns", {})
COL = cfg["dlc_columns"]
# t_ms は空だと後続が壊れるので "t_ms" を明示
if not COL.get("t_ms"):
    COL["t_ms"] = "t_ms"
COL.setdefault("frame", "frame")
COL.setdefault("fps", 25)
# 口/目の列名は既存CSVに合わせて最低限を設定（必要に応じて追記）
COL.setdefault("flx", "eye_left_outer_x"); COL.setdefault("fly", "eye_left_outer_y")
COL.setdefault("frx", "eye_right_outer_x"); COL.setdefault("fry", "eye_right_outer_y")
COL.setdefault("ulx", "mouth_outer_upper_x"); COL.setdefault("uly", "mouth_outer_upper_y")
COL.setdefault("llx", "mouth_outer_lower_x"); COL.setdefault("lly", "mouth_outer_lower_y")
COL.setdefault("mlx", "mouth_outer_left_x"); COL.setdefault("mly", "mouth_outer_left_y")
COL.setdefault("mrx", "mouth_outer_right_x"); COL.setdefault("mry", "mouth_outer_right_y")
COL.setdefault("ilx", "mouth_inner_left_x"); COL.setdefault("ily", "mouth_inner_left_y")
COL.setdefault("irx", "mouth_inner_right_x"); COL.setdefault("iry", "mouth_inner_right_y")
COL.setdefault("iux", "mouth_inner_upper_x"); COL.setdefault("iuy", "mouth_inner_upper_y")
COL.setdefault("ilx2","mouth_inner_lower_x"); COL.setdefault("ily2","mouth_inner_lower_y")
COL.setdefault("lk_ul","mouth_outer_upper_likelihood")
COL.setdefault("lk_ll","mouth_outer_lower_likelihood")
COL.setdefault("lk_ml","mouth_outer_left_likelihood")
COL.setdefault("lk_mr","mouth_outer_right_likelihood")

cfg.setdefault("likelihood_threshold", 0.2)
cfg.setdefault("tau", [0.02, 0.05, 0.1])
cfg.setdefault("rho", [0.2, 0.3, 0.4])
cfg.setdefault("median_k", 5)
cfg.setdefault("hys_low_keep", 1)
cfg.setdefault("hys_high_keep", 1)

cfg.setdefault("train", {})
cfg["train"].setdefault("batch_size", 32)
cfg["train"].setdefault("epochs", 1) # ★ スモークなので 1 epoch
cfg["train"].setdefault("lr", 0.002)
cfg["train"].setdefault("weight_decay", 0.0001)
cfg["train"].setdefault("label_smoothing", 0.05)
cfg["train"].setdefault("early_stop_patience", 3)

cfg.setdefault("model", {})
cfg["model"].setdefault("emb_dim", 128)
cfg["model"].setdefault("n_layers", 2)
cfg["model"].setdefault("n_heads", 4)
cfg["model"].setdefault("ffn_dim", 256)
cfg["model"].setdefault("dropout", 0.1)

cfg.setdefault("gate", {})
cfg["gate"].setdefault("f1_macro_min", 0.0) # スモークなのでゲートOFF相当
cfg["gate"].setdefault("timing_mae_ms_max", 9999) # 同上
cfg["gate"].setdefault("switch_per_sec_max", 999) # 同上

cfg_p.parent.mkdir(parents=True, exist_ok=True)
cfg_p.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
print("Wrote:", cfg_p)
PY

echo "=== 4) ETL: DLC CSV -> pose_timeline.json ==="
# あなたのエントリに合わせて、dlc_to_pose.py を --config で実行
python -m etl.dlc_to_pose --config configs/default.yaml

echo "=== 5) ETL: pose_timeline.json -> train/val JSONL ==="
python -m etl.build_trainset --config configs/default.yaml

echo "=== 6) Train (epochs=1) ==="
# m3prime/train.py が --config を受け取る前提
python -m m3prime.train --config configs/default.yaml || true

echo "=== 7) Infer ==="
# train/build で作った検証セットをそのまま推論
if [ -f out/exp_smoke/val.jsonl ]; then
python -m m3prime.infer --config configs/default.yaml \
--in out/exp_smoke/val.jsonl --out out/exp_smoke/pred.jsonl || true
else
echo "WARN: out/exp_smoke/val.jsonl が見つかりません（ETLの出力先スキーマを確認してください）"
fi

echo "=== 8) Eval ==="
if [ -f out/exp_smoke/pred.jsonl ] && [ -f out/exp_smoke/val.jsonl ]; then
python -m m3prime.eval --gt out/exp_smoke/val.jsonl \
--pred out/exp_smoke/pred.jsonl --report out/exp_smoke/report.json || true
else
echo "WARN: 評価に必要なファイルが不足しています。"
fi

echo "=== 9) Quick peek artifacts ==="
ls -lah out/exp_smoke || true
python - <<'PY'
from pathlib import Path, PurePosixPath
p = Path("out/exp_smoke/report.json")
print("report exists:", p.exists())
if p.exists():
    print(p.read_text(encoding="utf-8")[:2000])
else:
    print("no report (train/infer/eval の出力を確認してください)")
for name in ["pose_timeline.json", "train_samples.jsonl", "val.jsonl", "pred.jsonl"]:
    q = Path("out/exp_smoke")/name
    print(f"{name}: {'ok' if q.exists() else 'missing'}")
PY

echo "=== DONE (smoke) ==="
