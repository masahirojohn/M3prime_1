#!/usr/bin/env bash
# ===============================================
# M3prime_1 — Smoke Test (epochs=1) for Jules
# 目的：
# - ETL(csv→pose)→ETL(pose→train/val)→Train(1ep)→Infer→Eval を一気通過
# - 生成物（artifact）5種が揃っているか必須チェック（足りなければ失敗）
# ポリシー：
# - NumPy 2.x 前提（依存競合を避ける）
# - import解決安定化のため python -m で実行
# - YAMLの最低限の既定値を自動パッチ（dlc_columns.t_ms など）
# ===============================================
set -euo pipefail

echo "=== 0) 環境＆依存セット（NumPy 2系に統一） ==="
python -V || true
pip -q install -U pip
# requirements.txt があっても、最終的に NumPy 2.x で上書きして整合を取る
if [ -f requirements.txt ]; then
pip -q install -r requirements.txt || true
fi
pip -q install "numpy>=2.0,<2.3" "pandas>=2.0" "scikit-learn>=1.6,<1.7" \
"matplotlib>=3.8" "scipy>=1.11" "PyYAML>=6.0" \
"opencv-python-headless==4.12.0.88"

python - <<'PY'
import numpy as np, pandas as pd, sklearn, cv2, yaml, sys
print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("pandas:", pd.__version__)
print("sklearn:", sklearn.__version__)
print("OpenCV:", cv2.__version__)
PY

echo "=== 1) 入出力ディレクトリの準備 ==="
mkdir -p in out/exp_smoke configs

# Julesの入力アーティファクトから CSV を受け取る場合の慣習的パス（環境により存在しないことも）
# あれば in/ にコピー、無ければスキップ。手元で in/1_0_1.csv が既にあるならそれでOK。
if [ -n "${JULES_INPUT_DIR:-}" ] && [ -f "$JULES_INPUT_DIR/1_0_1.csv" ]; then
cp "$JULES_INPUT_DIR/1_0_1.csv" in/1_0_1.csv
fi

# transcript は無くても落ちないようダミーを作る
cat << 'EOF' > in/transcript.json
[
  {"start_ms": 0, "end_ms": 1500, "surface": "こんにちは", "punct": "。"},
  {"start_ms": 1501, "end_ms": 5000, "surface": "これはスモークテストです", "punct": "。"}
]
EOF

# CSV の存在をソフトチェック（無くても続行するが後工程で生成物が無ければ失敗扱い）
if [ ! -f in/1_0_1.csv ]; then
echo "WARN: in/1_0_1.csv が見つかりません。Jules入力アーティファ-クト等で配置してください。"
fi

echo "=== 2) configs/default.yaml を安全パッチ（最低限の既定値を埋める） ==="
python - <<'PY'
from pathlib import Path
import yaml
p=Path("configs/default.yaml")
cfg=yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
cfg.setdefault("seed",42); cfg.setdefault("device","cpu"); cfg.setdefault("step_ms",40)

paths=cfg.setdefault("paths",{})
paths.setdefault("in_dir","in")
paths.setdefault("out_dir","out/exp_smoke")
paths.setdefault("transcript","in/transcript.json")
paths.setdefault("dlc_csv","in/1_0_1.csv")
paths.setdefault("pose_timeline","out/exp_smoke/pose_timeline.json")
paths.setdefault("train_samples","out/exp_smoke/train_samples.jsonl")
paths.setdefault("val_samples","out/exp_smoke/val.jsonl")

COL=cfg.setdefault("dlc_columns",{})
# ★ t_ms を空にすると壊れるので、必ず "t_ms" にする（CSVに無ければ後段で自動生成）
COL["t_ms"]=COL.get("t_ms") or "t_ms"
COL.setdefault("frame","frame"); COL.setdefault("fps",25)

# DLC列マッピングの既定（足りない場合だけ埋める）
for k,v in {
"flx":"eye_left_outer_x","fly":"eye_left_outer_y",
"frx":"eye_right_outer_x","fry":"eye_right_outer_y",
"ulx":"mouth_outer_upper_x","uly":"mouth_outer_upper_y",
"llx":"mouth_outer_lower_x","lly":"mouth_outer_lower_y",
"mlx":"mouth_outer_left_x","mly":"mouth_outer_left_y",
"mrx":"mouth_outer_right_x","mry":"mouth_outer_right_y",
"ilx":"mouth_inner_left_x","ily":"mouth_inner_left_y",
"irx":"mouth_inner_right_x","iry":"mouth_inner_right_y",
"iux":"mouth_inner_upper_x","iuy":"mouth_inner_upper_y",
"ilx2":"mouth_inner_lower_x","ily2":"mouth_inner_lower_y",
"lk_ul":"mouth_outer_upper_likelihood",
"lk_ll":"mouth_outer_lower_likelihood",
"lk_ml":"mouth_outer_left_likelihood",
"lk_mr":"mouth_outer_right_likelihood",
}.items():
  COL.setdefault(k,v)

# ETL/学習の既定
cfg.setdefault("likelihood_threshold",0.2)
cfg.setdefault("tau",[0.02,0.05,0.1])
cfg.setdefault("rho",[0.2,0.3,0.4])
cfg.setdefault("median_k",5)
cfg.setdefault("hys_low_keep",1)
cfg.setdefault("hys_high_keep",1)

train=cfg.setdefault("train",{})
train.setdefault("batch_size",32)
train.setdefault("epochs",1) # ★ smoke：1 epoch
train.setdefault("lr",0.002)
train.setdefault("weight_decay",0.0001)
train.setdefault("label_smoothing",0.05)
train.setdefault("early_stop_patience",3)

model=cfg.setdefault("model",{})
model.setdefault("emb_dim",128); model.setdefault("n_layers",2)
model.setdefault("n_heads",4); model.setdefault("ffn_dim",256)
model.setdefault("dropout",0.1)

gate=cfg.setdefault("gate",{})
gate.setdefault("f1_macro_min",0.0) # smoke では事実上OFF
gate.setdefault("timing_mae_ms_max",9999)
gate.setdefault("switch_per_sec_max",999)

p.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding="utf-8")
print("Wrote:", p)
PY

echo "=== 3) ETL-1: DLC CSV → pose_timeline.json ==="
python -m etl.dlc_to_pose --config configs/default.yaml

echo "=== 4) ETL-2: pose_timeline.json → train/val JSONL ==="
python -m etl.build_trainset --config configs/default.yaml

echo "=== 5) Train (epochs=1) ==="
# 実装の -h に合わせて最低限の引数だけ指定
python -m m3prime.train --config configs/default.yaml || true

echo "=== 6) Infer ==="
if [ -f out/exp_smoke/val.jsonl ]; then
python -m m3prime.infer --config configs/default.yaml \
--in out/exp_smoke/val.jsonl --out out/exp_smoke/pred.jsonl || true
else
echo "WARN: val.jsonl が見つからないため infer をスキップ"
fi

echo "=== 7) Eval ==="
if [ -f out/exp_smoke/pred.jsonl ] && [ -f out/exp_smoke/val.jsonl ]; then
python -m m3prime.eval --config configs/default.yaml \
--gt out/exp_smoke/val.jsonl --pred out/exp_smoke/pred.jsonl \
--report out/exp_smoke/report.json || true
else
echo "WARN: eval をスキップ"
fi

echo "=== 8) 生成物チェック（揃ってなければ失敗扱い） ==="
missing=0
for f in pose_timeline.json train_samples.jsonl val.jsonl pred.jsonl report.json; do
if [ ! -f "out/exp_smoke/$f" ]; then
echo "MISSING: out/exp_smoke/$f"; missing=1
else
echo "OK: out/exp_smoke/$f"
fi
done
[ "$missing" -eq 0 ] || { echo "ERROR: smoke artifacts missing"; exit 2; }

echo "=== 9) report の先頭プレビュー ==="
head -c 2000 out/exp_smoke/report.json || true

echo "=== DONE (smoke) ==="
