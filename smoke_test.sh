#!/usr/bin/env bash
set -euo pipefail

echo "=== 0) Deps (NumPy 2系に統一) ==="
python -V || true
pip -q install -U pip
if [ -f requirements.txt ]; then
  pip -q install -r requirements.txt || true
fi
pip -q install "numpy>=2.0,<2.3" "pandas>=2.0" "scikit-learn>=1.6,<1.7" \
               "matplotlib>=3.8" "scipy>=1.11" "PyYAML>=6.0" \
               "opencv-python-headless==4.12.0.88" jq

python - <<'PY'
import numpy as np, pandas as pd, sklearn, cv2, yaml, sys
print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("pandas:", pd.__version__)
print("sklearn:", sklearn.__version__)
print("OpenCV:", cv2.__version__)
PY

echo "=== 1) Prepare IO ==="
mkdir -p in out/exp_smoke configs
# Jules入力アーティファクト経由で CSV を受け取る場合
if [ -n "${JULES_INPUT_DIR:-}" ]; then
  # よく使う名前を拾う（1_0_1.csv / seg*.csv）
  for f in 1_0_1.csv seg1.csv seg2.csv seg3.csv seg4.csv seg5.csv; do
    if [ -f "$JULES_INPUT_DIR/$f" ]; then cp -f "$JULES_INPUT_DIR/$f" in/; fi
  done
fi
[ -f in/transcript.json ] || echo '{}' > in/transcript.json

echo "=== 2) Patch configs/default.yaml (safe defaults + fixed seed) ==="
python - <<'PY'
from pathlib import Path
import yaml
p=Path("configs/default.yaml")
cfg=yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
cfg.setdefault("seed",42)           # ★ Val分割固定
cfg.setdefault("device","cpu")
cfg.setdefault("step_ms",40)        # 25fps を想定

paths=cfg.setdefault("paths",{})
paths.setdefault("in_dir","in")
paths.setdefault("out_dir","out/exp_smoke")
paths.setdefault("transcript","in/transcript.json")
# dlc_csv は後段の「マージ結果」に応じて上書きする
paths.setdefault("dlc_csv","in/1_0_1.csv")
paths.setdefault("pose_timeline","out/exp_smoke/pose_timeline.json")
paths.setdefault("train_samples","out/exp_smoke/train_samples.jsonl")

COL=cfg.setdefault("dlc_columns",{})
COL["t_ms"]=COL.get("t_ms") or "t_ms"   # ★ 必ずこの列名で扱う
COL.setdefault("frame","frame")
COL.setdefault("fps",25)

# 口/目の列マッピング（足りない時だけ既定を埋める）
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
 "ilx2":"mouth_inner_lower_x","ily":"mouth_inner_lower_y",
 "lk_ul":"mouth_outer_upper_likelihood",
 "lk_ll":"mouth_outer_lower_likelihood",
 "lk_ml":"mouth_outer_left_likelihood",
 "lk_mr":"mouth_outer_right_likelihood",
}.items():
    COL.setdefault(k,v)

cfg.setdefault("likelihood_threshold",0.2)
cfg.setdefault("tau",[0.02,0.05,0.1])
cfg.setdefault("rho",[0.2,0.3,0.4])
cfg.setdefault("median_k",5)
cfg.setdefault("hys_low_keep",1)
cfg.setdefault("hys_high_keep",1)

train=cfg.setdefault("train",{})
train.setdefault("batch_size",32)
train.setdefault("epochs",1)  # smoke
train.setdefault("lr",0.002)
train.setdefault("weight_decay",0.0001)
train.setdefault("label_smoothing",0.05)
train.setdefault("early_stop_patience",3)

model=cfg.setdefault("model",{})
model.setdefault("emb_dim",128); model.setdefault("n_layers",2)
model.setdefault("n_heads",4);   model.setdefault("ffn_dim",256)
model.setdefault("dropout",0.1)

gate=cfg.setdefault("gate",{})
gate.setdefault("f1_macro_min",0.0)
gate.setdefault("timing_mae_ms_max",9999)
gate.setdefault("switch_per_sec_max",999)

p.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding="utf-8")
print("Wrote:", p)
PY

echo "=== 3) MULTI-CSV MERGE (if multiple files exist) ==="
python - <<'PY'
import os, re, json
from pathlib import Path
import pandas as pd, yaml, numpy as np

cfg=yaml.safe_load(open("configs/default.yaml","r",encoding="utf-8"))
in_dir=Path(cfg["paths"]["in_dir"])
dlc_cols=cfg["dlc_columns"]; TCOL=dlc_cols["t_ms"]; FCOL=dlc_cols["frame"]; FPS=dlc_cols.get("fps",25)
step_ms=cfg.get("step_ms",40)

# in/ 下の CSV を列挙（combined.csv は除外）
cands=[p for p in sorted(in_dir.glob("*.csv")) if p.name!="combined.csv"]
if not cands:
    raise SystemExit("ERROR: no CSVs found in ./in")

# 自然順ソート（seg1, seg2, ... を順番に）
def natural_key(s): 
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]
cands=sorted(cands, key=natural_key)

# 1本だけならそのまま使う
if len(cands)==1:
    src=cands[0]
    print(f"[merge] single CSV detected: {src.name}")
    # cfg.paths.dlc_csv を単一CSVに
    cfg["paths"]["dlc_csv"]=str(src.as_posix())
    open("configs/default.yaml","w",encoding="utf-8").write(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
    raise SystemExit(0)

# 複数ある → combined.csv に順次追記（t_ms オフセットを継ぎ足し）
combined=in_dir/"combined.csv"
if combined.exists(): combined.unlink()

offset=0.0
header_written=False
for i,p in enumerate(cands,1):
    df=pd.read_csv(p)
    # t_ms 無ければ frame/fps から生成（ms単位、小数OK）
    if TCOL not in df.columns:
        if FCOL not in df.columns:
            raise SystemExit(f"ERROR: {p.name} has neither {TCOL} nor {FCOL}")
        df[TCOL]=df[FCOL]*1000.0/float(FPS)
    # オフセット継ぎ足し
    df[TCOL]=df[TCOL]+offset
    # 次オフセット＝今回の最大t_ms＋step_ms（フレーム境界をまたがない程度の隙間）
    cur_max=float(np.nanmax(df[TCOL].values)) if len(df) else offset
    offset=cur_max+float(step_ms)
    # 追記
    df.to_csv(combined, mode="a", header=not header_written, index=False)
    header_written=True
    print(f"[merge] appended {p.name} rows={len(df)} next_offset={offset:.1f} ms")

# cfg.paths.dlc_csv を combined に
cfg["paths"]["dlc_csv"]=str(combined.as_posix())
open("configs/default.yaml","w",encoding="utf-8").write(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
print(f"[merge] created combined.csv -> {combined}")
PY

echo "=== 4) Baseline smoke: ETL→Train(1ep)→Infer→Eval ==="
python -m etl.dlc_to_pose     --config configs/default.yaml
python -m etl.build_trainset  --config configs/default.yaml
python -m m3prime.train       --config configs/default.yaml --out out/exp_smoke || true
if [ -f out/exp_smoke/val.jsonl ]; then
  python -m m3prime.infer     --config configs/default.yaml \
         --in out/exp_smoke/val.jsonl --out out/exp_smoke/pred.jsonl || true
  python -m m3prime.eval      --gt out/exp_smoke/val.jsonl \
         --pred out/exp_smoke/pred.jsonl --report out/exp_smoke/report.json || true
fi

echo "=== 5) Quick checks (events / N/K / count consistency) ==="
python - <<'PY'
import json, pathlib
run=pathlib.Path("out/exp_smoke")
pt = run/"pose_timeline.json"
if pt.exists():
    obj=json.loads(pt.read_text(encoding="utf-8"))
    ev=len(obj.get("timeline",[]))
    print(f"[baseline] pose events: {ev}")
else:
    print("[baseline] pose_timeline.json missing")

def read_jsonl(p):
    if not p.exists(): return []
    with p.open("r",encoding="utf-8") as f:
        import json
        return [json.loads(x) for x in f]
V=read_jsonl(run/"val.jsonl"); P=read_jsonl(run/"pred.jsonl")
N=len(V)
labels=set()
for r in V:
    for s in r.get("mouth_steps",[]):
        if isinstance(s, dict):
            lab=s.get("label") or s.get("mouth6")
            if lab: labels.add(lab)
K=len(labels)
print(f"[baseline] N={N}, K={K}")
if P and len(P)!=N:
    raise SystemExit(f"ERROR: count mismatch (val={N}, pred={len(P)})")
PY

echo "=== 6) Quantile init for tau/rho (from merged CSV if present) ==="
readarray -t QVALS < <(python - <<'PY'
import yaml, pandas as pd, numpy as np, json
from pathlib import Path
cfg=yaml.safe_load(open("configs/default.yaml","r",encoding="utf-8"))
csv=cfg["paths"]["dlc_csv"]; COL=cfg["dlc_columns"]; thr=cfg.get("likelihood_threshold",0.0)
df=pd.read_csv(csv)
fw=np.hypot(df[COL["flx"]]-df[COL["frx"]], df[COL["fly"]]-df[COL["fry"]]).replace(0,np.nan)
open_h=np.hypot(df[COL["ulx"]]-df[COL["llx"]], df[COL["uly"]]-df[COL["lly"]])/fw
mouth_w=np.hypot(df[COL["mlx"]]-df[COL["mrx"]], df[COL["mly"]]-df[COL["mry"]])/fw
lk_cols=[COL.get("lk_ul"),COL.get("lk_ll"),COL.get("lk_ml"),COL.get("lk_mr")]
lk_cols=[c for c in lk_cols if c and c in df.columns]
if lk_cols and thr>0:
    lk=df[lk_cols].min(axis=1)
    mask=(lk>=thr) | (lk.isna())
    open_h=open_h[mask]; mouth_w=mouth_w[mask]
oh_q=open_h.quantile([0.30,0.55,0.80]).tolist()
mw_q=mouth_w.quantile([0.30,0.55,0.80]).tolist()
base_tau=float(oh_q[1]); base_rho=float(mw_q[1])
print(json.dumps({"base_tau":base_tau,"base_rho":base_rho}))
PY
)
BASE_TAU=$(echo "${QVALS[0]}" | jq -r .base_tau)
BASE_RHO=$(echo "${QVALS[0]}" | jq -r .base_rho)
echo "base_tau=${BASE_TAU}, base_rho=${BASE_RHO}"

TAU_1=$(python - <<PY
print({:.6f})
PY".format(${BASE_TAU}*0.9)")
TAU_2=$(python - <<PY
print({:.6f})
PY".format(${BASE_TAU}*1.1)")
RHO_1=$(python - <<PY
print({:.6f})
PY".format(${BASE_RHO}*0.9)")
RHO_2=$(python - <<PY
print({:.6f})
PY".format(${BASE_RHO}*1.1)")
echo "grid: tau=[${TAU_1}, ${TAU_2}], rho=[${RHO_1}, ${RHO_2}]"

echo "=== 7) Run 2x2 grid with fixed seed (summary with N/K) ==="
mkdir -p out/grid
SUM="out/grid/summary.csv"
echo "tau,rho,N,K,f1_macro,timing_mae_ms,switch_per_sec,run_dir" > "$SUM"

run_one () {
  local T="$1" R="$2" RUN="out/grid/tau${T}_rho${R}"
  mkdir -p "$RUN"
  python - <<PY
import yaml
p="configs/default.yaml"
cfg=yaml.safe_load(open(p,"r",encoding="utf-8"))
cfg["paths"]["out_dir"]="${RUN}"
t=float("${T}"); r=float("${R}")
cfg["tau"]=[t, max(t*1.8, t+0.01), max(t*3.2, t+0.03)]
cfg["rho"]=[r, max(r+0.05, r*1.2), max(r+0.10, r*1.4)]
open(p,"w",encoding="utf-8").write(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
print("patched:", p)
PY

  python -m etl.dlc_to_pose     --config configs/default.yaml
  python -m etl.build_trainset  --config configs/default.yaml
  python -m m3prime.train       --config configs/default.yaml --out "${RUN}" || true

  if [ -f "${RUN}/val.jsonl" ]; then
    python -m m3prime.infer     --config configs/default.yaml --in "${RUN}/val.jsonl" --out "${RUN}/pred.jsonl" || true
    python -m m3prime.eval      --gt "${RUN}/val.jsonl" --pred "${RUN}/pred.jsonl" \
                                 --report "${RUN}/report.json" || true
  fi

  python - <<PY
import json, pathlib, sys
run=pathlib.Path("${RUN}")
need=["pose_timeline.json","train_samples.jsonl","val.jsonl","pred.jsonl","report.json"]
miss=[f for f in need if not (run/f).exists()]
if miss:
    print("MISSING:", miss); sys.exit(2)

def read_jsonl(p):
    with open(p,"r",encoding="utf-8") as f:
        return [json.loads(x) for x in f]
V=read_jsonl(run/"val.jsonl")
P=read_jsonl(run/"pred.jsonl")
N=len(V)
labels=set()
for r in V:
    for s in r.get("mouth_steps",[]):
        if isinstance(s, dict):
            lab=s.get("label") or s.get("mouth6")
            if lab: labels.add(lab)
K=len(labels)
if len(P)!=N:
    print(f"ERROR: count mismatch (val={N}, pred={len(P)})"); sys.exit(3)

rep=json.loads(open(run/"report.json","r",encoding="utf-8").read())
f1=rep.get("f1_macro","")
mae=rep.get("timing_mae_ms","")
sw =rep.get("switch_per_sec","")
print(f"{${T}},{${R}},{N},{K},{f1},{mae},{sw},{run.as_posix()}")
PY
}

run_one "${TAU_1}" "${RHO_1}" >> "$SUM"
run_one "${TAU_1}" "${RHO_2}" >> "$SUM"
run_one "${TAU_2}" "${RHO_1}" >> "$SUM"
run_one "${TAU_2}" "${RHO_2}" >> "$SUM"

echo "=== 8) Grid Summary (head) ==="
head -n 10 "$SUM" || true

echo "=== DONE ==="
