#!/usr/bin/env bash
set -euo pipefail

# 0) 前提：smoke_test.sh が repo にあること
python -V || true

# 1) ベースの smoke を一度実行（環境＆既定値の整備）
bash smoke_test.sh || true

# 2) 2×2 の小さなグリッド（値は例：必要に応じて調整）
taus=(0.03 0.06) # 開口のしきい
rhos=(0.25 0.35) # 口幅のしきい

mkdir -p out/grid
sum_csv="out/grid/summary.csv"
echo "tau,rho,f1_macro,timing_mae_ms,switch_per_sec,run_dir" > "$sum_csv"

for t in "${taus[@]}"; do
for r in "${rhos[@]}"; do
run="out/grid/tau${t}_rho${r}"
mkdir -p "$run"

# 2-1) default.yaml を一時パッチ
python - <<PY
import yaml, io, sys
p="configs/default.yaml"
cfg=yaml.safe_load(open(p,"r",encoding="utf-8"))
cfg["paths"]["out_dir"]="${run}"
cfg["tau"]=[${t}, max(${t}*1.8, ${t}+0.01), max(${t}*3.2, ${t}+0.03)]
cfg["rho"]=[${r}, max(${r}+0.05, ${r}*1.2), max(${r}+0.10, ${r}*1.4)]
open(p,"w",encoding="utf-8").write(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
print("patched:", p)
PY

# 2-2) パイプライン実行（ETL→Train(1ep)→Infer→Eval）
python -m etl.dlc_to_pose --config configs/default.yaml
python -m etl.build_trainset --config configs/default.yaml
python -m m3prime.train --config configs/default.yaml || true
if [ -f "${run}/val.jsonl" ]; then
python -m m3prime.infer --config configs/default.yaml --in "${run}/val.jsonl" --out "${run}/pred.jsonl" || true
python -m m3prime.eval --gt "${run}/val.jsonl" --pred "${run}/pred.jsonl" --report "${run}/report.json" || true
fi

# 2-3) 成果物チェック＆集計
if [ -f "${run}/report.json" ]; then
python - <<PY
import json, sys
r=json.load(open("${run}/report.json","r",encoding="utf-8"))
print("{t},{r},{f1},{mae},{sw},{run}".format(
t=${t}, r=${r},
f1=r.get("f1_macro",""),
mae=r.get("timing_mae_ms",""),
sw=r.get("switch_per_sec",""),
run="${run}"
))
PY
else
echo "${t},${r},,,,"
fi
done
done >> "$sum_csv"

# 3) 結果プレビュー
echo "=== Grid Summary (CSV head) ==="
head -n 10 "$sum_csv" || true
echo "=== Grid done ==="
