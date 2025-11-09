import argparse, yaml, json, os, torch
from m3prime.dataset import MouthDataset, collate, VOCAB
from m3prime.model_tiny import Text2Mouth
from m3prime.metrics import summarize_sample
import numpy as np

def main(cfg):
    dev=cfg["device"] if torch.cuda.is_available() else "cpu"
    out=cfg["paths"]["out_dir"]; os.makedirs(f"{out}/metrics", exist_ok=True)
    step_ms=cfg["step_ms"]

    ds=MouthDataset(cfg["paths"]["train_samples"])
    dl=torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)

    m=Text2Mouth(VOCAB, **cfg["model"]).to(dev)
    m.load_state_dict(torch.load(f"{out}/ckpt/best.pt", map_location=dev))
    m.eval()

    all=[]
    with torch.no_grad():
        for B in dl:
            x=B["x"].to(dev); mask=B["mask"].to(dev); T=B["T_list"]; y=B["y"].squeeze(0).tolist()
            logits=m(x,mask,T).squeeze(0)[:T[0]]
            pred=logits.argmax(-1).cpu().tolist()
            # 欠損(-100)を教師から除外
            y_ids=[t for t in y[:T[0]] if t>=0]
            p_ids=pred[:len(y_ids)]
            all.append(summarize_sample(y_ids, p_ids, step_ms))

    # 集計
    f1=np.mean([a["f1_macro"] for a in all])
    sw=np.mean([a["switch_per_sec"] for a in all])
    mae=np.mean([a["timing_mae_ms"] for a in all])
    metrics={"f1_macro":float(f1),"switch_per_sec":float(sw),"timing_mae_ms":float(mae)}
    # ゲート
    gate=cfg["gate"]; metrics["gate_pass"]= bool(
        (metrics["f1_macro"]>=gate["f1_macro_min"]) and
        (metrics["timing_mae_ms"]<=gate["timing_mae_ms_max"]) and
        (metrics["switch_per_sec"]<=gate["switch_per_sec_max"])
    )
    with open(f"{out}/metrics/metrics.json","w") as f: json.dump(metrics,f,indent=2)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True)
    args=ap.parse_args(); cfg=yaml.safe_load(open(args.config,"r",encoding="utf-8"))
    main(cfg)
