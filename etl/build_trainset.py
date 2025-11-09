import argparse, json, yaml, unicodedata, os
from pathlib import Path

MOUTHS = ["close","a","i","u","e","o"]
M2ID = {m:i for i,m in enumerate(MOUTHS)}

# かな化（pykakasi）: 依存を減らした最小実装（必要なら置換）
try:
    from pykakasi import kakasi
    _k=kakasi(); _k.setMode("J","H"); _k.setMode("K","H"); _k.setMode("H","H"); conv=_k.getConverter()
    def to_kana(s): 
        try: return conv.do(s.strip())
        except: return unicodedata.normalize("NFKC", s)
except:
    def to_kana(s): return unicodedata.normalize("NFKC", s)

def load_json(p): 
    with open(p,"r",encoding="utf-8") as f: return json.load(f)

def chunk_transcript(words, min_ms=300, max_ms=800):
    cur=[]; start=None; end=None
    for w in words:
        if start is None: start=w["start_ms"]
        cur.append(w); end=w["end_ms"]; dur=end-start
        punct=w.get("punct","")
        if dur>=min_ms and (dur>=max_ms or punct in ["。","！","？",".","!","?","\n"]):
            yield cur; cur=[]; start=None; end=None
    if cur: yield cur

def resample_steps(mouth_events, t0, t1, step_ms):
    ev=sorted(mouth_events, key=lambda x:x["t_ms"])
    ev = ev + ([{"t_ms":10**12, "mouth6": ev[-1]["mouth6"]}] if ev else [{"t_ms":10**12,"mouth6":"close"}])
    def mouth_at(t):
        prev=ev[0]["mouth6"]
        for e in ev[1:]:
            if t < e["t_ms"]: return prev
            prev=e["mouth6"]
        return prev
    T=max(1, round((t1-t0)/step_ms)); out=[]
    for i in range(T):
        out.append(M2ID.get(mouth_at(t0 + i*step_ms),0))
    return out

def main(cfg):
    paths=cfg["paths"]; step_ms=cfg["step_ms"]
    os.makedirs(Path(paths["train_samples"]).parent, exist_ok=True)
    words = load_json(paths["transcript"])
    pose  = load_json(paths["pose_timeline"])
    mouth_events = pose["timeline"]

    cnt=0
    with open(paths["train_samples"],"w",encoding="utf-8") as f:
        for ch in chunk_transcript(words):
            t0=ch[0]["start_ms"]; t1=ch[-1]["end_ms"]
            kana = to_kana("".join([w.get("surface","")+w.get("punct","") for w in ch]))
            steps = resample_steps(mouth_events, t0, t1, step_ms)
            f.write(json.dumps({"t0_ms":t0,"t1_ms":t1,"kana":kana,"T":len(steps),"mouth_steps":steps},ensure_ascii=False)+"\n"); cnt+=1
    print(f"Wrote samples: {cnt} -> {paths['train_samples']}")

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    with open(args.config,"r",encoding="utf-8") as f: cfg=yaml.safe_load(f)
    main(cfg)
