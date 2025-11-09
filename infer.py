import argparse, yaml, json, torch, time, unicodedata, os
from m3prime.dataset import VOCAB
from m3prime.model_tiny import Text2Mouth

MOUTHS = ["close","a","i","u","e","o"]

def to_kana(s):
    try:
        from pykakasi import kakasi
        _k=kakasi(); _k.setMode("J","H"); _k.setMode("K","H"); _k.setMode("H","H"); conv=_k.getConverter()
        return conv.do(s.strip())
    except:
        return unicodedata.normalize("NFKC", s)

C2I = {c:i+1 for i,c in enumerate(
    list("ぁあぃいぅうぇえぉおかがきぎくぐけげこご"
         "さざしじすずせぜそぞただちぢっつづてでとど"
         "なにぬねのはばぱひびぴふぶぷへべぺほぼぽ"
         "まみむめもやゃゆゅよょらりるれろわゐゑをんー、。！？.!?\n 　"))}
def text_to_ids(s, max_len=160):
    ids=[C2I.get(c,0) for c in s[:max_len]]; return ids, len(ids)

def write_mouth_tl(tl, path, step_ms):
    obj={"meta":{"version":"1.0.0","generator":"m3prime","created_at":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "time_unit":"ms","step_ms":step_ms},"timeline":tl}
    with open(path,"w",encoding="utf-8") as f: json.dump(obj,f,ensure_ascii=False,indent=2)

def main(cfg, text, audio_ms):
    dev=cfg["device"] if torch.cuda.is_available() else "cpu"; step_ms=cfg["step_ms"]; out=cfg["paths"]["out_dir"]
    os.makedirs(out, exist_ok=True)
    m=Text2Mouth(VOCAB, **cfg["model"]).to(dev)
    m.load_state_dict(torch.load(f"{out}/ckpt/best.pt", map_location=dev))
    m.eval()

    kana=to_kana(text); x_ids,L = text_to_ids(kana)
    x=torch.tensor([x_ids]).long().to(dev); mask=torch.zeros_like(x).bool(); mask[0,:L]=1
    T=max(1, round(audio_ms/step_ms))
    with torch.no_grad():
        logits=m(x,mask,[T]).squeeze(0)
        ids=logits.argmax(-1).cpu().tolist()

    # RLE→タイムライン
    timeline=[]; cur=ids[0]; run=1; t=0
    for k in ids[1:]:
        if k==cur: run+=1
        else:
            timeline.append({"t_ms": int(t), "mouth": MOUTHS[cur]})
            t += run*step_ms; cur=k; run=1
    timeline.append({"t_ms": int(t), "mouth": MOUTHS[cur]})
    write_mouth_tl(timeline, f"{out}/mouth_timeline.demo.json", step_ms)
    print(f"Wrote: {out}/mouth_timeline.demo.json")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--text",required=True)
    ap.add_argument("--audio_ms",type=int,required=True)
    a=ap.parse_args()
    cfg=yaml.safe_load(open(a.config,"r",encoding="utf-8"))
    main(cfg, a.text, a.audio_ms)
