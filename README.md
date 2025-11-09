# M3prime_1
# M3’ Text→Mouth6 Minimal (ETL内蔵・emoなし)

- 目的: かな化テキストから `mouth_timeline.json` を出力（6口型: close/a/i/u/e/o）
- 時間解像度: `step_ms=40`（=25fps相当）
- 依存: PyTorch 2.x / numpy / scikit-learn / pykakasi / pyyaml / pandas

## 入出力契約
- ETL 入力:
  - `in/transcript.json` … `[{start_ms,end_ms,surface,punct?}, ...]`
  - `in/dlc_landmarks.csv` … DLC出力（列名は configs でマッピング）
- ETL 出力:
  - `in/pose_timeline.json` … `{"timeline":[{"t_ms":..., "mouth6":"a|i|u|e|o|close"}...]}` 
  - `in/train_samples.jsonl` … 1行=1サンプル（かな/kana, T, mouth_steps[]）
- 学習出力: `out/ckpt/best.pt`, `out/logs/events.log`, `out/metrics/metrics.json`
- 推論出力: `out/mouth_timeline.demo.json`

## すぐ動かす
```bash
pip install -r requirements.txt
make etl CSV=in/dlc_landmarks.csv TRANSCRIPT=in/transcript.json
make train
make eval
make infer TEXT="はい、始めます。" AUDIO_MS=1200
