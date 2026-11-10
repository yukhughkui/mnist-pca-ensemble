"""Web 演示系统推理性能基准。

通过 HTTP 接口以 MNIST 测试集样本实测端到端时延与准确率，
结果写入 results/web_bench.json 供论文引用，保证数字可复现。
运行前需先启动：python src/web_demo.py
"""
import base64
import io
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
N = 200
URL = "http://127.0.0.1:8765/api/predict"


def post(image_b64, model):
    payload = json.dumps({"image": image_b64, "model": model}).encode()
    req = urllib.request.Request(URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read().decode())


def main():
    z = np.load(ROOT / "data" / "mnist.npz")
    X, y = z["X"], z["y"]
    rng = np.random.default_rng(2026)
    idx = rng.choice(np.arange(60000, 70000), N, replace=False)

    out = {"n": N}
    for model in ("ens", "mlp"):
        ms, ok = [], 0
        for i in idx:
            img = Image.fromarray(
                (255 - X[i]).reshape(28, 28).astype(np.uint8)
            ).resize((280, 280), Image.BILINEAR)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            r = post(b64, model)
            ms.append(r["ms"])
            ok += int(r["pred"] == int(y[i]))
        out[model] = {"acc": ok / N, "ms_mean": float(np.mean(ms)),
                      "ms_p95": float(np.percentile(ms, 95))}
        print(f"{model}: acc={ok / N * 100:.2f}%  mean={np.mean(ms):.1f}ms  "
              f"p95={np.percentile(ms, 95):.1f}ms")

    with open(ROOT / "results" / "web_bench.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("已写入 results/web_bench.json")


if __name__ == "__main__":
    main()
