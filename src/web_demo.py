"""手写数字识别 · Web 交互演示系统（后端）。

用 Python 标准库 http.server 提供本地服务：
  GET  /                 前端页面
  POST /api/predict      接收手绘图片，返回预测结果与十类概率
  GET  /api/sample       随机取一个 MNIST 测试集样本贴到画布上
  GET  /api/info         模型与实验元信息

推理链路与论文完全一致：
  280x280 手绘图 -> 反色与亮度归一 -> 包围盒裁切 -> 等比缩放至 20x20
  -> 质心居中放置于 28x28 -> /255 -> 标准化 -> PCA(154 维) -> 软投票集成

运行：python src/web_demo.py   然后浏览器访问 http://127.0.0.1:8765
"""
import base64
import io
import json
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import joblib

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
RES = ROOT / "results"
PORT = 8765
REJECT_THRESHOLD = 0.60


def preprocess_to_mnist(img):
    """把白纸黑字的手绘大图规范化为 MNIST 风格的 28x28 黑底白字样本。"""
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    arr = np.clip(255.0 - arr, 0, 255)          # 反色：黑底白字
    if arr.max() < 5:
        return None
    arr = arr / arr.max()                        # 亮度归一

    ys, xs = np.nonzero(arr > 0.1)
    if len(xs) == 0:
        return None
    crop = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    h, w = crop.shape
    scale = 20.0 / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    small = Image.fromarray((crop * 255).astype(np.uint8)).resize((nw, nh), Image.BILINEAR)

    a = np.asarray(small, dtype=np.float32) / 255.0
    gy, gx = np.mgrid[0:nh, 0:nw]
    cy = float((a * gy).sum() / a.sum())
    cx = float((a * gx).sum() / a.sum())
    ox = max(0, min(28 - nw, int(round(14 - cx))))
    oy = max(0, min(28 - nh, int(round(14 - cy))))

    out = Image.new("L", (28, 28), 0)
    out.paste(small, (ox, oy))
    return out.filter(ImageFilter.GaussianBlur(0.7))


class Models:
    """懒加载训练产物，避免启动时读取大文件。"""

    def __init__(self):
        self._cache = {}

    def get(self, key):
        if key not in self._cache:
            f = {"ens": "deploy_model.joblib", "mlp": "deploy_model_mlp.joblib"}[key]
            path = RES / f
            if not path.exists():
                raise FileNotFoundError(str(path))
            self._cache[key] = joblib.load(path)
        return self._cache[key]

    def info(self):
        met_path = RES / "metrics.json"
        meta = {"acc": None, "n_comp": None, "latency": None}
        if met_path.exists():
            M = json.load(open(met_path, encoding="utf-8"))
            for r in M["records"]:
                if r["model"] == "PCA + 软投票集成":
                    meta["acc"] = r["accuracy"]
                    meta["latency"] = r["latency_ms_per_sample"]
            meta["n_comp"] = M["pca"]["n95"]
        b = self.get("ens")
        meta["n_comp"] = int(b["n_comp"])
        meta["threshold"] = REJECT_THRESHOLD
        return meta


MODELS = Models()


def b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def decode(data_url):
    raw = base64.b64decode(data_url.split(",", 1)[1])
    return Image.open(io.BytesIO(raw))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB), **kw)

    def log_message(self, fmt, *args):          # 静音访问日志
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/info":
            self._send(MODELS.info())
            return
        if self.path == "/api/sample":
            self._sample()
            return
        return super().do_GET()

    def _sample(self):
        z = np.load(ROOT / "data" / "mnist.npz")
        X, y = z["X"], z["y"]
        i = int(np.random.randint(60000, 70000))         # 测试集区间
        arr = (255 - X[i].reshape(28, 28)).astype(np.uint8)   # 转白纸黑字
        img = Image.fromarray(arr, mode="L").resize((280, 280), Image.NEAREST)
        self._send({"image": b64(img), "label": int(y[i])})

    def do_POST(self):
        if self.path != "/api/predict":
            self._send({"error": "not found"}, 404)
            return
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n).decode("utf-8"))
        key = req.get("model", "ens")
        try:
            b = MODELS.get(key)
        except FileNotFoundError:
            self._send({"error": "模型文件缺失，请先运行 python src/run_experiment.py"})
            return

        try:
            std = preprocess_to_mnist(decode(req["image"]))
        except Exception:
            std = None
        if std is None:
            self._send({"error": "empty", "hint": "画布为空或图片无法解析"})
            return
        v = np.asarray(std, dtype=np.float32).reshape(1, -1) / 255.0
        v = b["pca"].transform(b["scaler"].transform(v))[:, : int(b["n_comp"])]

        t0 = time.perf_counter()
        probs = b["model"].predict_proba(v)[0]
        ms = (time.perf_counter() - t0) * 1000

        # 供前端展示：反色成白纸黑字并放大
        show = Image.fromarray((255 - np.asarray(std)).astype(np.uint8), mode="L") \
            .resize((112, 112), Image.NEAREST)
        self._send({
            "pred": int(np.argmax(probs)),
            "probs": [round(float(p), 5) for p in probs],
            "ms": round(ms, 2),
            "input": b64(show),
            "threshold": REJECT_THRESHOLD,
        })


def main():
    for port in range(PORT, PORT + 20):
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    else:
        raise SystemExit("端口被占用，请关闭后重试")

    url = f"http://127.0.0.1:{port}"
    info = MODELS.info()
    # 预热：提前把两个模型读进内存，避免首次点击时等待
    import threading
    threading.Thread(target=lambda: MODELS.get("mlp"), daemon=True).start()
    print("=" * 58)
    print("  手写数字识别 Web 演示系统已启动")
    print(f"  访问地址：{url}")
    print(f"  模型：PCA({info['n_comp']} 维) + 软投票集成，"
          f"测试集准确率 {info['acc'] * 100:.2f}%")
    print("  关闭此窗口即可停止服务")
    print("=" * 58)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    main()
