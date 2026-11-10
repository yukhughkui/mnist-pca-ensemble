"""
应用演示：把训练好的模型用于"票据手写数字自动录入"场景。
从 MNIST 测试集中取若干真实手写数字，拼合成一张表格化票据图像，
再按单元格裁切 -> 预处理 -> PCA -> 集成模型识别，输出识别结果与实际值对照图。

运行: python src/application_demo.py
输出: figures/fig13_application.png   results/application_result.json
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import joblib
from PIL import Image, ImageDraw, ImageFont
_FONT = ImageFont.truetype(r"C:\Windows\Fonts\simhei.ttf", 13)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"

bundle = joblib.load(RES / "deploy_model.joblib")
scaler, pca, n_comp, model = bundle["scaler"], bundle["pca"], bundle["n_comp"], bundle["model"]

z = np.load(ROOT / "data" / "mnist.npz")
X, y = z["X"], z["y"]
rng = np.random.RandomState(7)

labels_truth = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
picks = []
for c in labels_truth:
    cand = np.where(y[60000:] == c)[0]
    picks.append(60000 + rng.choice(cand))

# ---- 生成票据画布 ----
CELL, PAD = 56, 14
W = PAD * 2 + CELL * len(picks)
H = CELL + PAD * 2 + 30
canvas = Image.new("L", (W, H), 255)
dr = ImageDraw.Draw(canvas)
dr.text((PAD, 6), "存款金额（手写体）", fill=90, font=_FONT)
for i in range(len(picks) + 1):
    x = PAD + i * CELL
    dr.line([(x, PAD + 16), (x, PAD + 16 + CELL)], fill=170, width=1)
for j in range(2):
    dr.line([(PAD, PAD + 16 + j * CELL),
             (PAD + CELL * len(picks), PAD + 16 + j * CELL)], fill=170, width=1)
Y0 = PAD + 16

for i, idx in enumerate(picks):
    # MNIST 为黑底白字, 票据为白纸黑字: 反色后粘贴, 使画面符合真实票据
    img = Image.fromarray((255 - X[idx]).reshape(28, 28).astype(np.uint8)).resize(
        (40, 40), Image.BILINEAR)
    canvas.paste(img, (PAD + i * CELL + 8, Y0 + 8))

canvas.save(FIG / "ticket_input.png")

# ---- 逐格识别 ----
arr = np.asarray(canvas, dtype=np.float32)
probs_all, preds = [], []
for i in range(len(picks)):
    x0, y0 = PAD + i * CELL, Y0
    # 内缩 3px 裁切, 避开表格线; 反色回模型期望的"黑底白字"分布
    cell = arr[y0 + 3:y0 + CELL - 3, x0 + 3:x0 + CELL - 3]
    crop = Image.fromarray((255 - cell).astype(np.uint8)).resize((28, 28), Image.BILINEAR)
    v = np.asarray(crop, dtype=np.float32).reshape(1, -1) / 255.0
    v = pca.transform(scaler.transform(v))[:, :n_comp]
    p = model.predict_proba(v)[0]
    preds.append(int(np.argmax(p)))
    probs_all.append(float(np.max(p)))

correct = [p == t for p, t in zip(preds, labels_truth)]
print("真值:", labels_truth)
print("预测:", preds)
print("置信度:", [round(p, 3) for p in probs_all])
print("整串准确率: %.3f" % np.mean(correct))

# ---- 可视化 ----
fig, ax = plt.subplots(figsize=(9.2, 2.6))
ax.imshow(np.asarray(canvas), cmap="gray")
for i, (t, p, c) in enumerate(zip(labels_truth, preds, correct)):
    x = PAD + i * CELL
    ax.add_patch(Rectangle((x, Y0), CELL, CELL, fill=False,
                           edgecolor="#2F6B5E" if c else "#B04A3A", lw=1.6))
    ax.text(x + CELL / 2, Y0 + CELL + 4, f"{p}", ha="center", va="top",
            fontsize=13, color="#2F6B5E" if c else "#B04A3A", fontweight="bold")
    ax.text(x + CELL / 2, Y0 - 4, f"{probs_all[i]*100:.0f}%", ha="center", va="bottom",
            fontsize=7, color="#666666")
ax.axis("off")
ax.set_title("票据手写数字的自动识别结果（上：模型置信度，下：识别输出）", fontsize=11)
plt.savefig(FIG / "fig13_application.png", dpi=250, bbox_inches="tight")
plt.close()

json.dump({"truth": labels_truth, "pred": preds,
           "confidence": [round(p, 4) for p in probs_all],
           "exact_match": float(np.mean(correct))},
          open(RES / "application_result.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("已输出 figures/fig13_application.png")
