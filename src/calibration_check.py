"""补充两问：
1) 置信度校准：模型"说有 80% 把握"时，实际真对了多少？
2) 集成的三个成员在手绘口径下各自掉多少？谁拖了后腿？
"""
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026


def preprocess_to_mnist(img):
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    arr = np.clip(255.0 - arr, 0, 255)
    if arr.max() < 5:
        return None
    arr = arr / arr.max()
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


def main():
    z = np.load(ROOT / "data" / "mnist.npz")
    X, y = z["X"], z["y"]
    Xte_raw, yte = X[60000:], y[60000:]

    ens = joblib.load(ROOT / "results" / "deploy_model.joblib")
    mlp = joblib.load(ROOT / "results" / "deploy_model_mlp.joblib")
    scaler, pca, n_comp = ens["scaler"], ens["pca"], ens["n_comp"]
    model = ens["model"]

    def feat(raw):
        return pca.transform(scaler.transform(raw.astype(np.float32).reshape(len(raw), -1) / 255.0))[:, :n_comp]

    rng = np.random.default_rng(SEED)
    idx = rng.choice(np.arange(10000), 2000, replace=False)

    # ---------- 手绘管线 ----------
    hand = np.array([np.asarray(preprocess_to_mnist(
        Image.fromarray((255 - Xte_raw[i]).reshape(28, 28).astype(np.uint8)
                        ).resize((280, 280), Image.BILINEAR)), dtype=np.float32).reshape(-1)
        for i in idx])
    Xstd, Xhand = feat(Xte_raw[idx]), feat(hand)
    truth = yte[idx]

    out = {}

    # ---------- 1) 校准曲线 ----------
    print("=" * 60)
    print("1. 置信度校准（手绘口径, n=2000）")
    print("   分箱: 模型声称的把握区间 -> 该区间内实际答对的比例")
    print("=" * 60)
    bins = [(0.0, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.01)]
    labels = ["<60%", "60-80%", "80-90%", "90-95%", ">95%"]
    cal = {}
    for name, mdl, Xq in (("集成", model, Xhand), ("MLP", mlp["model"], Xhand)):
        pr = mdl.predict_proba(Xq)
        conf, pred = pr.max(axis=1), pr.argmax(axis=1)
        correct = (pred == truth).astype(float)
        row = []
        print(f"\n  [{name}]")
        print(f"    {'声称把握':<10}{'样本数':>8}{'实际正确率':>12}{'偏差':>10}")
        for (lo, hi), lab in zip(bins, labels):
            sel = (conf >= lo) & (conf < hi)
            if sel.sum() < 5:
                row.append(None)
                continue
            emp = correct[sel].mean()
            mid = conf[sel].mean()
            row.append({"bin": lab, "n": int(sel.sum()),
                        "claimed": float(mid), "actual": float(emp)})
            print(f"    {lab:<10}{int(sel.sum()):>8}{emp*100:>11.1f}%{(emp-mid)*100:>+9.1f}pp")
        cal[name] = row
    out["calibration"] = cal

    # ---------- 2) 三个成员在两种口径下的表现 ----------
    print()
    print("=" * 60)
    print("2. 集成三成员：标准口径 vs 手绘口径（谁掉得多？）")
    print("=" * 60)
    members = []
    for est in model.estimators_:
        members.append(type(est).__name__)
    print(f"  成员: {members}")
    print(f"\n  {'成员/整体':<16}{'标准口径':>12}{'手绘口径':>12}{'落差':>12}")
    member_acc = {}
    for name, est in zip(["随机森林", "MLP", "kNN"], model.estimators_):
        a1 = (est.predict(Xstd) == truth).mean()
        a2 = (est.predict(Xhand) == truth).mean()
        member_acc[name] = {"standard": float(a1), "handdrawn": float(a2),
                            "drop_pp": float((a2 - a1) * 100)}
        print(f"  {name:<16}{a1*100:>11.2f}%{a2*100:>11.2f}%{(a2-a1)*100:>+11.2f}pp")
    for name, mdl in (("集成(整体)", model), ("单独MLP", mlp["model"])):
        a1 = (mdl.predict(Xstd) == truth).mean()
        a2 = (mdl.predict(Xhand) == truth).mean()
        member_acc[name] = {"standard": float(a1), "handdrawn": float(a2),
                            "drop_pp": float((a2 - a1) * 100)}
        print(f"  {name:<16}{a1*100:>11.2f}%{a2*100:>11.2f}%{(a2-a1)*100:>+11.2f}pp")
    out["members"] = member_acc

    with open(ROOT / "results" / "calibration_check.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n已写入 results/calibration_check.json")


if __name__ == "__main__":
    main()
