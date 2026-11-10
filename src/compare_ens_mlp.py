"""公平对比：PCA + 软投票集成 vs PCA + MLP。

回答"为什么感觉集成不如 MLP 准"这个问题。分两种口径实测：
  A. 标准测试集（论文口径）：10 000 张原图直接输入
  B. 手绘管线（界面口径）：2 000 张经反色/裁切/居中/高斯模糊，模拟用户手绘

两种口径都统计：各自准确率、只有集成对的、只有 MLP 对的、两者都错的，
并做 McNemar 检验判断差异是否显著。
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageFilter
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026


# ---------- 与 web_demo.py 完全一致的手绘预处理 ----------
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


def mcnemar(n01, n10):
    """n01: 集成对 MLP 错；n10: 集成错 MLP 对。返回 (p 值, 检验说明)。"""
    n = n01 + n10
    if n == 0:
        return 1.0, "两次判断完全一致，无需检验"
    return binomtest(min(n01, n10), n, 0.5).pvalue, f"分歧样本 n={n}"


def main():
    z = np.load(ROOT / "data" / "mnist.npz")
    X, y = z["X"], z["y"]
    Xtr_raw, Xte_raw = X[:60000], X[60000:]
    ytr, yte = y[:60000], y[60000:]

    ens = joblib.load(ROOT / "results" / "deploy_model.joblib")
    mlp = joblib.load(ROOT / "results" / "deploy_model_mlp.joblib")
    scaler, pca, n_comp = ens["scaler"], ens["pca"], ens["n_comp"]

    def feat(raw):
        v = raw.astype(np.float32).reshape(len(raw), -1) / 255.0
        return pca.transform(scaler.transform(v))[:, :n_comp]

    out = {}

    # ================= A. 标准测试集（全量 10 000） =================
    print("=" * 62)
    print("A. 标准测试集口径（论文口径，10 000 张原图）")
    print("=" * 62)
    Xte = feat(Xte_raw)
    p_ens = ens["model"].predict(Xte)
    p_mlp = mlp["model"].predict(Xte)
    acc_e = (p_ens == yte).mean()
    acc_m = (p_mlp == yte).mean()
    n01 = int(((p_ens == yte) & (p_mlp != yte)).sum())   # 集成对、MLP 错
    n10 = int(((p_ens != yte) & (p_mlp == yte)).sum())   # 集成错、MLP 对
    pval, note = mcnemar(n01, n10)
    print(f"  集成 {acc_e*100:.2f}%   错 {int((p_ens!=yte).sum())} 张")
    print(f"  MLP  {acc_m*100:.2f}%   错 {int((p_mlp!=yte).sum())} 张")
    print(f"  落差 {(acc_e-acc_m)*100:+.2f} 个百分点（{int(round((acc_e-acc_m)*10000))} 张）")
    print(f"  只有集成对: {n01} 张   只有 MLP 对: {n10} 张   都错: "
          f"{int(((p_ens!=yte)&(p_mlp!=yte)).sum())} 张")
    print(f"  McNemar 检验 p = {pval:.4f}  ({note})")
    out["A_standard"] = {"n": 10000, "acc_ens": float(acc_e), "acc_mlp": float(acc_m),
                         "only_ens": n01, "only_mlp": n10,
                         "both_wrong": int(((p_ens != yte) & (p_mlp != yte)).sum()),
                         "p_value": float(pval)}

    # ================= B. 手绘管线（界面口径） =================
    N_HAND = 2000
    rng = np.random.default_rng(SEED)
    idx = rng.choice(np.arange(10000), N_HAND, replace=False)
    print()
    print("=" * 62)
    print(f"B. 手绘管线口径（界面口径，{N_HAND} 张经反色/裁切/居中/模糊）")
    print("=" * 62)
    t0 = time.time()
    hand = []
    for i in idx:
        img = Image.fromarray((255 - Xte_raw[i]).reshape(28, 28).astype(np.uint8)
                              ).resize((280, 280), Image.BILINEAR)
        std = preprocess_to_mnist(img)
        hand.append(np.asarray(std, dtype=np.float32).reshape(-1))
    hand = np.array(hand)
    print(f"  预处理完成 {time.time()-t0:.1f}s")

    Xh = pca.transform(scaler.transform(hand / 255.0))[:, :n_comp]
    ph_ens = ens["model"].predict(Xh)
    ph_mlp = mlp["model"].predict(Xh)
    truth = yte[idx]
    acc_he = (ph_ens == truth).mean()
    acc_hm = (ph_mlp == truth).mean()
    h01 = int(((ph_ens == truth) & (ph_mlp != truth)).sum())
    h10 = int(((ph_ens != truth) & (ph_mlp == truth)).sum())
    pval2, note2 = mcnemar(h01, h10)
    print(f"  集成 {acc_he*100:.2f}%   错 {int((ph_ens!=truth).sum())} 张")
    print(f"  MLP  {acc_hm*100:.2f}%   错 {int((ph_mlp!=truth).sum())} 张")
    print(f"  落差 {(acc_he-acc_hm)*100:+.2f} 个百分点")
    print(f"  只有集成对: {h01} 张   只有 MLP 对: {h10} 张   都错: "
          f"{int(((ph_ens!=truth)&(ph_mlp!=truth)).sum())} 张")
    print(f"  McNemar 检验 p = {pval2:.4f}  ({note2})")
    out["B_handdrawn"] = {"n": N_HAND, "acc_ens": float(acc_he), "acc_mlp": float(acc_hm),
                          "only_ens": h01, "only_mlp": h10,
                          "both_wrong": int(((ph_ens != truth) & (ph_mlp != truth)).sum()),
                          "p_value": float(pval2)}

    # ================= C. 置信度行为：谁更容易"拿不准" =================
    print()
    print("=" * 62)
    print("C. 置信度行为（阈值 0.60 拒识）")
    print("=" * 62)
    pe = ens["model"].predict_proba(Xh).max(axis=1)
    pm = mlp["model"].predict_proba(Xh).max(axis=1)
    for name, p, pred, acc in (("集成", pe, ph_ens, acc_he), ("MLP", pm, ph_mlp, acc_hm)):
        low = p < 0.60
        acc_lo = (pred[low] == truth[low]).mean() if low.sum() else float("nan")
        print(f"  {name}: 平均置信度 {p.mean()*100:.1f}%  |  "
              f"拒识 {int(low.sum())} 张 ({low.mean()*100:.1f}%)  |  "
              f"低置信样本准确率 {acc_lo*100:.1f}%")
    out["confidence"] = {
        "ens": {"mean": float(pe.mean()), "reject": int((pe < 0.60).sum())},
        "mlp": {"mean": float(pm.mean()), "reject": int((pm < 0.60).sum())},
    }

    with open(ROOT / "results" / "ens_vs_mlp.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print()
    print("已写入 results/ens_vs_mlp.json")


if __name__ == "__main__":
    main()
