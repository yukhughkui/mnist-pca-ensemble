"""
基于 PCA 特征降维与多分类器集成的手写数字识别 —— 实验主程序
================================================================
运行方式:  python src/run_experiment.py
输出:      figures/*.png  results/metrics.json  results/summary.md
"""
import os
import json
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report, precision_recall_fscore_support)
from skimage.feature import hog

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"
DATA = ROOT / "data"
for d in (FIG, RES, DATA):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ---------------- 中文字体 ----------------
for cand in ["SimHei", "Microsoft YaHei", "SimSun"]:
    try:
        font_manager.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [cand]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["grid.linestyle"] = "--"

C_MAIN = "#2F6B5E"
C_ACC = "#C08552"
C_GREY = "#8C8C8C"
PALETTE = plt.cm.tab10.colors


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# 1. 数据加载
# ============================================================
def load_mnist():
    cache = DATA / "mnist.npz"
    if cache.exists():
        z = np.load(cache)
        return z["X"], z["y"]
    log("正在获取 MNIST (首次运行需下载约 55MB) ...")
    d = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = d.data.astype(np.float32)
    y = d.target.astype(np.int64)
    np.savez_compressed(cache, X=X, y=y)
    return X, y


X_all, y_all = load_mnist()
log(f"数据规模: X={X_all.shape}, y={y_all.shape}")

# 官方划分: 前 60000 训练, 后 10000 测试
X_train_raw, y_train = X_all[:60000], y_all[:60000]
X_test_raw, y_test = X_all[60000:], y_all[60000:]

# 归一化到 [0,1]
X_train = X_train_raw / 255.0
X_test = X_test_raw / 255.0

# ============================================================
# 2. 数据探索图
# ============================================================
log("绘制数据样例与类别分布 ...")
fig, axes = plt.subplots(6, 10, figsize=(10, 6.2))
idx = {c: np.where(y_train == c)[0][:6] for c in range(10)}
for c in range(10):
    for r in range(6):
        ax = axes[r, c]
        ax.imshow(X_train[idx[c][r]].reshape(28, 28), cmap="gray_r")
        ax.axis("off")
fig.suptitle("MNIST 手写数字样本示例（每列对应一个类别 0–9）", fontsize=13)
plt.tight_layout()
plt.savefig(FIG / "fig01_samples.png")
plt.close()

fig, ax = plt.subplots(figsize=(7.2, 3.4))
cnt_tr = np.bincount(y_train, minlength=10)
cnt_te = np.bincount(y_test, minlength=10)
w = 0.38
ax.bar(np.arange(10) - w / 2, cnt_tr, w, label="训练集 (60 000)", color=C_MAIN)
ax.bar(np.arange(10) + w / 2, cnt_te, w, label="测试集 (10 000)", color=C_ACC)
ax.set_xticks(range(10))
ax.set_xlabel("数字类别")
ax.set_ylabel("样本数")
ax.set_title("训练集与测试集的类别分布")
ax.legend(frameon=False)
plt.savefig(FIG / "fig02_class_dist.png")
plt.close()

# ============================================================
# 3. 特征工程: PCA
# ============================================================
log("PCA 拟合 ...")
t0 = time.perf_counter()
scaler = StandardScaler(with_mean=True, with_std=False)  # 仅中心化, 保留像素稀疏性
X_train_c = scaler.fit_transform(X_train)
X_test_c = scaler.transform(X_test)
pca_full = PCA(n_components=300, svd_solver="randomized", random_state=SEED)
pca_full.fit(X_train_c)
t_pca = time.perf_counter() - t0

cum = np.cumsum(pca_full.explained_variance_ratio_)
n95 = int(np.searchsorted(cum, 0.95) + 1)
n99 = int(np.searchsorted(cum, 0.99) + 1)
log(f"PCA 完成: 95% 方差需 {n95} 维, 99% 需 {n99} 维, 耗时 {t_pca:.1f}s")

fig, ax = plt.subplots(figsize=(6.6, 3.8))
ax.plot(np.arange(1, 301), cum, color=C_MAIN, lw=1.8)
ax.axhline(0.95, color=C_ACC, ls="--", lw=1)
ax.axvline(n95, color=C_ACC, ls=":", lw=1)
ax.scatter([n95], [cum[n95 - 1]], color=C_ACC, zorder=5)
ax.annotate(f"95% 方差\n{n95} 个主成分", xy=(n95, cum[n95 - 1]),
            xytext=(n95 + 25, 0.80), fontsize=10,
            arrowprops=dict(arrowstyle="->", color=C_ACC, lw=0.9))
ax.set_xlabel("主成分个数")
ax.set_ylabel("累计解释方差比")
ax.set_title("PCA 累计解释方差曲线")
plt.savefig(FIG / "fig03_pca_variance.png")
plt.close()

N_COMP = n95
X_train_pca = pca_full.transform(X_train_c)[:, :N_COMP]
X_test_pca = pca_full.transform(X_test_c)[:, :N_COMP]
log(f"PCA 特征维度: {X_train_pca.shape[1]}")

# ============================================================
# 4. HOG 特征
# ============================================================
log("提取 HOG 特征 ...")
t0 = time.perf_counter()


def hog_batch(imgs):
    out = [hog(im.reshape(28, 28), orientations=9, pixels_per_cell=(7, 7),
               cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)
           for im in imgs]
    return np.asarray(out, dtype=np.float32)


X_train_hog = hog_batch(X_train)
X_test_hog = hog_batch(X_test)
log(f"HOG 维度: {X_train_hog.shape[1]}, 耗时 {time.perf_counter()-t0:.1f}s")

# ============================================================
# 5. 模型训练与评估
# ============================================================
def evaluate(name, clf, Xtr, ytr, Xte, yte, note=""):
    t0 = time.perf_counter()
    clf.fit(Xtr, ytr)
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = clf.predict(Xte)
    t_pred = time.perf_counter() - t0

    acc = accuracy_score(yte, y_pred)
    f1m = f1_score(yte, y_pred, average="macro")
    p, r, f, _ = precision_recall_fscore_support(yte, y_pred, labels=range(10))
    rec = {
        "model": name, "note": note, "dim": int(Xtr.shape[1]),
        "train_size": int(Xtr.shape[0]),
        "train_time_s": round(t_fit, 2),
        "predict_time_s": round(t_pred, 2),
        "latency_ms_per_sample": round(t_pred / len(yte) * 1000, 4),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(f1m), 4),
        "per_class_precision": [round(float(v), 4) for v in p],
        "per_class_recall": [round(float(v), 4) for v in r],
        "per_class_f1": [round(float(v), 4) for v in f],
    }
    log(f"  {name:<28} acc={acc:.4f}  f1={f1m:.4f}  fit={t_fit:.1f}s")
    return rec, y_pred


log("开始模型训练 ...")
results = []
preds = {}
cms = {}

# --- 5.1 PCA + kNN
results.append(("PCA + kNN (k=3)",
                *evaluate("PCA + kNN (k=3)",
                          KNeighborsClassifier(n_neighbors=3, metric="euclidean", n_jobs=-1),
                          X_train_pca, y_train, X_test_pca, y_test)))
preds["knn"] = results[-1][-1]

# --- 5.2 PCA + LinearSVC
results.append(("PCA + 线性SVM",
                *evaluate("PCA + 线性SVM", LinearSVC(C=1.0, max_iter=5000, random_state=SEED),
                          X_train_pca, y_train, X_test_pca, y_test)))
preds["lsvc"] = results[-1][-1]

# --- 5.3 PCA + RBF-SVM (训练集抽 10000 以控制核矩阵规模)
rng = np.random.RandomState(SEED)
sub = rng.choice(60000, 10000, replace=False)
results.append(("PCA + RBF-SVM",
                *evaluate("PCA + RBF-SVM", SVC(C=10, gamma="scale", kernel="rbf", random_state=SEED),
                          X_train_pca[sub], y_train[sub], X_test_pca, y_test,
                          note="训练子集 10 000")))
preds["rsvc"] = results[-1][-1]

# --- 5.4 PCA + 随机森林
rf = RandomForestClassifier(n_estimators=300, max_features="sqrt",
                            n_jobs=-1, random_state=SEED)
results.append(("PCA + 随机森林",
                *evaluate("PCA + 随机森林", rf, X_train_pca, y_train, X_test_pca, y_test)))
preds["rf"] = results[-1][-1]

# --- 5.5 PCA + MLP
mlp = MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu",
                    solver="adam", alpha=1e-4, batch_size=256,
                    learning_rate_init=1e-3, max_iter=60, early_stopping=True,
                    n_iter_no_change=8, random_state=SEED)
results.append(("PCA + MLP",
                *evaluate("PCA + MLP", mlp, X_train_pca, y_train, X_test_pca, y_test)))
preds["mlp"] = results[-1][-1]

# --- 5.6 HOG + 线性SVM
results.append(("HOG + 线性SVM",
                *evaluate("HOG + 线性SVM", LinearSVC(C=1.0, max_iter=5000, random_state=SEED),
                          X_train_hog, y_train, X_test_hog, y_test)))
preds["hog_svm"] = results[-1][-1]

# --- 5.7 原始像素 + MLP（对照）
results.append(("原始像素 + MLP",
                *evaluate("原始像素 + MLP",
                          MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu",
                                        solver="adam", alpha=1e-4, batch_size=256,
                                        learning_rate_init=1e-3, max_iter=60,
                                        early_stopping=True, n_iter_no_change=8,
                                        random_state=SEED),
                          X_train, y_train, X_test, y_test)))
preds["raw_mlp"] = results[-1][-1]

# --- 5.8 软投票集成
log("训练软投票集成模型 ...")
ens = VotingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                      n_jobs=-1, random_state=SEED)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu",
                              solver="adam", alpha=1e-4, batch_size=256,
                              learning_rate_init=1e-3, max_iter=60,
                              early_stopping=True, n_iter_no_change=8,
                              random_state=SEED)),
        ("knn", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
    ],
    voting="soft", n_jobs=-1,
)
results.append(("PCA + 软投票集成",
                *evaluate("PCA + 软投票集成", ens, X_train_pca, y_train,
                          X_test_pca, y_test,
                          note="RF + MLP + kNN")))
preds["ens"] = results[-1][-1]

records = [r[1] for r in results]
for k in preds:
    cms[k] = confusion_matrix(y_test, preds[k], labels=range(10)).tolist()

# 最优模型
best = max(records, key=lambda r: r["accuracy"])
log(f"最优模型: {best['model']}  acc={best['accuracy']}")

# ============================================================
# 6. 交叉验证（10 000 子样本, 5 折）
# ============================================================
log("5 折交叉验证 ...")
cv_sub = rng.choice(60000, 10000, replace=False)
Xcv, ycv = X_train_pca[cv_sub], y_train[cv_sub]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_models = {
    "PCA + kNN": KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
    "PCA + 线性SVM": LinearSVC(C=1.0, max_iter=5000, random_state=SEED),
    "PCA + 随机森林": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED),
    "PCA + MLP": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=60,
                               early_stopping=True, n_iter_no_change=8, random_state=SEED),
}
cv_out = {}
for name, m in cv_models.items():
    t0 = time.perf_counter()
    sc = cross_val_score(m, Xcv, ycv, cv=skf, scoring="accuracy", n_jobs=1)
    cv_out[name] = {"scores": [round(float(s), 4) for s in sc],
                    "mean": round(float(sc.mean()), 4),
                    "std": round(float(sc.std()), 4),
                    "time_s": round(time.perf_counter() - t0, 1)}
    log(f"  CV {name:<16} {sc.mean():.4f} ± {sc.std():.4f}")

# ============================================================
# 7. 主成分维度-精度权衡
# ============================================================
log("维度-精度实验 ...")
dims = [20, 40, 60, 80, 100, 120, 154, 200, 300]
dim_sweep = []
Xtr_300 = pca_full.transform(X_train_c)
Xte_300 = pca_full.transform(X_test_c)
for d in dims:
    m = LinearSVC(C=1.0, max_iter=5000, random_state=SEED)
    t0 = time.perf_counter()
    m.fit(Xtr_300[:, :d], y_train)
    fit_t = time.perf_counter() - t0
    a = accuracy_score(y_test, m.predict(Xte_300[:, :d]))
    dim_sweep.append({"dim": d, "accuracy": round(float(a), 4),
                      "train_time_s": round(fit_t, 2)})
    log(f"  dim={d:<4} acc={a:.4f}")

# ============================================================
# 8. 学习曲线
# ============================================================
log("学习曲线 ...")
sizes = [500, 1000, 2000, 5000, 10000, 20000, 40000, 60000]
lc = {"sizes": [], "svm": [], "rf": []}
for s in sizes:
    s_idx = rng.choice(60000, s, replace=False)
    for key, model in (("svm", LinearSVC(C=1.0, max_iter=5000, random_state=SEED)),
                       ("rf", RandomForestClassifier(n_estimators=150, n_jobs=-1,
                                                     random_state=SEED))):
        model.fit(X_train_pca[s_idx], y_train[s_idx])
        lc[key].append(round(float(accuracy_score(y_test, model.predict(X_test_pca))), 4))
    lc["sizes"].append(s)
    log(f"  n={s:<6} svm={lc['svm'][-1]:.4f} rf={lc['rf'][-1]:.4f}")

# ============================================================
# 9. 可视化
# ============================================================
log("绘制 t-SNE ...")
sub_tsne = rng.choice(60000, 5000, replace=False)
ts = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED,
          learning_rate="auto").fit_transform(X_train_pca[sub_tsne])
fig, ax = plt.subplots(figsize=(6.8, 5.6))
for c in range(10):
    m = y_train[sub_tsne] == c
    ax.scatter(ts[m, 0], ts[m, 1], s=6, color=PALETTE[c], label=str(c), alpha=0.75)
ax.legend(markerscale=3, ncol=10, loc="upper center", bbox_to_anchor=(0.5, -0.07),
          frameon=False, fontsize=9)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("PCA 特征的 t-SNE 二维嵌入（5 000 个样本）")
plt.savefig(FIG / "fig04_tsne.png")
plt.close()

# --- 模型对比条形图
names = [r["model"] for r in records]
accs = [r["accuracy"] for r in records]
order = np.argsort(accs)
fig, ax = plt.subplots(figsize=(7.6, 4.0))
bars = ax.barh([names[i] for i in order], [accs[i] for i in order],
               color=[C_ACC if names[i] == best["model"] else C_MAIN for i in order],
               height=0.62)
for b, i in zip(bars, order):
    ax.text(b.get_width() + 0.002, b.get_y() + b.get_height() / 2,
            f"{accs[i]*100:.2f}%", va="center", fontsize=9)
ax.set_xlim(min(accs) - 0.05, 1.0)
ax.set_xlabel("测试集准确率")
ax.set_title("不同特征与分类器组合的识别准确率")
plt.savefig(FIG / "fig05_model_compare.png")
plt.close()

# --- 混淆矩阵
cm = np.array(cms["ens"])
cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(6.0, 5.2))
im = ax.imshow(cm_n, cmap="YlGnBu", vmin=0, vmax=1)
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
ax.set_title("软投票集成模型的归一化混淆矩阵")
for i in range(10):
    for j in range(10):
        if cm_n[i, j] > 0.005:
            ax.text(j, i, f"{cm_n[i,j]*100:.0f}", ha="center", va="center",
                    fontsize=7, color="white" if cm_n[i, j] > 0.55 else "#333333")
ax.grid(False)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="行归一化比例")
plt.savefig(FIG / "fig06_confusion.png")
plt.close()

# --- 各类别 F1 对比
fig, ax = plt.subplots(figsize=(7.6, 3.8))
show = ["PCA + kNN (k=3)", "PCA + 线性SVM", "PCA + 随机森林", "PCA + MLP", "PCA + 软投票集成"]
keys = ["knn", "lsvc", "rf", "mlp", "ens"]
w = 0.16
for i, (nm, k) in enumerate(zip(show, keys)):
    f1s = [r["per_class_f1"] for r in records if r["model"] == nm][0]
    ax.bar(np.arange(10) + (i - 2) * w, f1s, w, label=nm.split("+ ")[-1], color=PALETTE[i])
ax.set_xticks(range(10))
ax.set_xlabel("数字类别"); ax.set_ylabel("F1 分数")
ax.set_ylim(0.85, 1.005)
ax.set_title("主要模型在各类别上的 F1 分数")
ax.legend(frameon=False, ncol=5, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.32))
plt.savefig(FIG / "fig07_per_class_f1.png")
plt.close()

# --- 错误样本可视化
y_ens = np.array(preds["ens"])
wrong = np.where(y_ens != y_test)[0]
log(f"集成模型错误样本数: {len(wrong)}")
pick = rng.choice(wrong, 20, replace=False)
fig, axes = plt.subplots(2, 10, figsize=(11, 2.9))
for ax, i in zip(axes.ravel(), pick):
    ax.imshow(X_test[i].reshape(28, 28), cmap="gray_r")
    ax.set_title(f"真:{y_test[i]} 预:{y_ens[i]}", fontsize=8, color="#B04A3A")
    ax.axis("off")
fig.suptitle("集成模型的典型误分类样本", fontsize=12)
plt.tight_layout()
plt.savefig(FIG / "fig08_errors.png")
plt.close()

# --- 维度-精度曲线
fig, ax1 = plt.subplots(figsize=(6.6, 3.8))
ds = [d["dim"] for d in dim_sweep]
as_ = [d["accuracy"] for d in dim_sweep]
ax1.plot(ds, as_, "o-", color=C_MAIN, lw=1.8, ms=5, label="测试集准确率")
ax1.set_xlabel("PCA 主成分个数"); ax1.set_ylabel("准确率", color=C_MAIN)
ax1.axvline(n95, color=C_ACC, ls="--", lw=1)
ax1.text(n95 + 5, min(as_) + 0.002, f"95% 方差处 ({n95} 维)", color=C_ACC, fontsize=9)
ax2 = ax1.twinx()
ax2.plot(ds, [d["train_time_s"] for d in dim_sweep], "s--", color=C_GREY, lw=1.2, ms=4)
ax2.set_ylabel("训练耗时 (s)", color=C_GREY)
ax2.grid(False)
ax1.set_title("主成分维度对精度与训练耗时的影响（线性 SVM）")
plt.savefig(FIG / "fig09_dim_tradeoff.png")
plt.close()

# --- 学习曲线
fig, ax = plt.subplots(figsize=(6.6, 3.8))
ax.plot(lc["sizes"], lc["svm"], "o-", color=C_MAIN, lw=1.8, ms=5, label="线性 SVM")
ax.plot(lc["sizes"], lc["rf"], "s-", color=C_ACC, lw=1.8, ms=5, label="随机森林")
ax.set_xscale("log")
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xlabel("训练样本数（对数坐标）"); ax.set_ylabel("测试集准确率")
ax.set_title("学习曲线")
ax.legend(frameon=False)
plt.savefig(FIG / "fig10_learning_curve.png")
plt.close()

# --- 交叉验证箱线图
fig, ax = plt.subplots(figsize=(6.4, 3.6))
data_box = [cv_out[k]["scores"] for k in cv_models]
bp = ax.boxplot(data_box, tick_labels=[k.replace("PCA + ", "") for k in cv_models],
                patch_artist=True, widths=0.5)
for patch, c in zip(bp["boxes"], PALETTE):
    patch.set_facecolor(c); patch.set_alpha(0.55)
for med in bp["medians"]:
    med.set_color("#333333")
ax.set_ylabel("5 折交叉验证准确率")
ax.set_title("主要模型 5 折交叉验证结果（10 000 样本）")
plt.savefig(FIG / "fig11_cv_box.png")
plt.close()

# --- 时延对比
fig, ax = plt.subplots(figsize=(6.8, 3.6))
lat = [r["latency_ms_per_sample"] for r in records]
ax.barh(names, lat, color=C_MAIN, height=0.6)
ax.set_xlabel("单样本推理时延 (ms)")
ax.set_title("各模型推理时延对比（CPU, 批量 10 000 样本）")
for i, v in enumerate(lat):
    ax.text(v * 1.02, i, f"{v:.2f}", va="center", fontsize=8)
ax.set_xscale("log")
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
plt.savefig(FIG / "fig12_latency.png")
plt.close()

# ============================================================
# 10. 保存结果
# ============================================================
# 保存可部署的模型（PCA + 软投票集成）供 predict_demo.py 调用
import joblib
joblib.dump({"scaler": scaler, "pca": pca_full, "n_comp": N_COMP, "model": ens},
            RES / "deploy_model.joblib", compress=3)
joblib.dump({"scaler": scaler, "pca": pca_full, "n_comp": N_COMP, "model": mlp},
            RES / "deploy_model_mlp.joblib", compress=3)
log("模型已保存至 results/deploy_model.joblib")

payload = {
    "dataset": {"n_train": 60000, "n_test": 10000, "n_features": 784},
    "pca": {"n95": n95, "n99": n99, "pca_fit_time_s": round(t_pca, 2),
            "explained_var_95": round(float(cum[n95 - 1]), 4)},
    "records": records,
    "cv": cv_out,
    "dim_sweep": dim_sweep,
    "learning_curve": lc,
    "best_model": best["model"],
    "n_errors_ensemble": int(len(wrong)),
    # 混淆矩阵随主流程一并持久化，保证单独运行本脚本即可生成完整指标
    "confusion": {"ens": cms["ens"]},
}
with open(RES / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

log("全部完成。结果写入 results/metrics.json")
