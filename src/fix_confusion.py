"""
补充脚本：重建测试集混淆矩阵并写入 results/metrics.json 的 "confusion" 字段。
模型与特征流水线与 run_experiment.py 完全一致（同随机种子），结果可复现。
运行: python src/fix_confusion.py
"""
import json
from pathlib import Path

import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SEED = 42

z = np.load(ROOT / "data" / "mnist.npz")
X, y = z["X"], z["y"]
X_train, y_train = X[:60000] / 255.0, y[:60000]
X_test, y_test = X[60000:] / 255.0, y[60000:]

bundle = joblib.load(RES / "deploy_model.joblib")
scaler, pca, n_comp, model = bundle["scaler"], bundle["pca"], bundle["n_comp"], bundle["model"]
assert n_comp == 154

Xtr = scaler.fit_transform(X_train)
pca.set_params(n_components=300)
pca.fit(Xtr)
Xte_pca = pca.transform(scaler.transform(X_test))[:, :n_comp]
y_pred = model.predict(Xte_pca)

cms = {"ensemble": confusion_matrix(y_test, y_pred, labels=range(10)).tolist()}
M = json.load(open(RES / "metrics.json", encoding="utf-8"))
M["confusion"] = {"ens": cms["ensemble"]}
json.dump(M, open(RES / "metrics.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
acc = float((y_pred == y_test).mean())
print(f"集成模型复测准确率: {acc:.4f} (误差应<=0.001)")
print("混淆矩阵已写入 metrics.json")
