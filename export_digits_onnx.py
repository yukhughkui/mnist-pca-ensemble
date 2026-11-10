# -*- coding: utf-8 -*-
"""把 deploy_model_mlp.joblib（scaler + PCA 截断 + MLP）合成单个 ONNX 并验证"""
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

bundle = joblib.load(r'results/deploy_model_mlp.joblib')
scaler, pca_full, n_comp, mlp = (bundle['scaler'], bundle['pca'],
                                 int(bundle['n_comp']), bundle['model'])
print('n_comp =', n_comp, '| PCA full dims =', pca_full.n_components_)

# 构造截断 PCA：transform 数学上等于 full PCA 输出的前 n_comp 列
pca = PCA(n_components=n_comp)
pca.mean_ = pca_full.mean_
pca.components_ = pca_full.components_[:n_comp]
pca.explained_variance_ = pca_full.explained_variance_[:n_comp]
pca.noise_variance_ = pca_full.noise_variance_
pca.n_components_ = n_comp

pipe = Pipeline([('scaler', scaler), ('pca', pca), ('clf', mlp)])
onx = to_onnx(pipe, X=(np.zeros((1, 784), dtype=np.float32)),
              options={id(mlp): {'zipmap': False}})
with open(r'docs/digit_mlp.onnx', 'wb') as f:
    f.write(onx.SerializeToString())

# 验证：MNIST 真实样本上对比原模型与 ONNX 输出
sess = ort.InferenceSession(r'docs/digit_mlp.onnx', providers=['CPUExecutionProvider'])
IN = sess.get_inputs()[0].name
print('ONNX input name:', IN)
z = np.load(r'data/mnist.npz')
X, y = z['X'].astype(np.float32), z['y']
idx = np.random.RandomState(0).choice(len(X), 200, replace=False)
probs_sk = mlp.predict_proba(pca.transform(scaler.transform(X[idx])))[:, :10]
outs = sess.run(None, {IN: X[idx]})
probs_on = next(o for o in outs if o.ndim == 2 and o.shape[1] == 10)
print('输出列表 shapes:', [o.shape for o in outs])
diff = np.abs(probs_sk - probs_on).max()
acc = ((probs_on.argmax(1) == y[idx]).mean())
print('最大概率偏差:', diff)
print('ONNX 在 200 个样本上的准确率:', acc)
