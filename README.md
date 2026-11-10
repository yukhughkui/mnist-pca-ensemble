# 基于 PCA 特征降维与多分类器集成的手写数字识别

**在线体验（免安装，点开即用）**：https://yukhughkui.github.io/mnist-pca-ensemble/
—— 浏览器画板手写数字，本地推理，与本地演示同一套标准化链路（MLP 版模型）。

机器学习与 Python 实践课程大作业。以 MNIST（70 000 张 28×28 手写数字图像）为对象，
构建"预处理 → PCA/HOG 特征 → 五种分类器对比 → 软投票集成 → 票据录入应用 → 在线演示系统"的完整流水线。

## 快速开始（Windows 双击即可）

| 脚本 | 作用 |
|---|---|
| **启动识别演示.bat** | 启动浏览器交互演示：画板上手写数字，实时显示识别结果、置信度与概率分布 |
| **一键复现实验.bat** | 从零训练全部模型并生成 13+ 张图表与指标（CPU 约 6 分钟） |
| **一键生成论文.bat** | 用最新实验数据重新生成报告 docx |

## 目录结构

```
ml_project/
├── data/                  # mnist.npz（70 000 张，已下载解析）
├── figures/               # 全部实验图表（fig01–fig16）
├── web/                   # 演示系统前端（index.html）
├── results/               # metrics.json / 训练好的模型 / 运行日志
├── src/
│   ├── run_experiment.py  # 主实验：训练 8 组模型、CV、维度扫描、学习曲线、全部图表
│   ├── fix_confusion.py   # 可选校验：独立复测集成模型混淆矩阵（主脚本已内置该步骤）
│   ├── application_demo.py# 应用演示：票据手写数字自动录入
│   ├── web_demo.py        # 在线演示系统后端（Python 标准库，无额外依赖）
│   ├── bench_web.py       # 演示系统推理基准（结果写入 results/web_bench.json）
│   ├── compare_ens_mlp.py # 集成 vs MLP 公平对比 + McNemar 检验（论文 3.6 节数据）
│   └── calibration_check.py # 置信度校准分析（论文 3.6 节 + 图 13）
└── report/
    ├── paper_template.md  # 论文源稿（LaTeX 公式）
    ├── make_reference.py  # 生成中文科技论文排版模板 reference.docx
    ├── make_paper.py      # 填充实验数据 → pandoc → docx 后处理（三线表/字体/缩进）
    └── 机器学习课程报告_基于PCA与集成学习的手写数字识别.docx
```

## 运行方式（命令行等价写法）

```bash
pip install -r requirements.txt
python src/run_experiment.py     # 约 6 分钟，产出全部指标与图表（含混淆矩阵）
python src/application_demo.py  # 票据演示，产出图 13
python src/web_demo.py           # 启动在线演示，浏览器访问 http://127.0.0.1:8765
python src/bench_web.py          # 需保持 web_demo.py 正在运行
python src/fix_confusion.py      # 可选：独立复测混淆矩阵，用于校验结果
python report/make_reference.py  # 仅需一次
python report/make_paper.py      # 重新生成论文 docx
```

**依赖关系**：7 个脚本互不 import，全部通过 `data/` 与 `results/` 里的文件通信。因此除 `bench_web.py` 外都能单独运行；前提是所需的产物文件已存在：

> 注意：`results/deploy_model.joblib`（集成模型，133MB）超过 GitHub 单文件上限，**不随仓库分发**。
> clone 后先运行 `python src/run_experiment.py`（约 6 分钟）即可生成，随后全部演示可用。
> `data/mnist.npz`、`results/deploy_model_mlp.joblib` 等其余产物均已随仓库附带。

| 脚本 | 需要先存在的文件 |
|---|---|
| `run_experiment.py` | 无（无缓存时自动下载 MNIST） |
| `application_demo.py` / `web_demo.py` | `deploy_model.joblib`、`mnist.npz`、`metrics.json` |
| `fix_confusion.py` | 同上（读模型重算，不依赖主脚本是否在运行） |
| `compare_ens_mlp.py` / `calibration_check.py` | `deploy_model(.mlp).joblib`、`mnist.npz` |
| `bench_web.py` | 以上全部，且 `web_demo.py` 服务必须在 8765 端口运行 |

`一键生成论文.bat` 内置了兜底：若 `metrics.json` 缺少 `confusion` 字段，会自动先执行 `fix_confusion.py` 补齐再生成论文。

## 在线演示系统说明

`启动识别演示.bat` 会启动本地服务并自动打开浏览器（`?demo` 参数可直接进入示例模式）：

- 画板手写（鼠标/触屏），抬笔自动识别；也可载入 MNIST 测试集样本或本地图片
- 实时显示：识别结果、置信度、Top-3 候选、十类概率分布、按阈值 0.60 的拒识判定
- 右侧同步展示模型实际接收的 28×28 标准化输入（反色 → 包围盒裁切 → 质心居中）
- 支持在「软投票集成 / MLP」间一键切换，直观对比精度与时延
- 推理链路与论文完全一致；基准见 `results/web_bench.json`（集成约 0.2 s，MLP 约 0.2 ms）

## 主要结果（测试集 10 000 张）

| 模型 | 准确率 | 宏平均 F1 |
|---|---|---|
| PCA + 软投票集成 | **98.27%** | 0.9827 |
| 原始像素 + MLP | 98.24% | 0.9823 |
| PCA + MLP | 97.99% | 0.9797 |
| HOG + 线性 SVM | 97.29% | 0.9728 |
| PCA + RBF-SVM | 97.26% | 0.9724 |
| PCA + kNN (k=3) | 97.22% | 0.9721 |
| PCA + 随机森林 | 95.13% | 0.9508 |
| PCA + 线性 SVM | 91.65% | 0.9151 |

PCA 将 784 维压缩至 154 维（95% 累计方差）；票据录入演示整串准确率 90%（10 个数字错 1 个，
错误样本置信度仅 30%，可被拒识机制拦截）。

## 环境

Python 3.13，CPU（Intel i7-14700HX）即可复现全部实验，无需 GPU。
演示系统仅用 Python 标准库（http.server）+ 训练依赖（numpy/scikit-learn/pillow/joblib）。
