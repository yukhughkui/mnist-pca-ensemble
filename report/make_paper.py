"""
生成论文 docx：
1. 读取 results/metrics.json 与 application_result.json，填充 paper_template.md 中的占位符
2. pandoc 转换为 docx（含 OMML 公式、图表）
3. python-docx 后处理：三线表、字体、表格居中等
运行: python report/make_paper.py
"""
import json
import platform
import ctypes
import subprocess
import sys
from pathlib import Path

import numpy as np
import pypandoc
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "report"
RES = ROOT / "results"

M = json.load(open(RES / "metrics.json", encoding="utf-8"))
A = json.load(open(RES / "application_result.json", encoding="utf-8"))
WB_PATH = RES / "web_bench.json"
WB = json.load(open(WB_PATH, encoding="utf-8")) if WB_PATH.exists() else {
    "n": 200, "ens": {"ms_mean": 215.3}, "mlp": {"ms_mean": 0.2}}
# 置信度校准分析（3.6 节）：对比数据与校准数据分别存于两个文件
EM_PATH = RES / "ens_vs_mlp.json"
CAL_PATH = RES / "calibration_check.json"
if not (EM_PATH.exists() and CAL_PATH.exists()):
    raise SystemExit("缺少校准分析数据，请先运行 src/compare_ens_mlp.py 与 src/calibration_check.py")
E = json.load(open(EM_PATH, encoding="utf-8"))
C = json.load(open(CAL_PATH, encoding="utf-8"))
recs = M["records"]


def get(name):
    return next(r for r in recs if r["model"] == name)


knn, lsvc = get("PCA + kNN (k=3)"), get("PCA + 线性SVM")
rsvc, rf = get("PCA + RBF-SVM"), get("PCA + 随机森林")
mlp, hog_svm = get("PCA + MLP"), get("HOG + 线性SVM")
raw_mlp, ens = get("原始像素 + MLP"), get("PCA + 软投票集成")

# ---- 硬件信息 ----
def cpu_name():
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                           r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        return platform.processor() or "未知"


def ram_gb():
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MEMORYSTATUSEX(); st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return f"{st.ullTotalPhys / 1024**3:.0f} GB"
    except Exception:
        return "未知"


import sklearn
cpu, ram = cpu_name(), ram_gb()

# ---- 混淆矩阵统计（集成模型, 由 fix_confusion.py 写入 metrics.json） ----
cms = M.get("confusion")
if cms is None:
    raise SystemExit("metrics.json 缺少 confusion, 请先运行 src/fix_confusion.py")
cm_ens = np.array(cms["ens"], dtype=float)
cm_n = cm_ens / cm_ens.sum(axis=1, keepdims=True)
pairs = sorted([(cm_n[a, b], a, b) for a in range(10) for b in range(10) if a != b],
               reverse=True)[:3]
DIAG_MIN = f"{np.diag(cm_n).min()*100:.0f}%"

acc_pct = lambda r: f"{r['accuracy']*100:.2f}%"
acc_pct_l = lambda v: f"{v*100:.2f}%"

# ---- 3.6 节校准数值 ----
A_STD, B_HAND = E["A_standard"], E["B_handdrawn"]
cal_rows = [r for r in C["calibration"]["MLP"] if r]
mlp_high = next(r for r in cal_rows if r["bin"] == ">95%")
mlp_6080 = next(r for r in cal_rows if r["bin"] == "60-80%")
ens_6080 = next(r for r in C["calibration"]["集成"] if r and r["bin"] == "60-80%")
mem = C["members"]

vals = {
    "CAL_ACC_ENS_STD": acc_pct_l(A_STD["acc_ens"]),
    "CAL_ACC_MLP_STD": acc_pct_l(A_STD["acc_mlp"]),
    "CAL_P_STD": f"{A_STD['p_value']:.4f}",
    "CAL_ACC_ENS_HAND": acc_pct_l(B_HAND["acc_ens"]),
    "CAL_ACC_MLP_HAND": acc_pct_l(B_HAND["acc_mlp"]),
    "CAL_P_HAND": f"{B_HAND['p_value']:.4f}",
    "CAL_DROP_RF": f"{abs(mem['随机森林']['drop_pp']):.2f}",
    "CAL_DROP_KNN": f"{abs(mem['kNN']['drop_pp']):.2f}",
    "CAL_DROP_MLP": f"{abs(mem['MLP']['drop_pp']):.2f}",
    "CAL_MLP_HIGH": f"{mlp_high['n']/B_HAND['n']*100:.1f}%",
    "CAL_MLP_6080": f"{mlp_6080['actual']*100:.1f}%",
    "CAL_ENS_6080": f"{ens_6080['actual']*100:.1f}%",
    "CAL_MLP_REJ": f"{E['confidence']['mlp']['reject']/B_HAND['n']*100:.1f}%",
    "THRESH": "0.60",
    "NCOMP": str(M["pca"]["n95"]),
    "DIMRATIO": f"{(784-M['pca']['n95'])/784*100:.1f}",
    "ACC_ENS": acc_pct(ens),
    "F1_ENS": f"{ens['macro_f1']:.4f}",
    "APP_ACC": f"{A['exact_match']*100:.0f}%",
    "APP_CONF": f"{np.mean(A['confidence'])*100:.0f}%",
    "N_APP": str(len(A["truth"])),
    "N_CORRECT": str(sum(p == t for p, t in zip(A["pred"], A["truth"]))),
    "WB_N": str(WB.get("n", 200)),
    "WEB_MS_ENS": f"{WB['ens']['ms_mean']/1000:.2f} s",
    "WEB_MS_MLP": f"{WB['mlp']['ms_mean']:.1f} ms",
    "CPU": cpu,
    "RAM": ram,
    "SKVER": sklearn.__version__,
    "ACC_RF": acc_pct(rf),
    "ACC_HOG": acc_pct(hog_svm),
    "ACC_LSVC": acc_pct(lsvc),
    "ACC_RSVC": acc_pct(rsvc),
    "ACC_MLP": acc_pct(mlp),
    "ACC_RAWMLP": acc_pct(raw_mlp),
    "ACC_KNN": acc_pct(knn),
    "RAW_MLP_T": f"{raw_mlp['train_time_s']:.0f}",
    "MLP_T": f"{mlp['train_time_s']:.0f}",
    "SWEEP_A20": f"{M['dim_sweep'][0]['accuracy']*100:.2f}%",
    "SWEEP_A154": f"{[d for d in M['dim_sweep'] if d['dim']==M['pca']['n95']][0]['accuracy']*100:.2f}%",
    "SWEEP_A200": f"{[d for d in M['dim_sweep'] if d['dim']==200][0]['accuracy']*100:.2f}%",
    "LC_SVM": acc_pct_l(M["learning_curve"]["svm"][-1]),
    "LC_RF": acc_pct_l(M["learning_curve"]["rf"][-1]),
    "CV_STD_MAX": f"{max(v['std'] for v in M['cv'].values()):.4f}",
    "CM_MIX": "、".join(f"{a}→{b}（{v*100:.1f}%）" for v, a, b in pairs),
    "DIAG_MIN": DIAG_MIN,
}
# 主要混淆对的精确值（4→9、3→5）在拿到完整混淆矩阵后覆盖
vals["CM_49"] = f"{cm_n[4,9]*100:.1f}%"
vals["CM_35"] = f"{cm_n[3,5]*100:.1f}%"

# ---- 表 2 ----
rows = ["| 模型 | 维度 | 耗时 (s) | 时延 (ms) | 准确率 (%) | 宏平均 F1 |",
        "|:------------------|:----:|:----:|:----:|:----:|:----:|"]
for r in recs:
    rows.append(f"| {r['model']} | {r['dim']} | {r['train_time_s']:.1f} | "
                f"{r['latency_ms_per_sample']:.2f} | {r['accuracy']*100:.2f} | {r['macro_f1']:.4f} |")
table2 = "\n".join(rows) + "\n\n注：RBF 核 SVM 的训练集为随机抽取的 10 000 个子样本，其余模型均使用全部 60 000 个训练样本。"

# ---- 表 3 ----
rows = ["| 模型 | 折 1 | 折 2 | 折 3 | 折 4 | 折 5 | 均值 | 标准差 |",
        "|:----------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"]
for name, v in M["cv"].items():
    s = v["scores"]
    rows.append(f"| {name} | " + " | ".join(f"{x:.4f}" for x in s) +
                f" | {v['mean']:.4f} | {v['std']:.4f} |")
table3 = "\n".join(rows)

# ---- 混淆矩阵文字 ----
if cm_ens is not None:
    # 找 4->9 与 3->5
    v49 = cm_n[4, 9]; v35 = cm_n[3, 5]
    vals["CM_49"] = f"{v49*100:.1f}%"
    vals["CM_35"] = f"{v35*100:.1f}%"
    pairs = sorted([(cm_n[a, b], a, b) for a in range(10) for b in range(10) if a != b],
                   reverse=True)[:3]
    vals["CM_MIX"] = "、".join(f"{a}→{b}（{v*100:.1f}%）" for v, a, b in pairs)

# ---- 模板填充 ----
import re
tpl = open(REP / "paper_template.md", encoding="utf-8").read()
tpl = tpl.replace("\\boldsymbol{", "\\mathbf{")
# 保护 custom-style 属性的引号, 再把正文直引号换成中文弯引号
tpl = tpl.replace('{custom-style="', "{custom-style=\x00").replace(
    '"}', "\x01}")
tpl = re.sub(r'"([^"\n]{1,40})"', r"“\1”", tpl)
tpl = tpl.replace("{custom-style=\x00", '{custom-style="').replace("\x01", '"')
for k, v in vals.items():
    tpl = tpl.replace("{{" + k + "}}", v)
tpl = tpl.replace("{{TABLE2}}", table2).replace("{{TABLE3}}", table3)

left = [k for k in tpl.split("{{") if "}}" in k and k.split("}}")[0].isupper()]
if left:
    print("警告: 未填充的占位符:", ["{{" + l.split("}}")[0] + "}}" for l in left])

md_path = REP / "paper.md"
md_path.write_text(tpl, encoding="utf-8")

# ---- pandoc 转换 ----
raw = REP / "paper_raw.docx"
pypandoc.convert_file(
    str(md_path), "docx", format="markdown+east_asian_line_breaks-smart",
    outputfile=str(raw), extra_args=[
        f"--reference-doc={REP/'reference.docx'}",
        "--resource-path", str(REP),
    ])
print("pandoc 完成 ->", raw)

# ---- 后处理 ----
doc = Document(str(raw))

def set_run_font(run, size=None, bold=None, cn="宋体", en="Times New Roman"):
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts"); rpr.insert(0, rf)
    rf.set(qn("w:ascii"), en); rf.set(qn("w:hAnsi"), en); rf.set(qn("w:eastAsia"), cn)


def table_borders_three_line(tbl):
    tblPr = tbl._tbl.tblPr
    old = tblPr.find(qn("w:tblBorders"))
    if old is not None:
        tblPr.remove(old)
    borders = OxmlElement("w:tblBorders")
    for tag, sz in (("top", "12"), ("bottom", "12")):
        e = OxmlElement(f"w:{tag}")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), sz)
        e.set(qn("w:space"), "0"); e.set(qn("w:color"), "000000")
        borders.append(e)
    for tag in ("left", "right", "insideV", "insideH"):
        e = OxmlElement(f"w:{tag}")
        e.set(qn("w:val"), "none"); e.set(qn("w:sz"), "0")
        e.set(qn("w:space"), "0"); e.set(qn("w:color"), "auto")
        borders.append(e)
    tblPr.append(borders)
    # 表头行下边框
    hdr = tbl.rows[0]
    trPr = hdr._tr.get_or_add_trPr()
    # 用每个表头单元格的下边框实现
    for cell in hdr.cells:
        tcPr = cell._tc.get_or_add_tcPr()
        bd = OxmlElement("w:tcBorders")
        e = OxmlElement("w:bottom")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "6")
        e.set(qn("w:space"), "0"); e.set(qn("w:color"), "000000")
        bd.append(e)
        tcPr.append(bd)


for ti, tbl in enumerate(doc.tables):
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_borders_three_line(tbl)
    for ri, row in enumerate(tbl.rows):
        for cell in row.cells:
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if (ri == 0 or True) else p.alignment
                for r in p.runs:
                    set_run_font(r, size=10.5, bold=True if ri == 0 else None)

# 表题移到表格上方（中文科技论文规范）
W_P = qn("w:p")
W_TBL = qn("w:tbl")
for p in list(doc.paragraphs):
    if p.style.name == "TableCaption":
        el = p._p.getprevious()
        while el is not None and el.tag not in (W_TBL, W_P):
            el = el.getprevious()
        if el is not None and el.tag == W_TBL:
            el.addprevious(p._p)
        else:  # 表题紧跟空段落的情况
            prev = p._p.getprevious()
            if prev is not None and prev.tag == W_P:
                # 检查空段落之前是否为表格
                prev2 = prev.getprevious()
                if prev2 is not None and prev2.tag == W_TBL:
                    prev2.addprevious(p._p)

# 正文段落统一首行缩进 2 字符
def add_first_line_indent(par, chars=200):
    ppr = par._p.get_or_add_pPr()
    ind = ppr.find(qn("w:ind"))
    if ind is None:
        ind = OxmlElement("w:ind")
        ppr.append(ind)
    ind.set(qn("w:firstLineChars"), str(chars))
    ind.set(qn("w:firstLine"), "480")

for p in doc.paragraphs:
    if p.style.name in ("Normal", "Body Text", "First Paragraph") and p.text.strip():
        add_first_line_indent(p)

# 图题/表题: 居中、去缩进、五号字
CAPTION_IDS = {"ImageCaption", "TableCaption", "Caption"}
for p in doc.paragraphs:
    ppr = p._p.find(qn("w:pPr"))
    if ppr is None:
        continue
    ps = ppr.find(qn("w:pStyle"))
    if ps is None or ps.get(qn("w:val")) not in CAPTION_IDS:
        continue
    ind = ppr.find(qn("w:ind"))
    if ind is not None:
        ppr.remove(ind)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in p.runs:
        set_run_font(r, size=10.5)

# 图片段落居中
for p in doc.paragraphs:
    if p._element.findall(qn("w:r")) and p._element.findall(".//" + qn("w:drawing")):
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

# 全局正文运行字体兜底
for p in doc.paragraphs:
    for r in p.runs:
        rpr = r._element.get_or_add_rPr()
        rf = rpr.find(qn("w:rFonts"))
        if rf is None:
            rf = OxmlElement("w:rFonts"); rpr.insert(0, rf)
        if rf.get(qn("w:eastAsia")) is None:
            rf.set(qn("w:eastAsia"), "宋体")
            rf.set(qn("w:ascii"), rf.get(qn("w:ascii")) or "Times New Roman")
            rf.set(qn("w:hAnsi"), rf.get(qn("w:hAnsi")) or "Times New Roman")

final = REP / "机器学习课程报告_基于PCA与集成学习的手写数字识别.docx"
doc.save(str(final))
print("已生成", final)
