"""生成 pandoc 参考模板 reference.docx，并按中文科技论文规范定义样式。"""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "report" / "reference.docx"
OUT.parent.mkdir(parents=True, exist_ok=True)

doc = Document()

# ---------- 页面设置 A4 ----------
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.top_margin, sec.bottom_margin = Cm(2.5), Cm(2.5)
sec.left_margin, sec.right_margin = Cm(2.8), Cm(2.8)


def set_font(style, cn="宋体", en="Times New Roman", size=12, bold=False, color=None):
    f = style.font
    f.name = en
    f.size = Pt(size)
    f.bold = bold
    if color:
        f.color.rgb = color
    rpr = style.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.append(rf)
    rf.set(qn("w:ascii"), en)
    rf.set(qn("w:hAnsi"), en)
    rf.set(qn("w:eastAsia"), cn)


def set_indent_chars(style, chars=200):
    """首行缩进 2 字符（firstLineChars 以 1/100 字符为单位）"""
    ppr = style.element.get_or_add_pPr()
    ind = ppr.find(qn("w:ind"))
    if ind is None:
        ind = OxmlElement("w:ind")
        ppr.append(ind)
    ind.set(qn("w:firstLineChars"), str(chars))
    ind.set(qn("w:firstLine"), "480")


def set_spacing(style, line=1.5, before=0, after=0, rule=WD_LINE_SPACING.ONE_POINT_FIVE):
    pf = style.paragraph_format
    pf.line_spacing_rule = rule
    pf.line_spacing = line
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)


styles = doc.styles

# 正文
for name in ["Normal", "Body Text", "First Paragraph", "Compact"]:
    try:
        st = styles[name]
    except KeyError:
        continue
    set_font(st, size=12)
    set_spacing(st, line=1.5, before=0, after=0)
    st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if name in ("Body Text", "First Paragraph"):
        set_indent_chars(st, 200)

# 标题
h_cfg = {
    "Heading 1": (14, 16, 8),
    "Heading 2": (12, 10, 6),
    "Heading 3": (12, 8, 4),
    "Heading 4": (12, 6, 4),
}
for name, (size, before, after) in h_cfg.items():
    st = styles[name]
    set_font(st, cn="黑体", en="Times New Roman", size=size, bold=True,
             color=RGBColor(0, 0, 0))
    set_spacing(st, line=1.4, before=before, after=after)
    st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    st.paragraph_format.keep_with_next = True
    st.paragraph_format.first_line_indent = Pt(0)

# Title
try:
    st = styles["Title"]
    set_font(st, cn="黑体", en="Times New Roman", size=18, bold=True,
             color=RGBColor(0, 0, 0))
    st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    st.paragraph_format.space_before = Pt(6)
    st.paragraph_format.space_after = Pt(6)
except KeyError:
    pass

# 图题、表题
for name in ["Image Caption", "Table Caption", "Caption"]:
    try:
        st = styles[name]
    except KeyError:
        continue
    set_font(st, cn="宋体", en="Times New Roman", size=10.5, bold=False,
             color=RGBColor(0, 0, 0))
    st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    st.paragraph_format.space_before = Pt(3)
    st.paragraph_format.space_after = Pt(9)
    st.paragraph_format.first_line_indent = Pt(0)
    st.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE

# 表格内文字
try:
    st = styles["Table"]
    set_font(st, size=10.5)
except KeyError:
    pass

# ---------- 自定义段落样式 ----------
from docx.enum.style import WD_STYLE_TYPE

def add_para_style(name, base="Normal"):
    try:
        return styles[name]
    except KeyError:
        return styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)

# 作者行
st = add_para_style("AuthorLine")
set_font(st, size=12)
st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
set_spacing(st, line=1.3, before=0, after=4)
st.paragraph_format.first_line_indent = Pt(0)

# 英文摘要
st = add_para_style("AbstractEN")
set_font(st, cn="宋体", en="Times New Roman", size=10.5)
st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
set_spacing(st, line=1.3, before=0, after=6)
st.paragraph_format.first_line_indent = Pt(0)

# 表题
st = add_para_style("TableCaption")
set_font(st, size=10.5)
st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
set_spacing(st, line=1.2, before=3, after=9, rule=WD_LINE_SPACING.SINGLE)
st.paragraph_format.first_line_indent = Pt(0)
st.paragraph_format.keep_with_next = False

# 参考文献条目
st = add_para_style("RefText")
set_font(st, size=10.5)
st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
set_spacing(st, line=1.25, before=0, after=3, rule=WD_LINE_SPACING.SINGLE)
st.paragraph_format.first_line_indent = Pt(0)

# 图片段落（居中）
st = add_para_style("Figure")
set_font(st, size=10.5)
st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
set_spacing(st, line=1.0, before=6, after=0, rule=WD_LINE_SPACING.SINGLE)
st.paragraph_format.first_line_indent = Pt(0)
st.paragraph_format.keep_with_next = True

# pandoc 引用的图片相关样式（ID 无空格）
st = add_para_style("Image Caption")
set_font(st, size=10.5)
st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
set_spacing(st, line=1.2, before=3, after=9, rule=WD_LINE_SPACING.SINGLE)
st.paragraph_format.first_line_indent = Pt(0)

st = add_para_style("Captioned Figure")
set_font(st, size=10.5)
st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
set_spacing(st, line=1.0, before=6, after=0, rule=WD_LINE_SPACING.SINGLE)
st.paragraph_format.first_line_indent = Pt(0)
st.paragraph_format.keep_with_next = True

# 页脚页码
sec.footer.is_linked_to_previous = False
p = sec.footer.paragraphs[0]
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run()
fld1 = OxmlElement("w:fldChar"); fld1.set(qn("w:fldCharType"), "begin")
instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve"); instr.text = " PAGE "
fld2 = OxmlElement("w:fldChar"); fld2.set(qn("w:fldCharType"), "end")
run._r.append(fld1); run._r.append(instr); run._r.append(fld2)
run.font.size = Pt(9)
run.font.name = "Times New Roman"

doc.save(str(OUT))
print("已生成", OUT)
