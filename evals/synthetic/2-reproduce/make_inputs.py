#!/usr/bin/env python3
"""합성데이터 md에서 파싱 파이프라인 입력 파일을 만든다.

  (a) documents/*.md        → build/{code}-{name}.pdf        텍스트 레이어 있음
  (b) PARSE_NO_TEXT_LAYER   → build/{code}-{title}-스캔본.pdf  텍스트 레이어 없음
  (c) PARSE_IMAGE_ONLY      → build/{code}-{title}-사진.jpg

`--check`를 주면 새로 만들지 않고 build/만 검사한다.
이 스크립트는 .venv 의 파이썬으로 실행해야 한다 (reportlab·pypdf·fontTools 필요).
"""
import re
import shutil
import subprocess
import tempfile
import sys
import unicodedata
from pathlib import Path

import yaml
from pypdf import PdfReader, PdfWriter
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

# 생성 시각과 문서 ID를 고정한다. 켜지 않으면 다시 구울 때마다 51개 파일의
# 바이트가 전부 달라져, 버전 관리에서 실제 변경분을 가려낼 수 없다.
rl_config.invariant = 1

ROOT = Path(__file__).resolve().parent          # 2-reproduce/ — 원본 md 와 매니페스트가 있는 곳
BUILD = ROOT.parent / "1-dataset" / "build"     # 완성물은 1-dataset/ 으로 나간다
FONT = "AppleGothic"
MAX_BYTES = 10 * 1024 * 1024  # Files.file_size 제한
# AppleSDGothicNeo 는 TTC 묶음인데다 PostScript 아웃라인이라 reportlab 이 못 읽는다.
# AppleGothic 은 단일 TTF + glyf 아웃라인이라 그대로 쓸 수 있다.
FONT_TTF = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"


def ensure_font():
    pdfmetrics.registerFont(TTFont(FONT, FONT_TTF))


def nfc(p):
    """macOS는 한글 파일명을 NFD로 저장한다. 비교 전에 NFC로 맞춘다."""
    return unicodedata.normalize("NFC", p.name)


def safe(name):
    return re.sub(r"[ ·/]+", "_", name).strip("_")


def parse_md(text):
    """쓰는 문법만 다룬다: # 제목, ## 절, | 표 |, - 목록, "- N -" 쪽나눔, 그 외 문단."""
    blocks, buf, table = [], [], []
    PAGE_MARK = re.compile(r"-?\s*\d+\s*-")

    def clean(s):
        s = re.sub(r"\*\*([^*]*)\*\*", r"\1", s)
        # 「302호 김미정님」이 줄바꿈으로 갈라지면 개인정보 정규식이 놓친다
        return re.sub(r"(\d{3,4}호) ([가-힣]{2,4}님)", r"\1&nbsp;\2", s)

    def flush_p():
        if buf:
            blocks.append(("p", " ".join(buf)))
            buf.clear()

    def flush_table():
        if table:
            blocks.append(("table", list(table)))
            table.clear()

    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|"):
            flush_p()
            cells = [clean(c.strip()) for c in s.strip("|").split("|")]
            if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):  # 구분행은 버린다
                table.append(cells)
            continue
        flush_table()
        if not s:
            flush_p()
        elif PAGE_MARK.fullmatch(s):
            flush_p()
            blocks.append(("pagebreak", ""))
        elif s.startswith("## "):
            flush_p()
            blocks.append(("h2", clean(s[3:])))
        elif s.startswith("# "):
            flush_p()
            blocks.append(("h1", clean(s[2:])))
        elif s.startswith("- "):
            flush_p()
            blocks.append(("li", clean(s[2:])))
        else:
            buf.append(clean(s))
    flush_p()
    flush_table()
    return blocks


def styles():
    base = dict(fontName=FONT, leading=15)
    return {
        "h1": ParagraphStyle("h1", fontSize=15, spaceAfter=10, leading=20, fontName=FONT),
        "h2": ParagraphStyle("h2", fontSize=12, spaceBefore=10, spaceAfter=5, leading=16, fontName=FONT),
        "p": ParagraphStyle("p", fontSize=10, spaceAfter=5, **base),
        "li": ParagraphStyle("li", fontSize=10, leftIndent=10, bulletIndent=2, spaceAfter=2, **base),
        "cell": ParagraphStyle("cell", fontSize=9, leading=12, fontName=FONT),
    }


def render(blocks, out, running_header=None):
    """running_header 가 있으면 모든 페이지 상단에 반복 머리말로, 하단에 쪽번호를 찍는다."""
    st = styles()
    avail = A4[0] - 40 * mm
    flow = []
    for kind, val in blocks:
        if kind == "pagebreak":
            flow.append(PageBreak())
        elif kind == "table":
            data = [[Paragraph(c, st["cell"]) for c in row] for row in val]
            # 열 너비를 내용 길이에 비례시켜 셀 안 줄바꿈을 줄인다
            ncol = max(len(r) for r in val)
            widest = [max((len(r[i]) if i < len(r) else 0) for r in val) or 1 for i in range(ncol)]
            total = sum(widest)
            widths = [max(18 * mm, avail * w / total) for w in widest]
            scale = avail / sum(widths)
            t_ = Table(data, colWidths=[w * scale for w in widths], repeatRows=1, hAlign="LEFT")
            t_.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#444444")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]))
            flow += [Spacer(1, 4), t_, Spacer(1, 6)]
        elif kind == "li":
            flow.append(Paragraph(val, st["li"], bulletText="·"))
        else:
            flow.append(Paragraph(val, st[kind]))

    def decorate(canvas, doc):
        if not running_header:
            return
        canvas.saveState()
        canvas.setFont(FONT, 8)
        canvas.drawString(20 * mm, A4[1] - 12 * mm, running_header)
        canvas.drawCentredString(A4[0] / 2, 12 * mm, f"- {doc.page} -")
        canvas.restoreState()

    top = 22 * mm if running_header else 18 * mm
    doc = BaseDocTemplate(str(out), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                          topMargin=top, bottomMargin=18 * mm, title=out.stem)
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=decorate)])
    doc.build(flow)


def sips(args):
    r = subprocess.run(["sips", *args], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"sips 실패: {r.stderr.decode()[:200]}")


def out_name(code, doc, kind):
    if kind == "scan":
        return BUILD / f"{code}-{safe(doc['document_title'])}-스캔본.pdf"
    if kind == "photo":
        return BUILD / f"{code}-{safe(doc['document_title'])}-사진.{doc['file_type']}"
    return BUILD / f"{code}-{Path(doc['file']).stem}.pdf"


def plan():
    """매니페스트가 곧 기대값이다. 파일 개수를 손으로 적지 않는다."""
    items = []
    for d in sorted((ROOT / "buildings").glob("b*")):
        b = yaml.safe_load((d / "building.yaml").read_text(encoding="utf-8"))
        for doc in b["documents"]:
            intent = set(doc["intent"])
            kind = ("scan" if "PARSE_NO_TEXT_LAYER" in intent
                    else "photo" if "PARSE_IMAGE_ONLY" in intent else "pdf")
            items.append((d, b, doc, kind, out_name(b["building_code"], doc, kind)))
    return items


def rasterize(src, out):
    """텍스트 레이어를 없앤다. sips 는 PDF 첫 장만 변환하므로 쪽마다 따로 굽고 다시 합친다."""
    reader = PdfReader(str(src))
    writer = PdfWriter()
    for i in range(len(reader.pages)):
        one = src.with_name(f"p{i}.pdf")
        w1 = PdfWriter()
        w1.add_page(reader.pages[i])
        with open(one, "wb") as fh:
            w1.write(fh)
        png = src.with_name(f"p{i}.png")
        flat = src.with_name(f"p{i}-flat.pdf")
        sips(["-s", "format", "png", str(one), "--out", str(png)])
        sips(["-s", "format", "pdf", str(png), "--out", str(flat)])
        writer.add_page(PdfReader(str(flat)).pages[0])
    with open(out, "wb") as fh:
        writer.write(fh)


def build():
    ensure_font()
    if BUILD.exists():
        shutil.rmtree(BUILD)  # 이전 산출물이 섞여 검사가 잘못 통과하는 것을 막는다
    BUILD.mkdir()
    counts = {"pdf": 0, "scan": 0, "photo": 0}

    for d, b, doc, kind, out in plan():
        blocks = parse_md((d / doc["file"]).read_text(encoding="utf-8"))
        header = None
        if {"CLEAN_REPEATED_HEADER", "CLEAN_OVER_REMOVAL"} & set(doc["intent"]):
            # 본문에 글자로 박아둔 반복 머리말을 실제 페이지 머리말로 올린다
            seen = {}
            for k, v in blocks:
                if k == "p":
                    seen[v] = seen.get(v, 0) + 1
            repeated = [v for v, n in seen.items() if n >= 2]
            header = repeated[0] if repeated else f"{b['building_name']} 관리사무소"
            blocks = [(k, v) for k, v in blocks if not (k == "p" and v in repeated)]

        if kind == "pdf":
            render(blocks, out, header)
        else:
            tmp = Path(tempfile.mkdtemp()) / "src.pdf"
            try:
                render(blocks, tmp, header)
                if kind == "scan":
                    rasterize(tmp, out)
                else:
                    # 굽는 포맷도 매니페스트가 정한다. formatOptions 는 JPEG 품질 옵션이라
                    # png 에 넘기면 안 된다.
                    fmt = "jpeg" if doc["file_type"] == "jpg" else doc["file_type"]
                    opts = ["-s", "formatOptions", "70"] if fmt == "jpeg" else []
                    sips(["-s", "format", fmt, *opts, str(tmp), "--out", str(out)])
            finally:
                shutil.rmtree(tmp.parent, ignore_errors=True)
        counts[kind] += 1
    return counts


def pdf_pages(p):
    return len(PdfReader(str(p)).pages)


def pdf_text(p):
    return re.sub(r"\s", "", "".join(pg.extract_text() or "" for pg in PdfReader(str(p)).pages))


def check():
    items = plan()
    expect = {k: sum(1 for *_, kind, _ in items if kind == k) for k in ("pdf", "scan", "photo")}
    # macOS 가 파일명을 NFD 로 저장하므로 이름은 NFC 로 맞춰 비교한다
    on_disk = {nfc(p): p for p in BUILD.iterdir() if p.is_file()} if BUILD.exists() else {}
    planned = {unicodedata.normalize("NFC", out.name): out for *_, out in items}

    missing = [n for n in planned if n not in on_disk]
    extra = [n for n in on_disk if n not in planned]   # 이전 실행 잔재
    lost, thin, oversize = [], [], []

    for d, b, doc, kind, out in items:
        if unicodedata.normalize("NFC", out.name) not in on_disk:
            continue
        if out.stat().st_size > MAX_BYTES:
            oversize.append(out.name)
        if kind == "scan":
            if out.read_bytes().count(b"/Font"):
                lost.append(f"{out.name}: 스캔본인데 텍스트 레이어가 있다")
            continue
        if kind == "photo":
            continue
        got = pdf_text(out)
        for k, v in parse_md((d / doc["file"]).read_text(encoding="utf-8")):
            if k in ("p", "li", "h1", "h2"):
                want = re.sub(r"\s", "", v.replace("&nbsp;", ""))
                if len(want) > 10 and want not in got:
                    lost.append(f"{out.name}: {v[:40]}…")
                    break
        if "CLEAN_REPEATED_HEADER" in doc["intent"] and pdf_pages(out) < 2:
            thin.append(f"{out.name}: {pdf_pages(out)}쪽 — 머리말이 반복되지 않는다")

    rows = [
        ("본문 PDF", sum(1 for *_, k, o in items
                       if k == "pdf" and unicodedata.normalize("NFC", o.name) in on_disk),
         expect["pdf"]),
        ("스캔본 PDF", expect["scan"], expect["scan"]),
        ("사진 이미지", expect["photo"], expect["photo"]),
        ("매니페스트에 없는 잔재", len(extra), 0),
        ("빠진 산출물", len(missing), 0),
        ("본문이 잘린 문서", len(lost), 0),
        ("머리말 반복 미달", len(thin), 0),
        ("10MB 초과", len(oversize), 0),
    ]
    ok = True
    for label, got, want in rows:
        ok &= got == want
        print(f"  {label:24} {got:3} / 기대 {want:3}{'' if got == want else '  FAIL'}")
    for x in (lost + thin + extra + missing)[:10]:
        print(f"    {x}")
    return ok


def main():
    if "--check" not in sys.argv:
        made = build()
        print(f"생성: 본문 {made['pdf']} · 스캔본 {made['scan']} · 사진 {made['photo']}")
    print("검사")
    if not check():
        print("\nFAIL")
        sys.exit(1)
    print("\nPASS")


if __name__ == "__main__":
    main()
