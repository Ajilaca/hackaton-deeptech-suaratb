"""Bangun result.pdf dari result.md: Markdown -> HTML (python-markdown, tabel) -> PDF (LibreOffice headless).

    python docs/result/build_pdf.py

Butuh paket `markdown` dan `libreoffice` pada PATH. Keluaran: docs/result/result.pdf (HTML antara di direktori sementara).
"""
from __future__ import annotations
import pathlib
import shutil
import subprocess
import tempfile
import markdown

HERE = pathlib.Path(__file__).resolve().parent
CSS = """
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 10pt; line-height: 1.35; }
h1 { font-size: 20pt; } h2 { font-size: 14pt; margin-top: 16pt; border-bottom: 1px solid #888; }
h3 { font-size: 11.5pt; margin-top: 12pt; }
table { border-collapse: collapse; margin: 6pt 0; width: 100%; }
th, td { border: 1px solid #999; padding: 3pt 5pt; font-size: 8.5pt; vertical-align: top; }
th { background: #e8e8e8; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.5pt; }
blockquote { color: #444; }
"""


def main() -> None:
    soffice = shutil.which("libreoffice") or shutil.which("soffice")
    if not soffice:
        raise SystemExit("libreoffice tidak ditemukan")
    md = (HERE / "result.md").read_text()
    body = markdown.markdown(md, extensions=["tables", "sane_lists"]).replace("<table>", "<table border=\"1\" cellspacing=\"0\" cellpadding=\"3\">")
    html = f"<!doctype html><html><head><meta charset='utf-8'><title>SuaraTB: laporan hasil</title><style>{CSS}</style></head><body>{body}</body></html>"
    with tempfile.TemporaryDirectory() as td:
        src = pathlib.Path(td) / "result.html"
        src.write_text(html, encoding="utf-8")
        subprocess.run([soffice, "--headless", "--convert-to", "pdf:writer_web_pdf_Export", "--outdir", td, str(src)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.copy(pathlib.Path(td) / "result.pdf", HERE / "result.pdf")
    print("ditulis", HERE / "result.pdf")


if __name__ == "__main__":
    main()
