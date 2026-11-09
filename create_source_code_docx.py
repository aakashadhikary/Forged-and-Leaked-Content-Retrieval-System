"""
Generates a Microsoft Word (.docx) document containing all source code files
for the Content Forensics project (Image pipeline first, then Video pipeline),
with all comments and docstrings stripped.
"""

import ast
import io
import sys
import tokenize
import zipfile
import xml.sax.saxutils as saxutils
from pathlib import Path

class DocstringRemover(ast.NodeTransformer):
    """AST Transformer to strip docstrings from modules, classes, and functions."""
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if (node.body and isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:]
            if not node.body:
                node.body = [ast.Pass()]
        return node

    def visit_AsyncFunctionDef(self, node):
        self.generic_visit(node)
        if (node.body and isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:]
            if not node.body:
                node.body = [ast.Pass()]
        return node

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        if (node.body and isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:]
            if not node.body:
                node.body = [ast.Pass()]
        return node

    def visit_Module(self, node):
        self.generic_visit(node)
        if (node.body and isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:]
        return node


def clean_source_code(filepath: Path) -> str:
    """Reads a Python file and strips both single-line comments (#) and docstrings."""
    with open(filepath, "r", encoding="utf-8") as f:
        src = f.read()

    # Step 1: Strip # comments via tokenize
    io_obj = io.BytesIO(src.encode('utf-8'))
    tokens = tokenize.tokenize(io_obj.readline)
    filtered = [t for t in tokens if t.type != tokenize.COMMENT]
    no_comments = tokenize.untokenize(filtered).decode('utf-8')

    # Step 2: Strip docstrings via AST
    tree = ast.parse(no_comments)
    DocstringRemover().visit(tree)
    ast.fix_missing_locations(tree)
    clean_code = ast.unparse(tree)

    # Normalize excessive blank lines (max 1 blank line between statements)
    lines = clean_code.splitlines()
    normalized = []
    prev_blank = False
    for line in lines:
        if not line.strip():
            if not prev_blank:
                normalized.append("")
                prev_blank = True
        else:
            normalized.append(line)
            prev_blank = False

    return "\n".join(normalized).strip()


def build_docx(output_filename: str):
    image_files = [
        ("build_corpus.py", "1. Image Corpus Generation & Synthetic Augmentation"),
        ("fingerprints.py", "2. Multi-Modal Feature Extraction & Score Fusion Core"),
        ("index_corpus.py", "3. Corpus Feature Extraction & Index Serializer"),
        ("search.py", "4. Forensic Search, Top-K Retrieval & Match Strip Generator"),
        ("evaluate.py", "5. Evaluation Benchmark & Performance Metrics"),
    ]

    video_files = [
        ("build_video_corpus.py", "1. Video Corpus Generation & Augmentation Engine"),
        ("video_fingerprints.py", "2. Video Frame Sampling & Multi-Modal Feature Extraction"),
        ("index_videos.py", "3. Video Corpus Indexing & Frame Database Caching"),
        ("search_video.py", "4. Video Forensic Retrieval, Temporal Alignment & Localization"),
        ("evaluate_video.py", "5. Video Retrieval & Localization Benchmark"),
    ]

    shared_files = [
        ("visualize_results.py", "Shared Multi-Modal Visualization Engine & Architecture Diagrams"),
    ]

    # OpenXML Structure
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

    doc_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

    styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
        <w:sz w:val="22"/>
      </w:rPr>
    </w:rPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:styleId="DocTitle">
    <w:name w:val="Title"/>
    <w:pPr>
      <w:spacing w:before="400" w:after="160"/>
      <w:jc w:val="center"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:b/>
      <w:sz w:val="48"/>
      <w:color w:val="0D233A"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="DocSubtitle">
    <w:name w:val="Subtitle"/>
    <w:pPr>
      <w:spacing w:before="0" w:after="400"/>
      <w:jc w:val="center"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:sz w:val="26"/>
      <w:color w:val="5A6B82"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="PartHeading">
    <w:name w:val="Part Heading"/>
    <w:pPr>
      <w:spacing w:before="360" w:after="160"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:b/>
      <w:sz w:val="34"/>
      <w:color w:val="1F497D"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="FileHeading">
    <w:name w:val="File Heading"/>
    <w:pPr>
      <w:spacing w:before="280" w:after="80"/>
      <w:pageBreakBefore/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>
      <w:b/>
      <w:sz w:val="28"/>
      <w:color w:val="2B579A"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="FileDesc">
    <w:name w:val="File Description"/>
    <w:pPr>
      <w:spacing w:before="0" w:after="160"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:i/>
      <w:sz w:val="20"/>
      <w:color w:val="666666"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="CodeLine">
    <w:name w:val="Code Line"/>
    <w:pPr>
      <w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/>
      <w:shd w:val="clear" w:color="auto" w:fill="F7F8FA"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>
      <w:sz w:val="18"/>
      <w:color w:val="1E1E1E"/>
    </w:rPr>
  </w:style>
</w:styles>"""

    body_xml_parts = []

    # Title & Subtitle
    body_xml_parts.append("""
    <w:p>
      <w:pPr><w:pStyle w:val="DocTitle"/></w:pPr>
      <w:r><w:t>Content Forensics &amp; Cross-Platform Retrieval</w:t></w:r>
    </w:p>
    <w:p>
      <w:pPr><w:pStyle w:val="DocSubtitle"/></w:pPr>
      <w:r><w:t>Complete Source Code Documentation (Comments &amp; Docstrings Removed)</w:t></w:r>
    </w:p>
    """)

    def add_section(part_title, file_list):
        body_xml_parts.append(f"""
        <w:p>
          <w:pPr><w:pStyle w:val="PartHeading"/></w:pPr>
          <w:r><w:t>{saxutils.escape(part_title)}</w:t></w:r>
        </w:p>
        """)

        for fname, desc in file_list:
            fpath = Path(fname)
            if not fpath.exists():
                print(f"Warning: File not found: {fname}")
                continue

            cleaned_code = clean_source_code(fpath)

            # Heading for file
            body_xml_parts.append(f"""
            <w:p>
              <w:pPr><w:pStyle w:val="FileHeading"/></w:pPr>
              <w:r><w:t>{saxutils.escape(fname)}</w:t></w:r>
            </w:p>
            <w:p>
              <w:pPr><w:pStyle w:val="FileDesc"/></w:pPr>
              <w:r><w:t>{saxutils.escape(desc)}</w:t></w:r>
            </w:p>
            """)

            # Code lines
            for line in cleaned_code.splitlines():
                escaped_line = saxutils.escape(line)
                # Word requires xml:space="preserve" on w:t to retain leading indentation
                body_xml_parts.append(f"""
                <w:p>
                  <w:pPr><w:pStyle w:val="CodeLine"/></w:pPr>
                  <w:r><w:t xml:space="preserve">{escaped_line if escaped_line else " "}</w:t></w:r>
                </w:p>
                """)

    # 1. Image Pipeline Files
    print("Processing Image Pipeline files...")
    add_section("PART 1: IMAGE PIPELINE SOURCE CODE", image_files)

    # 2. Video Pipeline Files
    print("Processing Video Pipeline files...")
    add_section("PART 2: VIDEO PIPELINE SOURCE CODE", video_files)

    # 3. Shared Modules
    print("Processing Shared Visualization module...")
    add_section("PART 3: SHARED VISUALIZATION ENGINE", shared_files)

    # Close document body
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {''.join(body_xml_parts)}
    <w:sectPr/>
  </w:body>
</w:document>"""

    # Write Zip (.docx)
    out_path = Path(output_filename)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/_rels/document.xml.rels", doc_rels)
        zf.writestr("word/styles.xml", styles)
        zf.writestr("word/document.xml", document_xml)

    print(f"\nDocument successfully generated: {out_path.resolve()}")
    print(f"Size: {out_path.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    out_file = "Content_Forensics_Source_Code.docx"
    build_docx(out_file)

