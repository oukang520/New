"""Create the authored Chinese research explanation as a portable DOCX.

Uses only the Python standard library. Tables remain native editable Word tables;
relative source links are preserved. No experimental output is modified.
"""
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/RESEARCH_STORY_CN.md'
OUTPUT = ROOT / 'docs/RESEARCH_STORY_CN.docx'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
REL = 'http://schemas.openxmlformats.org/package/2006/relationships'
ET.register_namespace('w', W)
ET.register_namespace('r', R)


def add(parent, name, attrs=None, text=None):
    element = ET.SubElement(parent, '{%s}%s' % (W, name),
                            {'{%s}%s' % (W, k): str(v) for k, v in (attrs or {}).items()})
    element.text = text
    return element


def xml(element):
    return ET.tostring(element, encoding='utf-8', xml_declaration=True)


def parse_blocks():
    lines = SOURCE.read_text(encoding='utf-8').splitlines()
    blocks, i = [], 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        heading = re.match(r'^(#{1,3}) (.*)$', line)
        if heading:
            blocks.append(('heading', len(heading[1]), heading[2]))
            i += 1
        elif line == r'\[':
            formula = []
            i += 1
            while i < len(lines) and lines[i].strip() != r'\]':
                formula.append(lines[i].strip())
                i += 1
            if i == len(lines):
                raise ValueError('Unclosed display equation')
            latex = ' '.join(formula)
            if r'\lambda_i' in latex:
                display = 'λᵢ(x) = exp(βᵢ + Σⱼ≠ᵢ θᵢ←ⱼ xⱼ)'
            elif 'q_e=' in latex:
                display = 'qₑ = Aₑ [pϑ ⊙ exp(Fₑφₑ)] / Zₑ\nZₑ = 1ᵀ[pϑ ⊙ exp(Fₑφₑ)]'
            else:
                raise ValueError('Add an explicit readable rendering for this equation: ' + latex)
            blocks.append(('equation', display))
            i += 1
        elif line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', c) for c in cells):
                    rows.append(cells)
                i += 1
            if any(len(row) != len(rows[0]) for row in rows):
                raise ValueError('Table columns are inconsistent')
            blocks.append(('table', rows))
        else:
            paragraph = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(('#', '|', r'\[')):
                paragraph.append(lines[i].strip())
                i += 1
            blocks.append(('paragraph', ' '.join(paragraph)))
    return blocks


class Writer:
    def __init__(self):
        self.document = ET.Element('{%s}document' % W)
        self.body = add(self.document, 'body')
        self.relationships = ET.Element('{%s}Relationships' % REL)
        for name, filename in [('styles', 'styles.xml'), ('settings', 'settings.xml'), ('footer', 'footer1.xml')]:
            self.relationship(name, R + '/' + name, filename)
        self.links = {}

    def relationship(self, identifier, kind, target, external=False):
        attrs = {'Id': identifier, 'Type': kind, 'Target': target}
        if external:
            attrs['TargetMode'] = 'External'
        ET.SubElement(self.relationships, '{%s}Relationship' % REL, attrs)

    def run(self, parent, text, bold=False, code=False, link=False):
        run = add(parent, 'r')
        if bold or code or link:
            props = add(run, 'rPr')
            if bold:
                add(props, 'b')
            if code:
                add(props, 'rFonts', {'ascii': 'Consolas', 'hAnsi': 'Consolas', 'eastAsia': '宋体'})
                add(props, 'sz', {'val': 20})
            if link:
                add(props, 'color', {'val': '0072B2'})
        value = add(run, 't', text=text)
        value.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')

    def inline(self, parent, value, bold=False):
        cursor = 0
        pattern = r'(\*\*.*?\*\*|\[[^\]]+\]\([^)]+\)|`[^`]+`)'
        for match in re.finditer(pattern, value):
            self.run(parent, value[cursor:match.start()], bold=bold)
            token = match[0]
            if token.startswith('**'):
                self.run(parent, token[2:-2], bold=True)
            elif token.startswith('`'):
                self.run(parent, token[1:-1], bold=bold, code=True)
            else:
                label, target = re.fullmatch(r'\[([^\]]+)\]\(([^)]+)\)', token).groups()
                if target not in self.links:
                    identifier = 'link%d' % (len(self.links) + 1)
                    self.links[target] = identifier
                    self.relationship(identifier, R + '/hyperlink', target, external=True)
                hyperlink = add(parent, 'hyperlink')
                hyperlink.set('{%s}id' % R, self.links[target])
                self.run(hyperlink, label, bold=bold, link=True)
            cursor = match.end()
        self.run(parent, value[cursor:], bold=bold)

    def paragraph(self, parent, value, style='Normal', bold=False):
        paragraph = add(parent, 'p')
        add(add(paragraph, 'pPr'), 'pStyle', {'val': style})
        for index, line in enumerate(value.split('\n')):
            if index:
                add(add(paragraph, 'r'), 'br')
            self.inline(paragraph, line, bold=bold)
        return paragraph

    def table(self, rows):
        count = len(rows[0])
        widths = {3: [2200, 3600, 3838], 4: [800, 2946, 2946, 2946],
                  5: [800, 1900, 3200, 2800, 938]}.get(count)
        if count == 4 and rows[0][0] == '主要策略':
            widths = [3838, 1933, 1933, 1934]
        if widths is None:
            widths = [9638 // count] * count
            widths[-1] += 9638 - sum(widths)
        table = add(self.body, 'tbl')
        props = add(table, 'tblPr')
        add(props, 'tblW', {'w': 9638, 'type': 'dxa'})
        add(props, 'tblLayout', {'type': 'fixed'})
        borders = add(props, 'tblBorders')
        for edge in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
            add(borders, edge, {'val': 'single', 'sz': 4, 'color': 'B7C6CE'})
        margins = add(props, 'tblCellMar')
        for edge in ['top', 'left', 'bottom', 'right']:
            add(margins, edge, {'w': 80, 'type': 'dxa'})
        grid = add(table, 'tblGrid')
        for width in widths:
            add(grid, 'gridCol', {'w': width})
        for index, row in enumerate(rows):
            tr = add(table, 'tr')
            row_props = add(tr, 'trPr')
            add(row_props, 'cantSplit')
            if index == 0:
                add(row_props, 'tblHeader')
            for width, value in zip(widths, row):
                cell = add(tr, 'tc')
                cell_props = add(cell, 'tcPr')
                add(cell_props, 'tcW', {'w': width, 'type': 'dxa'})
                if index == 0:
                    add(cell_props, 'shd', {'val': 'clear', 'fill': 'EAF3F8'})
                add(cell_props, 'vAlign', {'val': 'top'})
                self.paragraph(cell, value, 'TableText', bold=index == 0)
        self.paragraph(self.body, '')

    def build(self, blocks):
        subtitle = True
        for block in blocks:
            if block[0] == 'heading':
                _, level, value = block
                style = 'Title' if level == 1 else 'Heading1' if level == 2 else 'Heading2'
                if level == 2 and subtitle:
                    style, subtitle = 'Subtitle', False
                self.paragraph(self.body, value, style)
            elif block[0] == 'table':
                self.table(block[1])
            else:
                self.paragraph(self.body, block[1], 'Equation' if block[0] == 'equation' else 'Normal')
        section = add(self.body, 'sectPr')
        footer = add(section, 'footerReference', {'type': 'default'})
        footer.set('{%s}id' % R, 'footer')
        add(section, 'pgSz', {'w': 11906, 'h': 16838})
        add(section, 'pgMar', {'top': 1134, 'right': 1134, 'bottom': 1134, 'left': 1134,
                               'header': 567, 'footer': 567, 'gutter': 0})


def styles():
    root = ET.Element('{%s}styles' % W)
    defaults = add(root, 'docDefaults')
    rp = add(add(defaults, 'rPrDefault'), 'rPr')
    add(rp, 'rFonts', {'ascii': 'Arial', 'hAnsi': 'Arial', 'eastAsia': '宋体'})
    add(rp, 'sz', {'val': 22})
    add(rp, 'lang', {'val': 'zh-CN', 'eastAsia': 'zh-CN'})
    specifications = [('Normal', 22, '202020', False), ('Title', 40, '0072B2', True),
                      ('Subtitle', 26, '606060', False), ('Heading1', 28, '0072B2', True),
                      ('Heading2', 24, '202020', True), ('TableText', 19, '202020', False),
                      ('Equation', 22, '202020', False)]
    for name, size, color, bold in specifications:
        props = {'type': 'paragraph', 'styleId': name}
        if name == 'Normal':
            props['default'] = 1
        style = add(root, 'style', props)
        add(style, 'name', {'val': name})
        if name != 'Normal':
            add(style, 'basedOn', {'val': 'Normal'})
        pp = add(style, 'pPr')
        heading = name in ['Title', 'Subtitle', 'Heading1', 'Heading2']
        if heading:
            add(pp, 'keepNext')
        add(pp, 'spacing', {'before': 180 if heading else 0, 'after': 80 if name == 'TableText' else 120,
                            'line': 270 if name == 'TableText' else 330, 'lineRule': 'auto'})
        if name.startswith('Heading'):
            add(pp, 'outlineLvl', {'val': int(name[-1]) - 1})
        if name == 'Equation':
            add(pp, 'jc', {'val': 'center'})
        rp = add(style, 'rPr')
        if bold:
            add(rp, 'b')
        add(rp, 'sz', {'val': size})
        add(rp, 'color', {'val': color})
    return root


def main():
    blocks = parse_blocks()
    writer = Writer()
    writer.build(blocks)
    settings = ET.Element('{%s}settings' % W)
    add(settings, 'updateFields', {'val': 'true'})
    footer = ET.Element('{%s}ftr' % W)
    paragraph = add(footer, 'p')
    add(add(paragraph, 'pPr'), 'jc', {'val': 'center'})
    field = add(paragraph, 'fldSimple', {'instr': ' PAGE '})
    writer.run(field, '1')
    content_types = b'''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
<Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>
</Types>'''
    root_rels = ('<?xml version="1.0" encoding="UTF-8"?>'
                 '<Relationships xmlns="%s"><Relationship Id="document" Type="%s/officeDocument" '
                 'Target="word/document.xml"/></Relationships>' % (REL, R)).encode('utf-8')
    with ZipFile(OUTPUT, 'w', ZIP_DEFLATED) as archive:
        for name, value in {'[Content_Types].xml': content_types, '_rels/.rels': root_rels,
                            'word/document.xml': xml(writer.document), 'word/styles.xml': xml(styles()),
                            'word/settings.xml': xml(settings), 'word/footer1.xml': xml(footer),
                            'word/_rels/document.xml.rels': xml(writer.relationships)}.items():
            archive.writestr(name, value)
    print('Created %s: %d editable tables, %d source links' %
          (OUTPUT.relative_to(ROOT), sum(b[0] == 'table' for b in blocks), len(writer.links)))


if __name__ == '__main__':
    main()
