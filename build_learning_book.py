"""Build the separate beginner quant textbook from editable chapter sources."""

import argparse
from datetime import datetime
from html import escape
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether, PageBreak,
                               PageTemplate, Paragraph, Preformatted, Spacer, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

from reporting import REPORT_TIMEZONE
from research_report_pdf import _fonts, _inline, _cells, _table, render_pages

ROOT = Path(__file__).resolve().parent


def make_figures():
    directory = ROOT / 'learning/figures'
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.set(xlim=(0, 3), ylim=(0, 2))
    ax.axis('off')
    boxes = [('1  Data', 'Universe and dates'), ('2  Formation', 'Fit using earlier prices'),
             ('3  Signal', 'Deviation and rules'), ('4  Execution', 'Trade at a later close'),
             ('5  Account', 'Shares cash and costs'), ('6  Evaluation', 'Evidence and limits')]
    for i, (title, subtitle) in enumerate(boxes):
        x, y = (i % 3) + .06, 1.11 if i < 3 else .11
        ax.add_patch(FancyBboxPatch((x, y), .87, .74, boxstyle='round,pad=0.025',
                                    facecolor='#eff4f8', edgecolor='#b9cbd8'))
        ax.text(x + .435, y + .47, title, ha='center', va='center', fontsize=14, weight='bold', color='#234c72')
        ax.text(x + .435, y + .22, subtitle, ha='center', va='center', fontsize=10)
        if i % 3 != 2:
            ax.annotate('', xy=(x + .995, y + .37), xytext=(x + .9, y + .37),
                        arrowprops={'arrowstyle': '->', 'color': '#234c72', 'lw': 1.3})
    ax.text(1.5, 1.0, 'Steps 1-3 define the decision. Steps 4-6 measure what follows.',
            ha='center', va='center', fontsize=10, color='#556778')
    fig.tight_layout(pad=.1)
    fig.savefig(directory / 'research_process.png', dpi=190, bbox_inches='tight')
    plt.close(fig)

    x, y = np.array([10, 11, 12, 13]), np.array([22, 24, 27, 29])
    fit = -2.1 + 2.4 * x
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.plot(x, fit, color='#24688f', label='Fitted line: A = -2.1 + 2.4 B')
    ax.scatter(x, y, color='#172b3a', s=32, zorder=3, label='Observations')
    ax.vlines(x, fit, y, color='#b5682b', lw=2)
    ax.set(xlabel='B price in dollars', ylabel='A price in dollars', xticks=x)
    ax.grid(alpha=.18)
    ax.legend(fontsize=10, loc='upper left')
    fig.tight_layout()
    fig.savefig(directory / 'ols_example.png', dpi=190, bbox_inches='tight')
    plt.close(fig)

    rng = np.random.default_rng(2026)
    n = 240
    stationary = np.zeros(n)
    stationary[0] = rng.normal(0, .6 / np.sqrt(1 - .8 ** 2))
    for t in range(1, n):
        stationary[t] = .8 * stationary[t - 1] + rng.normal(0, .6)
    wandering = np.cumsum(rng.normal(0, .6, n))
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 4.4), sharex=True, sharey=True)
    for ax, data, label, color in zip(axes, [stationary, wandering],
                                    ['Stationary construction: u(t) = 0.8 u(t-1) + noise',
                                     'Random-walk construction: v(t) = v(t-1) + noise'],
                                    ['#24688f', '#ad632a']):
        ax.plot(data, color=color, linewidth=1.4)
        ax.axhline(0, color='#94a3b8', lw=.8)
        ax.set_title(label, fontsize=11, loc='left')
        ax.set_ylabel('Spread units')
        ax.grid(alpha=.15)
    axes[-1].set_xlabel('Simulated observation number')
    fig.tight_layout()
    fig.savefig(directory / 'spread_examples.png', dpi=190, bbox_inches='tight')
    plt.close(fig)


def styles():
    regular, bold, mono = _fonts()
    common = dict(fontName=regular, textColor=colors.HexColor('#172b3a'),
                  allowWidows=False, allowOrphans=False, splitLongWords=True)
    definitions = {
        'Body': dict(fontSize=10.2, leading=14.7, spaceAfter=8),
        'Title': dict(fontName=bold, fontSize=27, leading=33, spaceAfter=20, textColor=colors.HexColor('#234c72')),
        'H1': dict(fontName=bold, fontSize=19, leading=24, spaceBefore=15, spaceAfter=13, keepWithNext=True, textColor=colors.HexColor('#234c72')),
        'H2': dict(fontName=bold, fontSize=12.3, leading=17, spaceBefore=11, spaceAfter=7, keepWithNext=True),
        'H3': dict(fontName=bold, fontSize=10.5, leading=15, spaceBefore=7, spaceAfter=5, keepWithNext=True),
        'Bullet': dict(fontSize=10.2, leading=14.7, leftIndent=13, firstLineIndent=-10, spaceAfter=5),
        'Caption': dict(fontSize=8.7, leading=12, textColor=colors.HexColor('#556778'), spaceBefore=6, spaceAfter=12),
        'Code': dict(fontName=mono, fontSize=8.1, leading=12, backColor=colors.HexColor('#eff4f8'), borderPadding=9, spaceAfter=12),
        'TableBody': dict(fontSize=9.1, leading=12.3),
        'TableHead': dict(fontName=bold, fontSize=9.1, leading=12.3, textColor=colors.white),
        'TableSmall': dict(fontSize=8.5, leading=11.6),
        'TableHeadSmall': dict(fontName=bold, fontSize=8.5, leading=11.6, textColor=colors.white),
        'TOCHeading': dict(fontName=bold, fontSize=23, leading=28, spaceAfter=20),
    }
    return {name: ParagraphStyle(name, **(common | definition)) for name, definition in definitions.items()}


class LearningDoc(BaseDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and hasattr(flowable, '_bookmarkName'):
            key = flowable._bookmarkName
            self.canv.bookmarkPage(key)
            self.canv.bookmarkPage(flowable._headingSlug)
            self.canv.addOutlineEntry(flowable.getPlainText(), key, level=0, closed=False)
            self.notify('TOCEntry', (0, flowable.getPlainText(), self.page, key))


def render_book(markdown, output):
    theme = styles()
    margin = 48
    doc = LearningDoc(str(output), pagesize=A4, leftMargin=margin, rightMargin=margin,
                      topMargin=53, bottomMargin=48, title='A learning book for your quant project',
                      author='Pairs trading project learning edition')

    def page(canvas, document):
        canvas.saveState()
        canvas.setFont('ResearchSans', 8)
        canvas.setFillColor(colors.HexColor('#556778'))
        canvas.drawString(margin, A4[1] - 29, 'QUANT PROJECT LEARNING BOOK')
        canvas.setStrokeColor(colors.HexColor('#d4dfe7'))
        canvas.line(margin, A4[1] - 37, A4[0] - margin, A4[1] - 37)
        canvas.line(margin, 33, A4[0] - margin, 33)
        canvas.drawString(margin, 21, 'Concepts - worked examples - procedure - dictionary')
        canvas.drawRightString(A4[0] - margin, 21, str(document.page))
        canvas.restoreState()

    frame = Frame(margin, doc.bottomMargin, doc.width, doc.height, leftPadding=0,
                  rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id='Book', frames=[frame], onPage=page))
    story = []
    lines = markdown.read_text().splitlines()
    index, section = 0, 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line == '<!-- contents -->':
            story += [PageBreak(), Paragraph('Contents', theme['TOCHeading'])]
            toc = TableOfContents()
            toc.levelStyles = [ParagraphStyle('TOCEntry', fontName='ResearchSans', fontSize=10.4,
                                             leading=14, spaceBefore=4, textColor=colors.HexColor('#234c72'))]
            story += [toc]
            index += 1
            continue
        heading = re.match(r'^(#{1,6})\s+(.+)', line)
        if heading:
            level, text = len(heading[1]), heading[2]
            style = 'Title' if level == 1 else f'H{min(level-1,3)}'
            if level == 2:
                story.append(PageBreak() if section == 0 or text in {'Dictionary', 'Exercises', 'Worked answers'} else CondPageBreak(265))
            paragraph = Paragraph(_inline(text, markdown.parent), theme[style])
            if level == 2:
                section += 1
                paragraph._bookmarkName = f'chapter-{section}'
                paragraph._headingSlug = re.sub(r'[^\w\s-]', '', text.lower()).replace(' ', '-')
            story.append(paragraph)
            index += 1
            continue
        if line.startswith('<!--'):
            index += 1
            continue
        if line.startswith('```'):
            index += 1
            code = []
            while index < len(lines) and not lines[index].startswith('```'):
                code.append(lines[index])
                index += 1
            story.append(KeepTogether([Preformatted('\n'.join(code), theme['Code'], maxLineLength=89)]))
            index += 1
            continue
        picture = re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)', line)
        if picture:
            caption, target = picture.groups()
            visual = Image(str((markdown.parent / target).resolve()))
            scale = min(doc.width / visual.imageWidth, 265 / visual.imageHeight)
            visual.drawWidth, visual.drawHeight = visual.imageWidth * scale, visual.imageHeight * scale
            visual.hAlign = 'CENTER'
            story.append(KeepTogether([Spacer(1, 4), visual, Paragraph(_inline(caption, markdown.parent), theme['Caption'])]))
            index += 1
            continue
        if line.startswith('|') and index + 1 < len(lines) and re.match(r'^\s*\|?\s*:?-{2,}', lines[index + 1]):
            rows = [_cells(line)]
            index += 2
            while index < len(lines) and lines[index].strip().startswith('|'):
                rows.append(_cells(lines[index]))
                index += 1
            grid = _table(rows, doc.width, theme, markdown.parent, profile='clean')
            grid.setStyle(TableStyle([('GRID', (0,0), (-1,-1), .35, colors.HexColor('#d9d9d9'))]))
            story += [grid, Spacer(1, 11)]
            continue
        bullet = re.match(r'^(?:([-*+])\s+|(\d+[.)])\s+)(.+)', line)
        if bullet:
            prefix = '- ' if bullet[1] else bullet[2] + ' '
            story.append(Paragraph(prefix + _inline(bullet[3], markdown.parent), theme['Bullet']))
            index += 1
            continue
        parts = [line.lstrip('> ') if line.startswith('>') else line]
        index += 1
        while index < len(lines) and lines[index].strip():
            following = lines[index].strip()
            if following.startswith(('#','|','![','```','<!--')) or re.match(r'^([-*+]\s+|\d+[.)]\s+)', following):
                break
            parts.append(following)
            index += 1
        story.append(Paragraph(_inline(' '.join(parts), markdown.parent), theme['Body']))
    doc.multiBuild(story)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-stem', default=None)
    parser.add_argument('--render-dir', type=Path, default=ROOT / 'tmp/pdfs/learning-book')
    args = parser.parse_args()
    appendix = ROOT / 'learning/glossary_and_exercises.md'
    if not appendix.exists():
        raise FileNotFoundError('The learning glossary and exercise appendix is not yet available.')
    make_figures()
    core = (ROOT / 'learning/core.md').read_text()
    core = core.replace('\n## 1 ', '\n<!-- contents -->\n\n## 1 ', 1)
    core = core.replace('](figures/', '](../../learning/figures/')
    extra = appendix.read_text()
    if extra.startswith('# '):
        extra = '\n'.join(extra.splitlines()[1:]).lstrip()
    introduction, separator, dictionary = extra.partition('## Dictionary\n')
    if separator:
        extra = '## Dictionary\n\n' + introduction.strip() + '\n\n' + dictionary.lstrip()
    moment = datetime.now(REPORT_TIMEZONE)
    stem = args.output_stem or f'{moment:%Y-%m-%d_%H-%M-%S}_quant_project_learning_book'
    output = ROOT / 'output/pdf'
    output.mkdir(parents=True, exist_ok=True)
    manuscript = output / f'{stem}.md'
    pdf = output / f'{stem}.pdf'
    provenance = ('\n\n## Project evidence and further reading\n\n'
                  f'Learning edition prepared {moment:%B %d, %Y}. All results refer to saved historical simulations. '
                  'The original assignment is Simple Pairs Trading Strategy Using Cointegration, supplied in 4. Pairs Trading Strategy.pdf. '
                  'The textbook follows its statistical topics and explains which research requirements remain incomplete.\n\n'
                  '- [Project summary and original report links](../../PROJECT_SUMMARY.md)\n'
                  '- [Dated reports and full data records](../../REPORTS.md)\n'
                  '- [Exact OLS strategy plan](../../OLS_STRATEGY_PLAN.md)\n'
                  '- [Online examples and their limitations](../../ONLINE_LEARNING.md)\n'
                  '- [Chart presentation guidance](https://analysisfunction.civilservice.gov.uk/policy-store/data-visualisation-charts/)\n'
                  '- [Table presentation guidance](https://analysisfunction.civilservice.gov.uk/policy-store/data-visualisation-tables/)\n\n'
                  'The worked numerical examples and synthetic figures were created for this learning book. They are not observed returns. '
                  'The sources support specific definitions and presentation practices; they do not validate this project\'s trading profitability.\n')
    manuscript.write_text(core + '\n\n' + extra + provenance)
    render_book(manuscript, pdf)
    pages = render_pages(pdf, args.render_dir, dpi=110)
    print(f'Manuscript: {manuscript}\nPDF: {pdf}\nRendered pages: {len(pages)}\nQA directory: {args.render_dir}')


if __name__ == '__main__':
    main()
