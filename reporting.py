"""Consistent, non-overwriting names for research and backtest reports."""

from datetime import datetime
from pathlib import Path
import re
from zoneinfo import ZoneInfo


REPORT_TIMEZONE = ZoneInfo('America/New_York')


def _slug(value):
    result = re.sub(r'[^a-z0-9]+', '-', str(value).lower()).strip('-')
    if not result:
        raise ValueError('Report names must contain letters or numbers.')
    return result


def create_run_directory(base_dir, run_name, stock_count, *, started_at=None):
    """Create a dated run directory; same-second collisions get a suffix.

    The date is the run date in New York, not the market-data period.
    Legacy naive timestamps are interpreted as New York local time.
    """
    when = started_at or datetime.now(REPORT_TIMEZONE)
    when = (when.replace(tzinfo=REPORT_TIMEZONE) if when.tzinfo is None
            else when.astimezone(REPORT_TIMEZONE))
    if not isinstance(stock_count, int) or stock_count < 0:
        raise ValueError('stock_count must be a nonnegative integer.')
    stem = f'{when:%Y-%m-%d_%H-%M-%S}_{_slug(run_name)}_{stock_count}-stocks'
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        directory = base / (stem if number == 1 else f'{stem}-{number:02d}')
        try:
            directory.mkdir()
            return directory
        except FileExistsError:
            number += 1


def report_path(run_dir, kind, extension):
    """Give each file its run's date, name and stock count, plus a description."""
    directory = Path(run_dir)
    suffix = str(extension).lstrip('.')
    if not re.fullmatch(r'[a-zA-Z0-9]+', suffix):
        raise ValueError('Expected a filename extension such as md, json or png.')
    return directory / f'{directory.name}_{_slug(kind)}.{suffix.lower()}'


def update_report_index():
    """Refresh the project's clickable catalog of dated report folders."""
    root = Path(__file__).resolve().parent
    directories = []
    for base in (root / 'results', root / 'output'):
        if base.exists():
            directories.extend(p for p in base.rglob('*') if p.is_dir() and
                               not p.name.endswith('_figures') and
                               re.match(r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_', p.name))
    directories = sorted(directories, key=lambda p: (p.name, str(p)), reverse=True)
    clean_reports = [p for p in directories if 'pairs-trading-clean-report' in p.name
                     and report_path(p, 'report', 'md').exists() and (p / 'START_HERE.md').exists()]
    latest = clean_reports[0] if clean_reports else None
    experiments = [p for p in directories if 'online-learning-walk-forward' in p.name
                   and report_path(p, 'report', 'md').exists()]
    latest_experiment = experiments[0] if experiments else None
    improvements = [p for p in directories if 'ols-strategy-improvement' in p.name
                    and report_path(p, 'report', 'md').exists()]
    latest_improvement = improvements[0] if improvements else None
    forecast_experiments = [p for p in directories if 'linear-ecm-forecast-gate' in p.name
                            and report_path(p, 'report', 'md').exists()]
    latest_forecast = forecast_experiments[0] if forecast_experiments else None
    featured = tuple(p for p in (latest, latest_experiment, latest_improvement,
                                latest_forecast) if p is not None)
    lines = ['# Report index', '']
    documents = root / 'output' / 'pdf'
    learning_books = sorted(documents.glob('*_quant_project_learning_book.pdf'), reverse=True)
    briefs = sorted(documents.glob('*_pairs_trading_project_brief.pdf'), reverse=True)
    if learning_books or briefs:
        lines.extend(['## Learning and presentation', ''])
        for label, candidates in [('Quant project learning book', learning_books),
                                  ('Project presentation brief', briefs)]:
            if candidates:
                pdf = candidates[0]
                line = f'- [{label}]({pdf.relative_to(root).as_posix()})'
                markdown = pdf.with_suffix('.md')
                if markdown.exists():
                    line += f' | [Editable Markdown]({markdown.relative_to(root).as_posix()})'
                lines.append(line)
        lines.extend(['', 'The learning book explains the concepts, calculations, procedure and vocabulary. '
                      'The brief presents measured results, the additional forecast strategy and assignment coverage.', ''])
    if latest:
        lines.extend(['## Current clean report', '',
                      f'[{latest.name}]({(latest / "START_HERE.md").relative_to(root).as_posix()})', '',
                      'Strategy, required charts and statistics, and profit for every selected pair. '
                      'Full records are in its data folder; older runs are grouped below.', ''])
        for label, path in [
            ('Read the PDF', report_path(latest, 'report', 'pdf')),
            ('Read the Markdown report', report_path(latest, 'report', 'md')),
            ('Profit and statistics for every pair', latest / 'data' / report_path(latest, 'pair-profits-and-statistics', 'csv').name),
            ('Complete later-period trade log', latest / 'data' / report_path(latest, 'later-trade-log', 'csv').name),
        ]:
            if path.exists():
                lines.append(f'- [{label}]({path.relative_to(root).as_posix()})')
    if latest_experiment:
        experiment_report = report_path(latest_experiment, 'report', 'md')
        lines.extend(['', '## Learning experiment', '',
                      f'[Rolling pair selection comparison]({experiment_report.relative_to(root).as_posix()})', '',
                      'A separate pilot with fixed selection, monthly selection and a stricter multiple-testing filter. '
                      'See [ONLINE_LEARNING.md](ONLINE_LEARNING.md) for the primary examples and adaptations.', ''])
    if latest_improvement:
        improvement_report = report_path(latest_improvement, 'report', 'md')
        lines.extend(['', '## OLS strategy improvement', '',
                      f'[Five strategy variants and earlier-window selection]({improvement_report.relative_to(root).as_posix()})', '',
                      'Keeps linear regression and tests pair stability, entry checks and allocation limits. '
                      'See [OLS_STRATEGY_PLAN.md](OLS_STRATEGY_PLAN.md) for the exact rules.', ''])
    if latest_forecast:
        forecast_report = report_path(latest_forecast, 'report', 'md')
        lines.extend(['', '## Linear spread forecast experiment', '',
                      f'[Forecast entry gate versus the monthly baseline]({forecast_report.relative_to(root).as_posix()})', '',
                      'Keeps the OLS hedge ratio and tests whether a second linear regression can filter entries '
                      'using predicted spread recovery and a cost hurdle. The report includes the frozen '
                      'earlier-window selection, complete trade logs and pair profits.', ''])
    if featured:
        lines.extend(['', '<details>', '<summary>Earlier runs and underlying research files</summary>', ''])
    lines.extend([
             'Names use the run date and time in America/New_York, report name, and number of stocks used. '
             'The historical data period is recorded inside each report or its metadata.', '',
             'New runs get separate folders. A numeric suffix distinguishes runs started in the same second. '
             'Recovered older reports identify their timestamp source in archive metadata.', ''])
    for directory in directories:
        if directory in featured:
            continue
        lines.extend([f'## {directory.name}', ''])
        for file in sorted(directory.iterdir()):
            if file.is_file():
                description = file.name.removeprefix(directory.name + '_')
                lines.append(f'- [{description}]({file.relative_to(root).as_posix()})')
        lines.append('')
    if featured:
        lines.extend(['</details>', ''])
    (root / 'REPORTS.md').write_text('\n'.join(lines) + '\n')
