"""同一環境で運用基準と候補の全診断を記録し、新しい静的回帰だけを拒否する。"""

from collections import Counter
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

RUFF_FILES = (
    'app/core/config.py', 'app/ai/model_policy.py', 'app/ai/providers/openai/client.py',
    'app/api/internal/model_execution.py', 'tests/test_document_translation_model_opt_in.py',
)


def execute(root, tool, output, label):
    """設定を変更せず完全な診断を保存し、実際のソース行で比較できるキーを返す。"""
    if tool == 'ruff':
        files = [path for path in RUFF_FILES if (root / path).is_file()]
        command = [sys.executable, '-m', 'ruff', 'check', '--output-format=json', *files]
    else:
        command = ['pyright', '--outputjson', '--pythonpath', sys.executable, 'app']
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=180)
    (output / (label + '-' + tool + '.stdout.json')).write_text(result.stdout)
    (output / (label + '-' + tool + '.stderr.txt')).write_text(result.stderr)
    assert result.returncode in (0, 1), (tool, 'TOOL_EXECUTION_FAILED', result.returncode)
    data = json.loads(result.stdout)
    diagnostics = data if tool == 'ruff' else data['generalDiagnostics']
    keys = []
    for diagnostic in diagnostics:
        path = Path(diagnostic['filename' if tool == 'ruff' else 'file']).resolve()
        relative = path.relative_to(root.resolve()).as_posix()
        row = diagnostic['location']['row'] - 1 if tool == 'ruff' else diagnostic['range']['start']['line']
        line = path.read_text(encoding='utf-8-sig').splitlines()[row].strip()
        code = diagnostic['code'] if tool == 'ruff' else diagnostic.get('rule', diagnostic['severity'])
        keys.append((relative, code, diagnostic['message'], line))
    return Counter(keys)


def main():
    """固定基準を安全な一時コピーへ展開し、診断を抑制せず増分を検出する。"""
    baseline = sys.argv[1]
    assert baseline == '4ff987f8f79df9e417e81a8a41e931909e10eef7'
    root = Path.cwd().resolve()
    output = root / 'test-results/document-static'
    output.mkdir(parents=True, exist_ok=True)
    archive = subprocess.check_output(['git', 'archive', '--format=tar', baseline])
    report = {'baseline': baseline, 'baselineDiagnosticsPreserved': True, 'tools': {}}
    with tempfile.TemporaryDirectory(prefix='document-ai-baseline-') as folder:
        original = Path(folder).resolve()
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(original, filter='data')
        for tool in ('ruff', 'pyright'):
            before = execute(original, tool, output, 'baseline')
            after = execute(root, tool, output, 'candidate')
            added = after - before
            removed = before - after
            report['tools'][tool] = {
                'baselineCount': before.total(), 'candidateCount': after.total(),
                'newCount': added.total(), 'resolvedCount': removed.total(),
                'newDiagnostics': [list(key) for key in added.elements()],
                'unchangedDiagnostics': [list(key) for key in (before & after).elements()],
            }
    report['noNewDiagnostics'] = all(tool['newCount'] == 0 for tool in report['tools'].values())
    (output / 'comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    for tool, data in report['tools'].items():
        if data['baselineCount']:
            print('::warning::' + tool + ': retained unchanged baseline diagnostics: ' + str(data['baselineCount']))
    assert report['noNewDiagnostics'], 'NEW_STATIC_DIAGNOSTICS'


if __name__ == '__main__':
    main()
