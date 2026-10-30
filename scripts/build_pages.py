"""Package only the reviewed self-contained web page, never the whole repository."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import shutil
from html.parser import HTMLParser

ROOT = Path(__file__).resolve().parents[1]
DEPLOYED_ASSETS = frozenset({
    'ort.min.js', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm',
    'ORT_LICENSE.txt', 'ORT_ThirdPartyNotices.txt', 'runtime-provenance.json',
    'model-cnn-ema.onnx', 'model-classifier.onnx', 'model-fusion.onnx', 'model-config.json',
})


def reviewed_assets():
    directory = ROOT / 'web/inference-assets'
    manifest_path = directory / 'asset-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    rows = manifest['files']
    names = [row['path'] for row in rows]
    if len(names) != len(set(names)) or set(names) != DEPLOYED_ASSETS:
        raise ValueError('The browser runtime must contain exactly the reviewed model/runtime assets')
    for row in rows:
        path = directory / row['path']
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing or unreviewed browser asset')
        content = path.read_bytes()
        if len(content) != row['size'] or hashlib.sha256(content).hexdigest() != row['sha256']:
            raise ValueError('Browser asset hash/size mismatch: ' + row['path'])
    return directory, names


class DependencyCheck(HTMLParser):
    def __init__(self):
        super().__init__()
        self.external_resources = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag in ('script', 'img', 'iframe', 'source', 'video', 'audio'):
            source = values.get('src')
            if source and not source.startswith('data:'):
                self.external_resources.append(source)
        if tag == 'link' and values.get('rel') in ('stylesheet', 'preload', 'modulepreload'):
            self.external_resources.append(values.get('href', ''))


def stage_pages(output):
    output = Path(output).resolve()
    if output == ROOT or ROOT.is_relative_to(output):
        raise ValueError('The output must not be the repository or an ancestor')
    if output.is_relative_to(ROOT) and output != ROOT / '.pages-site':
        raise ValueError('Use .pages-site or a new output directory outside the checkout')
    if output.exists():
        raise FileExistsError('Use a new Pages artifact directory; existing files are never replaced')
    source = ROOT / 'web/index.html'
    if not source.is_file():
        raise FileNotFoundError('Reviewed public web/index.html is missing')
    spec = importlib.util.spec_from_file_location('public_web_builder', ROOT / 'web/build_public.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    page = source.read_text(encoding='utf-8')
    if page != builder.render():
        raise ValueError('The public web is stale; rebuild and review it before deploying')
    data = json.loads((ROOT / 'web/project-data.json').read_text(encoding='utf-8'))
    if data.get('patients') or data.get('normalization', {}).get('available'):
        raise ValueError('Private patients or fitted normalization must not be deployed')
    if data.get('normalization', {}).get('means') or data.get('normalization', {}).get('std'):
        raise ValueError('Fitted statistics must not be deployed')
    multimodal = json.loads((ROOT / 'web/multimodal-public.json').read_text(encoding='utf-8'))
    if multimodal.get('ready'):
        raise ValueError('This Pages export must not advertise a private inference backend')
    parser = DependencyCheck()
    parser.feed(page)
    if parser.external_resources:
        raise ValueError('The reviewed Pages page must be self-contained')
    asset_directory, asset_names = reviewed_assets()
    output.mkdir(parents=True, exist_ok=False)
    content = source.read_bytes()
    with (output / 'index.html').open('xb') as handle:
        handle.write(content)
    with (output / '.nojekyll').open('xb'):
        pass
    (output / 'inference-assets').mkdir()
    for name in asset_names + ['asset-manifest.json']:
        shutil.copyfile(asset_directory / name, output / 'inference-assets' / name)
    assert (output / 'index.html').read_bytes() == content
    return {'files': ['index.html', '.nojekyll'] + ['inference-assets/' + name for name in asset_names + ['asset-manifest.json']], 'index_sha256': hashlib.sha256(content).hexdigest(),
            'index_bytes': len(content), 'private_data_included': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / '.pages-site')
    args = parser.parse_args()
    try:
        print(json.dumps(stage_pages(args.output)))
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f'Error: {exc}\n')
