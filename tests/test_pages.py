"""Standard-library checks for the exact static artifact; no ML runtime needed."""
from pathlib import Path
import importlib.util
import tempfile
import unittest
import json
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('build_pages', ROOT / 'scripts/build_pages.py')
pages = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pages)


class PagesArtifactTests(unittest.TestCase):
    def test_only_exact_reviewed_html_is_deployed(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'site'
            report = pages.stage_pages(output)
            self.assertEqual(set(p.name for p in output.iterdir()), {'index.html', '.nojekyll', 'inference-assets'})
            self.assertEqual(set(p.name for p in (output/'inference-assets').iterdir()), pages.DEPLOYED_ASSETS | {'asset-manifest.json'})
            for name in pages.DEPLOYED_ASSETS:
                self.assertEqual((output/'inference-assets'/name).read_bytes(), (ROOT/'web/inference-assets'/name).read_bytes())
            self.assertEqual((output / 'index.html').read_bytes(), (ROOT / 'web/index.html').read_bytes())
            self.assertFalse(report['private_data_included'])

    def test_existing_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileExistsError):
                pages.stage_pages(Path(temp))

    def test_repository_is_never_a_build_destination(self):
        with self.assertRaises(ValueError):
            pages.stage_pages(ROOT)

    def test_changed_runtime_asset_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);assets=root/'web/inference-assets';assets.mkdir(parents=True)
            (assets/'model.onnx').write_bytes(b'changed')
            (assets/'asset-manifest.json').write_text(json.dumps({'files':[{'path':'model.onnx','size':7,'sha256':'0'*64}]}))
            with patch.object(pages,'ROOT',root), patch.object(pages,'DEPLOYED_ASSETS',{'model.onnx'}):
                with self.assertRaisesRegex(ValueError,'hash/size mismatch'):
                    pages.reviewed_assets()

    def test_manifest_cannot_include_an_unreviewed_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);assets=root/'web/inference-assets';assets.mkdir(parents=True)
            (assets/'asset-manifest.json').write_text(json.dumps({'files':[{'path':'../private.json','size':0,'sha256':'0'*64}]}))
            with patch.object(pages,'ROOT',root):
                with self.assertRaisesRegex(ValueError,'exactly the reviewed'):
                    pages.reviewed_assets()


if __name__ == '__main__':
    unittest.main()
