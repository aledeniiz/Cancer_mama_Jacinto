"""Standard-library checks for the exact static artifact; no ML runtime needed."""
from pathlib import Path
import importlib.util
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('build_pages', ROOT / 'scripts/build_pages.py')
pages = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pages)


class PagesArtifactTests(unittest.TestCase):
    def test_only_exact_reviewed_html_is_deployed(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'site'
            report = pages.stage_pages(output)
            self.assertEqual(set(p.name for p in output.iterdir()), {'index.html', '.nojekyll'})
            self.assertEqual((output / 'index.html').read_bytes(), (ROOT / 'web/index.html').read_bytes())
            self.assertFalse(report['private_data_included'])

    def test_existing_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileExistsError):
                pages.stage_pages(Path(temp))

    def test_repository_is_never_a_build_destination(self):
        with self.assertRaises(ValueError):
            pages.stage_pages(ROOT)


if __name__ == '__main__':
    unittest.main()
