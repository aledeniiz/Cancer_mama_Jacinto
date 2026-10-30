"""Reviewed scientific data must stay byte-identical; exceptions are path-bound."""
from pathlib import Path
import hashlib
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('release_audit',Path(__file__).resolve().parents[1]/'scripts/audit_release.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ScientificAllowlistTests(unittest.TestCase):
    def test_exact_csv_allowed_but_changed_csv_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'evaluation').mkdir();(root/'docs').mkdir()
            path=root/'evaluation/reference.csv';content=b'patient_id,label\nexample,1\n'
            path.write_bytes(content)
            (root/'docs/scientific_artifact_allowlist.json').write_text(json.dumps({'files':[{
                'path':'evaluation/reference.csv','sha256':hashlib.sha256(content).hexdigest(),'size':len(content)}]}))
            with patch.object(audit,'ROOT',root):
                self.assertEqual(audit.audit()[0]['status'],'passed')
                path.write_bytes(content+b'other,0\n')
                rules={x['rule'] for x in audit.audit()[0]['findings']}
                self.assertIn('reviewed_artifact_changed',rules)
                self.assertIn('excluded_file_type',rules)

    def test_unreviewed_csv_stays_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'unexpected.csv').write_text('patient_id,label\nexample,1\n')
            with patch.object(audit,'ROOT',root):
                self.assertEqual(audit.audit()[0]['status'],'blocked')
