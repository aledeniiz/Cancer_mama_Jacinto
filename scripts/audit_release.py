"""Inspect every release file locally; never upload findings or initialize Git."""
from pathlib import Path
import argparse, hashlib, json, re

ROOT=Path(__file__).resolve().parents[1]
FORBIDDEN_EXT={'.pyc','.pyo','.pt','.pth','.ckpt','.npz','.npy','.pkl','.pickle','.joblib',
    '.csv','.tsv','.parquet','.dcm','.nii','.h5','.hdf5','.pptx','.pdf','.zip','.tar','.7z'}
FORBIDDEN_DIR={'.git','.venv','venv','__pycache__','.pytest_cache','node_modules','backups','private','datasets','data'}
PATTERNS={
    'windows_local_path':re.compile(r'(?<![A-Za-z0-9])[A-Za-z]:[\\/](?:Users|home|tmp|workspace)[\\/]',re.I),
    'unix_local_path':re.compile(r'/(?:home|Users|mnt|workspace)/[A-Za-z0-9_.-]+'),
    'aws_access_key':re.compile(r'AKIA[0-9A-Z]{16}'),
    'github_token':re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})'),
    'openai_key':re.compile(r'sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}'),
    'private_key':re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'medical_case_identifier':re.compile(r'\b(?:Duke_Breast_MRI_\d+|ISPY[12][_-]\d{3,})\b',re.I),
    'embedded_binary':re.compile(r'data:image/(?:png|jpeg|webp);base64,',re.I),
}

def audit():
    findings=[];items=[];groups={}
    allowlist_path=ROOT/'docs/scientific_artifact_allowlist.json'
    allowed={}
    if allowlist_path.exists():
        try:
            entries=json.loads(allowlist_path.read_text(encoding='utf-8'))['files']
            for entry in entries:
                rel=entry['path'];parts=Path(rel).parts
                if (not rel.startswith(('evaluation/','models/trained/','presentation/')) or
                    '..' in parts or Path(rel).is_absolute() or rel in allowed or
                    not re.fullmatch('[0-9a-f]{64}',entry['sha256']) or
                    not isinstance(entry['size'],int) or entry['size']<0):
                    raise ValueError('Invalid reviewed artifact entry')
                allowed[rel]=entry
        except (KeyError,ValueError,TypeError):
            findings.append({'path':'docs/scientific_artifact_allowlist.json','rule':'invalid_artifact_allowlist'})
    for p in sorted(ROOT.rglob('*')):
        relative=p.relative_to(ROOT)
        # Git object storage is not part of a release or a Pages artifact.
        # Keep inspecting every other working-tree file, even when gitignored.
        if '.git' in relative.parts:continue
        if p.is_symlink():findings.append({'path':relative.as_posix(),'rule':'symlink_not_reviewed'});continue
        if not p.is_file():continue
        rel=relative.as_posix()
        content=p.read_bytes();sha=hashlib.sha256(content).hexdigest()
        reviewed=rel in allowed and allowed[rel]['sha256']==sha and allowed[rel]['size']==len(content)
        if rel in allowed and not reviewed:
            findings.append({'path':rel,'rule':'reviewed_artifact_changed'})
        if any(x in FORBIDDEN_DIR or x.endswith('.egg-info') for x in relative.parts):
            findings.append({'path':rel,'rule':'excluded_directory'})
        allowed_type=reviewed and p.suffix.lower() in {'.csv','.npz','.pptx','.pdf'}
        if (p.suffix.lower() in FORBIDDEN_EXT and not allowed_type) or p.name=='.env' or p.name.startswith('.env.'):
            findings.append({'path':rel,'rule':'excluded_file_type'})
        if len(content)>2*1024*1024 and not reviewed:findings.append({'path':rel,'rule':'large_file_requires_review'})
        try:text=content.decode('utf-8-sig')
        except UnicodeError:
            if not allowed_type:findings.append({'path':rel,'rule':'binary_requires_review'})
            text=''
        for rule,pattern in PATTERNS.items():
            # Only exact reviewed public-case scientific files may contain IDs.
            # Credential/path checks still apply to all decoded text.
            if reviewed and rule=='medical_case_identifier':continue
            if pattern.search(text):findings.append({'path':rel,'rule':rule})
        items.append({'path':rel,'size':len(content),'sha256':sha})
        if content:groups.setdefault(sha,[]).append(rel)
    duplicates=[v for v in groups.values() if len(v)>1]
    present={x['path'] for x in items}
    for rel in sorted(set(allowed)-present):
        findings.append({'path':rel,'rule':'reviewed_artifact_missing'})
    return {'status':'passed' if not findings else 'blocked','files':len(items),
        'logical_bytes':sum(x['size'] for x in items),'findings':findings,'duplicate_groups':duplicates,
        'scope':'All working-tree files except Git internal storage, including untracked/ignored files; findings omit matched values.',
        'limits':'Pattern scan is not a proof of absence of all secrets or a redistribution authorization.'},items

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest',type=Path)
    args=parser.parse_args();report,items=audit()
    if args.write_manifest:
        dest=args.write_manifest.resolve()
        if not dest.is_relative_to(ROOT) or dest.exists():parser.error('Manifest must be a NEW file inside the checkout')
        if report['findings']:parser.error('Blocked findings must be resolved before generating a release manifest')
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('x',encoding='utf-8') as f:json.dump({'schema_version':1,
            'note':'Manifest excludes itself; all listed files use repository-relative paths.',
            'files':items},f,indent=2)
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['status']=='passed' else 1)

if __name__=='__main__':main()
