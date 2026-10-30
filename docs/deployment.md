# GitHub Pages deployment

The public website is a controlled export of the finished Spanish platform.
It is not a redesign. Its layout, chapters, animations, CNN laboratory and
aggregate multimodal comparison remain available.

The workflow builds **only** the reviewed self-contained `web/index.html` and a
`.nojekyll` marker. It does not upload the whole repository as a website artifact.
No datasets, trained models, source Git storage or private patient records are
included. GitHub Pages serves static HTML; it does not run the Python inference
server. Private-resource controls remain unavailable with explicit notices.

This restriction concerns the **Pages artifact**, not the repository: preserved
model bundles, public teaching-case metadata, partitions and prediction references
are now available separately in `models/trained/` and `evaluation/`. The final
academic presentation is in `presentation/`. None is embedded into the website.

## Repository configuration

Repository: [aledeniiz/Cancer_mama_Jacinto](https://github.com/aledeniiz/Cancer_mama_Jacinto).
In **Settings → Pages → Build and deployment**, the source must be **GitHub Actions**.
The Pages workflow deploys pushes to `main` and can also be dispatched manually.
Its actions are pinned to reviewed commit SHAs, and the deployment job has only
the necessary Pages/OIDC write permissions.

The implementation follows the [official GitHub Pages custom-workflow
documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
The full local suite currently contains 19 tests: 16 core checks and 3 checks of
the exact static deployment artifact. No ML dependency installation is needed in
the Pages workflow itself.

## Verified publication

The website was published and verified on **30 October 2026**:
[https://aledeniiz.github.io/Cancer_mama_Jacinto/](https://aledeniiz.github.io/Cancer_mama_Jacinto/).
The [first deployment run](https://github.com/aledeniiz/Cancer_mama_Jacinto/actions/runs/41036915754)
completed successfully. An independent HTTPS fetch returned 200 and matched the
reviewed HTML byte-for-byte. Browser checks confirmed the multimodal comparison,
clinical AUC 0.6450, six disabled private-resource upload controls and no console
errors during that check.

HTML SHA-256: `4def168a02ba74e91ba79645baa408a7bc0ef215519be7a5467bb7e0e27f6e35`.
That hash identifies the first publication. The scientific-delivery update
retains the design and updates availability notices; its reviewed HTML SHA-256
is `2cc599c983d1431bea6feb789d224af8e6ec6b2f866c87958a2c556ab8e9e478`.
The same URL is linked in the repository's About section. Future deployments
must be checked through their Actions run and actual HTTPS response; the Actions
deployment output is the authoritative URL.

## Local artifact verification

```sh
python -B scripts/audit_release.py
python -B web/build_public.py --check
python -B -m unittest discover -s tests -p test_pages.py
python -B scripts/build_pages.py --output .pages-site
python -m http.server 8878 --bind 127.0.0.1 --directory .pages-site
```

Build output must be a new directory; no existing project directory is replaced.
The HTML is copied byte-for-byte, so its hash can be compared with the served
GitHub Pages HTML after deployment. No PyTorch installation, training run,
patient data or secret is needed to build the website.
