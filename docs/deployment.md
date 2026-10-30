# GitHub Pages deployment

The public website is a controlled export of the finished Spanish platform.
It is not a redesign. Its layout, chapters, animations, CNN laboratory and
aggregate multimodal comparison remain available.

The workflow builds **only** the reviewed self-contained `web/index.html` and a
`.nojekyll` marker. It does not upload the whole repository as a website artifact.
No datasets, trained models, source Git storage or private patient records are
included. GitHub Pages serves static HTML; it does not run the Python inference
server. Private-resource controls remain unavailable with explicit notices.

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

After a successful deployment the expected website address is
`https://aledeniiz.github.io/Cancer_mama_Jacinto/`.
This address is not evidence that deployment has completed: check the successful
Actions run and the actual HTTPS response. The Actions deployment output is the
authoritative URL.

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
