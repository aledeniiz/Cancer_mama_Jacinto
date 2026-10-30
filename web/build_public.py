"""Rebuild the sanitized original web using only the public files in this directory."""
from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parent


def read(name):
    return (ROOT / name).read_text(encoding="utf-8-sig")


def public_json(name):
    # Validate JSON before embedding and prevent closing-script sequences.
    value = json.loads(read(name))
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def render():
    page = read("shell.html")
    page = page.replace("</style></head>", "</style><style>" + read("cinematic.css") + "</style></head>")
    for name in ["premium-charts.css", "premium-lab.css", "multimodal.css", "dual-model.css"]:
        page = page.replace("</head>", "<style>" + read(name) + "</style></head>")
    page = page.replace("<!--LAB-->", read("lab.html"))
    page = page.replace('<section id="interpretation">', read("live.html") + '<section id="interpretation">')
    page = page.replace("__DATA__", public_json("project-data.json"))
    page = page.replace("__APP__", read("app.js"))
    page = page.replace("__LABJS__", read("lab.js"))
    page = page.replace('<section id="interpretation">', read("multimodal.html") + '<section id="interpretation">')
    page = page.replace("</body>", '<script id="multimodal-public-data" type="application/json">' + public_json("multimodal-public.json") + "</script></body>")
    for name in ["browser-inference.js", "cinematic.js", "live.js", "premium-ui.js", "multimodal.js", "dual-model.js"]:
        page = page.replace("</body>", "<script>" + read(name) + "</script></body>")
    return page


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check that index.html matches the public source files, without writing")
    args = parser.parse_args()
    output = render()
    destination = ROOT / "index.html"
    if args.check:
        if not destination.is_file() or destination.read_text(encoding="utf-8") != output:
            parser.exit(1, "Public index is absent or stale. Run python web/build_public.py.\n")
        print("Public index matches its sanitized source files.")
    else:
        destination.write_text(output, encoding="utf-8", newline="\n")
        print("Built public index from sanitized local sources only.")
