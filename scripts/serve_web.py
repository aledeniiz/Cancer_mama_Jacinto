"""Serve the public aggregate explorer locally; no inference endpoints."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path


class PublicHandler(SimpleHTTPRequestHandler):
    def translate_path(self,path):
        candidate=Path(super().translate_path(path)).resolve()
        root=Path(self.directory).resolve()
        if not candidate.is_relative_to(root) or any(part.startswith('.') for part in candidate.relative_to(root).parts):
            return str(root/'__not_public__')
        parts=candidate.relative_to(root).parts
        if parts and parts[0] not in ('web','results','docs') and parts not in (('README.md',),('LICENSE',)):
            return str(root/'__not_public__')
        return str(candidate)

    def list_directory(self,path):
        self.send_error(404,'Directory listing is disabled')
        return None

    def end_headers(self):
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Referrer-Policy','no-referrer')
        super().end_headers()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8000)
    args=p.parse_args();root=Path(__file__).resolve().parents[1]
    if not (root/'web/index.html').is_file():p.exit(2,'Error: public web resources unavailable\n')
    print(f'Public aggregate explorer: http://127.0.0.1:{args.port}/web/')
    with ThreadingHTTPServer(('127.0.0.1',args.port),partial(PublicHandler,directory=str(root))) as server:
        try:server.serve_forever()
        except KeyboardInterrupt:pass
