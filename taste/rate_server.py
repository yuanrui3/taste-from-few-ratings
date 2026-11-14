#!/usr/bin/env python3
"""Local rating page for the appearance taste model.

Serves ROOT/index.html, ROOT/items.json and ROOT/thumbs/<id>/<k>.jpg on 127.0.0.1 and appends
every rating to ROOT/ratings.jsonl. Accounts appear under opaque ids; the id -> account map is
kept elsewhere and never served. Standard library only.

Usage: python3 rate_server.py ROOT [PORT]
"""
import json
import os
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.abspath(sys.argv[1])
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
RATINGS = os.path.join(ROOT, "ratings.jsonl")


def effective():
    """Last entry per (id, pass), after undos."""
    out = {}
    if os.path.exists(RATINGS):
        for line in open(RATINGS):
            r = json.loads(line)
            key = "%s#%s" % (r["id"], r.get("pass", 1))
            if r.get("undo"):
                out.pop(key, None)
            else:
                out[key] = r
    return out


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/done"):
            return self._json(sorted(effective()))
        if self.path.startswith("/ratings.jsonl"):
            return self._json({"error": "not served"}, 403)
        return super().do_GET()

    def do_POST(self):
        if self.path != "/rate":
            return self._json({"error": "unknown"}, 404)
        n = int(self.headers.get("Content-Length", 0))
        try:
            r = json.loads(self.rfile.read(n))
            assert isinstance(r.get("id"), str)
            if not r.get("undo") and not r.get("skip"):
                assert isinstance(r.get("rating"), int) and 0 <= r["rating"] <= 10
        except Exception:
            return self._json({"error": "bad rating"}, 400)
        r["server_ts"] = time.time()
        with open(RATINGS, "a") as f:
            f.write(json.dumps(r) + "\n")
        return self._json({"ok": True, "rated": len(effective())})


if __name__ == "__main__":
    print("rating page on http://127.0.0.1:%d" % PORT, flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
