#!/usr/bin/env python3
"""Local rating page for the two per-photo ratings (sexual implication, sexual attractiveness).

Serves ROOT/index.html, ROOT/items.json and ROOT/img/<id>.jpg on 127.0.0.1 and appends every
answer to RATINGS, a file outside ROOT that is never served. Photos appear under opaque ids; the
id -> account map is kept elsewhere. Standard library only.

Usage: python3 suggest_server.py ROOT PORT RATINGS
"""
import json
import os
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.abspath(sys.argv[1])
PORT = int(sys.argv[2])
RATINGS = os.path.abspath(sys.argv[3])
SCALES = ("implication", "attractive")


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
        if self.path.startswith("/flagged"):  # photos answered "may be under 18"
            return self._json(sorted(r["id"] for r in effective().values() if r.get("reason") == "under18"))
        return super().do_GET()

    def do_POST(self):
        if self.path != "/rate":
            return self._json({"error": "unknown"}, 404)
        n = int(self.headers.get("Content-Length", 0))
        try:
            r = json.loads(self.rfile.read(n))
            assert isinstance(r.get("id"), str)
            if not r.get("undo") and not r.get("skip"):
                for s in SCALES:
                    assert isinstance(r.get(s), int) and not isinstance(r.get(s), bool) and 0 <= r[s] <= 10
        except Exception:
            return self._json({"error": "bad rating"}, 400)
        r["server_ts"] = time.time()
        with open(RATINGS, "a") as f:
            f.write(json.dumps(r) + "\n")
        return self._json({"ok": True, "rated": len(effective())})


if __name__ == "__main__":
    print("rating page on http://127.0.0.1:%d" % PORT, flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
