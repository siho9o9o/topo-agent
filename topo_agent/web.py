# SPDX-License-Identifier: AGPL-3.0-or-later
"""Local web GUI for trying the topo agent (stdlib only, no new dependencies).

Open a photo (upload or a sample wall), tap a hold, pick a method, and see the
found holds drawn over the photo.

    python -m topo_agent.web [--host 127.0.0.1] [--port 8000] [--log out/web.log]

Then open http://127.0.0.1:8000 in a browser.

Scene and find requests are appended to the log file as JSON lines
(time, event, scene/method, status, seconds, error). Pass --no-log to disable.

API keys come from the environment. A `.env` file (KEY=VALUE lines) in the
directory where the server starts is also read; real environment variables win.
`.env` is git-ignored, so keys never end up in a commit.
"""
import argparse
import base64
import binascii
import contextlib
import datetime
import io
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from PIL import Image

from topo_agent import eval as hold_eval
from topo_agent.baseline import bench as hold_bench

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, "static", "index.html")
METHODS = ("color", "sam_color", "meta", "gemini")
MAX_UPLOAD = 15 * 1024 * 1024  # bytes of request body

REASONS = {
    "sam_color": "MODEL_API_KEY is not set",
    "meta": "MODEL_API_KEY is not set",
    "gemini": "Gemini is not configured (GEMINI_API_KEY or MIRAI_GEMINI_VERTEX)",
}


def load_env_file(path):
    """KEY=VALUE pairs from a .env file (# comments, single/double quotes). Missing file → {}."""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except (FileNotFoundError, NotADirectoryError):
        return {}
    out = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key and key not in out:
            out[key] = val
    return out


def apply_env_file(path):
    """Set .env keys the real environment does not already have. Returns how many were set."""
    loaded = load_env_file(path)
    count = 0
    for key, val in loaded.items():
        if key not in os.environ:
            os.environ[key] = val
            count += 1
    return count


def method_status(env):
    """Method name → {"ok", "reason"} (what the GUI can run)."""
    with contextlib.redirect_stderr(io.StringIO()):  # finders() notes missing keys on stderr
        available = hold_eval.finders(env, list(METHODS))
    return {m: {"ok": m in available, "reason": "" if m in available else REASONS.get(m, "not available")}
            for m in METHODS}


def load_jpeg(data_url):
    """A data-URL photo → bytes (validated). Raises ValueError."""
    if not isinstance(data_url, str) or "," not in data_url:
        raise ValueError("photo must be a data URL")
    head, _, b64 = data_url.partition(",")
    if "image/" not in head:
        raise ValueError("photo must be a data URL")
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("photo is not valid base64")
    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.convert("RGB")
    except Exception:
        raise ValueError("photo is not a readable image")
    return raw


def scene_payload(name, seed):
    """A sample wall as JSON-able data (photo, size, tap, marked holds). Raises ValueError."""
    if name not in hold_bench.NAMES:
        raise ValueError(f"unknown scene: {name}")
    s = hold_bench.scene(name, seed)
    with Image.open(io.BytesIO(s["jpeg"])) as img:
        w, h = img.size
    return {"name": name, "seed": seed, "width": w, "height": h,
            "jpeg": "data:image/jpeg;base64," + base64.b64encode(s["jpeg"]).decode(),
            "tap": s["tap"], "truth": [{"points": t["points"]} for t in s["truth"]]}


def open_log(path):
    """Prepare the request log file (creating its directory). Empty or None disables logging."""
    if not path:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    return path


def log_event(server, event):
    """Append one JSON line to the server log file. Without a log path, do nothing."""
    path = getattr(server, "log_path", None)
    if not path:
        return
    record = {"time": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), **event}
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass  # logging must never break the server


def find_payload(jpeg, x, y, method, env):
    """Run one method. Finder failures become error text, never exceptions."""
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    with contextlib.redirect_stderr(io.StringIO()):
        finders = hold_eval.finders(env, [method])
    if method not in finders:
        return {"holds": [], "seconds": 0.0, "error": REASONS.get(method, "not available")}
    started = time.monotonic()
    try:
        holds = finders[method](jpeg, {"x": x, "y": y}) or []
        error = ""
    except Exception as err:  # one failing photo should not kill the server
        holds, error = [], f"{type(err).__name__}: {str(err)[:200]}"
    return {"holds": holds, "seconds": round(time.monotonic() - started, 1), "error": error}


class Handler(BaseHTTPRequestHandler):
    server_version = "TopoAgentWeb/1.0"

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False))

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            try:
                with open(INDEX, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._json(500, {"error": "index.html is missing"})
        elif url.path == "/api/status":
            self._json(200, {"methods": method_status(dict(os.environ)), "scenes": list(hold_bench.NAMES)})
        elif url.path == "/api/scene":
            q = parse_qs(url.query)
            name = q.get("name", [""])[0]
            started = time.monotonic()
            try:
                seed = int(q.get("seed", ["11"])[0])
            except ValueError:
                log_event(self.server, {"event": "scene", "name": name, "status": 400,
                                        "seconds": round(time.monotonic() - started, 3),
                                        "error": "seed must be an integer"})
                return self._json(400, {"error": "seed must be an integer"})
            try:
                payload = scene_payload(name, seed)
            except ValueError as err:
                log_event(self.server, {"event": "scene", "name": name, "seed": seed, "status": 400,
                                        "seconds": round(time.monotonic() - started, 3), "error": str(err)})
                return self._json(400, {"error": str(err)})
            log_event(self.server, {"event": "scene", "name": name, "seed": seed, "status": 200,
                                    "seconds": round(time.monotonic() - started, 3), "error": ""})
            self._json(200, payload)
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/find":
            return self._json(404, {"error": "not found"})
        started = time.monotonic()
        method = None

        def fail(code, text):
            log_event(self.server, {"event": "find", "method": method, "status": code,
                                    "seconds": round(time.monotonic() - started, 3), "error": text})
            return self._json(code, {"error": text})

        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            return fail(400, "bad Content-Length")
        if length > MAX_UPLOAD:
            return fail(413, "photo is too large (max 15 MB)")
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return fail(400, "body must be JSON")
        method = req.get("method", "")
        try:
            jpeg = load_jpeg(req.get("jpeg", ""))
        except ValueError as err:
            method = None  # the photo was never read, so the method never ran
            return fail(400, str(err))
        x, y = req.get("x"), req.get("y")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (x, y)):
            return fail(400, "tap x/y must be numbers")
        if not (0 <= x <= 1 and 0 <= y <= 1):
            return fail(400, "tap x/y must be within 0-1")
        try:
            out = find_payload(jpeg, x, y, method, dict(os.environ))
        except ValueError as err:
            return fail(400, str(err))
        log_event(self.server, {"event": "find", "method": method, "status": 200,
                                "holds": len(out["holds"]), "seconds": out["seconds"], "error": out["error"]})
        self._json(200, out)

    def log_message(self, *args):
        pass  # quiet; the console shows only the start line


def main(argv=None):
    ap = argparse.ArgumentParser(description="Local web GUI for the topo agent")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--log", default=os.path.join("out", "web.log"), help="request log file (JSON lines)")
    ap.add_argument("--no-log", action="store_true", help="do not write a request log")
    a = ap.parse_args(argv)
    dotenv = os.path.join(os.getcwd(), ".env")
    if os.path.isfile(dotenv):
        print(f"topo-agent web: loaded {apply_env_file(dotenv)} key(s) from .env (values never shown)")
    server = ThreadingHTTPServer((a.host, a.port), Handler)
    server.daemon_threads = True
    server.log_path = None if a.no_log else open_log(a.log)
    print(f"topo-agent web: open http://{a.host}:{server.server_port} in a browser (Ctrl-C to stop)")
    if server.log_path:
        print(f"topo-agent web: logging requests to {server.log_path}")
        log_event(server, {"event": "start", "host": a.host, "port": server.server_port})
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
