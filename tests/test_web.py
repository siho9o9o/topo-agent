# SPDX-License-Identifier: AGPL-3.0-or-later
"""Local web GUI (topo_agent/web.py), with a live server on an ephemeral port (no outside network)."""
import base64
import io
import json
import os
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest import mock
from urllib import error, request

from topo_agent import web
from topo_agent.baseline import bench as hold_bench


def sample(name="tape", seed=11):
    s = hold_bench.scene(name, seed)
    return "data:image/jpeg;base64," + base64.b64encode(s["jpeg"]).decode(), s["tap"], s["jpeg"]


class WebLogicTest(unittest.TestCase):
    def test_method_status_without_keys(self):
        st = web.method_status({})
        self.assertTrue(st["color"]["ok"])
        self.assertFalse(st["meta"]["ok"])
        self.assertIn("MODEL_API_KEY", st["meta"]["reason"])
        self.assertIn("MODEL_API_KEY", st["sam_color"]["reason"])
        self.assertIn("GEMINI", st["gemini"]["reason"])

    def test_method_status_with_dummy_key(self):
        st = web.method_status({"MODEL_API_KEY": "x"})  # availability only; no network call
        self.assertTrue(st["sam_color"]["ok"])
        self.assertTrue(st["meta"]["ok"])
        self.assertFalse(st["gemini"]["ok"])

    def test_env_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, ".env")
            with open(p, "w", encoding="utf-8") as f:
                f.write('# comment\n\nMODEL_API_KEY="abc"\nEMPTY=\nPLAIN=x y\nBADLINE\n')
            self.assertEqual(web.load_env_file(p), {"MODEL_API_KEY": "abc", "EMPTY": "", "PLAIN": "x y"})
            self.assertEqual(web.load_env_file(os.path.join(d, "missing")), {})
            with mock.patch.dict(os.environ, {"MODEL_API_KEY": "real"}, clear=True):
                self.assertEqual(web.apply_env_file(p), 2)  # real env wins, the rest are set
                self.assertEqual(os.environ["MODEL_API_KEY"], "real")
                self.assertEqual(os.environ["PLAIN"], "x y")

    def test_load_jpeg(self):
        jpeg, _, raw = sample()
        self.assertEqual(web.load_jpeg(jpeg), raw)
        for bad in ("nope", "data:text/plain;base64,aGk=", "data:image/jpeg;base64,!!!",
                    "data:image/jpeg;base64,aGVsbG8="):
            with self.assertRaises(ValueError, msg=bad):
                web.load_jpeg(bad)

    def test_scene_payload(self):
        s = web.scene_payload("tape", 11)
        self.assertEqual((s["width"], s["height"]), (hold_bench.WIDTH, hold_bench.HEIGHT))
        self.assertTrue(s["jpeg"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(len(s["truth"]), hold_bench.PER_PROBLEM)
        self.assertTrue(all(len(t["points"]) >= 3 for t in s["truth"]))
        self.assertTrue(0 <= s["tap"]["x"] <= 1 and 0 <= s["tap"]["y"] <= 1)
        with self.assertRaises(ValueError):
            web.scene_payload("nope", 11)

    def test_find_color_on_scene(self):
        _, tap, raw = sample()
        out = web.find_payload(raw, tap["x"], tap["y"], "color", {})
        self.assertEqual(out["error"], "")
        self.assertGreaterEqual(len(out["holds"]), 1)
        self.assertGreaterEqual(out["seconds"], 0)

    def test_find_unavailable_and_unknown(self):
        _, tap, raw = sample()
        out = web.find_payload(raw, tap["x"], tap["y"], "meta", {})
        self.assertEqual(out["holds"], [])
        self.assertIn("MODEL_API_KEY", out["error"])
        with self.assertRaises(ValueError):
            web.find_payload(raw, tap["x"], tap["y"], "nope", {})


class WebServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
        cls.server.daemon_threads = True
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def get(self, path):
        with request.urlopen(self.base + path) as r:
            return r.status, r.read()

    def post(self, obj):
        req = request.Request(self.base + "/api/find", data=json.dumps(obj).encode(),
                              headers={"Content-Type": "application/json"})
        try:
            with request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_root_serves_page(self):
        code, body = self.get("/")
        self.assertEqual(code, 200)
        self.assertIn(b"<title>topo-agent</title>", body)

    def test_status(self):
        code, body = self.get("/api/status")
        self.assertEqual(code, 200)
        st = json.loads(body)
        self.assertEqual(set(st["methods"]), {"color", "sam_color", "meta", "gemini"})
        self.assertEqual(st["scenes"], list(hold_bench.NAMES))

    def test_scene(self):
        code, body = self.get("/api/scene?name=tape&seed=11")
        self.assertEqual(code, 200)
        self.assertEqual(len(json.loads(body)["truth"]), hold_bench.PER_PROBLEM)
        with self.assertRaises(error.HTTPError) as ctx:
            self.get("/api/scene?name=nope")
        self.assertEqual(ctx.exception.code, 400)

    def test_find(self):
        jpeg, tap, _ = sample()
        code, out = self.post({"jpeg": jpeg, "x": tap["x"], "y": tap["y"], "method": "color"})
        self.assertEqual(code, 200)
        self.assertEqual(out["error"], "")
        self.assertGreaterEqual(len(out["holds"]), 1)
        code, out = self.post({"jpeg": jpeg, "x": tap["x"], "y": tap["y"], "method": "nope"})
        self.assertEqual(code, 400)
        code, out = self.post({"jpeg": jpeg, "method": "color"})
        self.assertEqual(code, 400)

    def test_find_missing_page(self):
        with self.assertRaises(error.HTTPError) as ctx:
            self.get("/missing")
        self.assertEqual(ctx.exception.code, 404)


class WebLogTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log = os.path.join(self.tmp.name, "logs", "web.log")  # nested: the server creates it
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
        self.server.daemon_threads = True
        self.server.log_path = web.open_log(self.log)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.server_close)

    def read_log(self):
        with open(self.log, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def post(self, obj):
        req = request.Request(self.base + "/api/find", data=json.dumps(obj).encode(),
                              headers={"Content-Type": "application/json"})
        try:
            with request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_scene_and_find_are_logged(self):
        with request.urlopen(self.base + "/api/scene?name=tape&seed=11") as r:
            self.assertEqual(r.status, 200)
        jpeg, tap, _ = sample()
        code, out = self.post({"jpeg": jpeg, "x": tap["x"], "y": tap["y"], "method": "color"})
        self.assertEqual(code, 200)
        lines = self.read_log()
        self.assertEqual([e["event"] for e in lines], ["scene", "find"])
        self.assertTrue(all(e["time"] and e["status"] == 200 and e["error"] == "" for e in lines))
        self.assertEqual((lines[0]["name"], lines[0]["seed"]), ("tape", 11))
        self.assertGreaterEqual(lines[0]["seconds"], 0)
        self.assertEqual(lines[1]["method"], "color")
        self.assertEqual(lines[1]["holds"], len(out["holds"]))
        self.assertGreaterEqual(lines[1]["holds"], 1)

    def test_failures_are_logged(self):
        try:
            request.urlopen(self.base + "/api/scene?name=nope")
        except error.HTTPError as e:
            self.assertEqual(e.code, 400)
        jpeg, tap, _ = sample()
        code, _ = self.post({"jpeg": jpeg, "x": tap["x"], "y": tap["y"], "method": "nope"})
        self.assertEqual(code, 400)
        code, _ = self.post({"jpeg": "nope", "method": "color"})
        self.assertEqual(code, 400)
        lines = self.read_log()
        self.assertEqual(len(lines), 3)
        self.assertTrue(all(e["status"] == 400 and e["error"] for e in lines))
        self.assertEqual(lines[0]["event"], "scene")
        self.assertEqual([e["method"] for e in lines[1:]], ["nope", None])

    def test_no_log_file_without_path(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)  # no log_path set
        try:
            web.log_event(server, {"event": "scene"})  # must not fail
        finally:
            server.server_close()
        self.assertIsNone(web.open_log(""))
        self.assertIsNone(web.open_log(None))
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "created")))


if __name__ == "__main__":
    unittest.main()
