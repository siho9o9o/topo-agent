# SPDX-License-Identifier: AGPL-3.0-or-later
"""Test the hold-finding agent (topo_agent/baseline/gemini_agent.py) with a fake model (never connects to Gemini)."""
import unittest

from PIL import Image, ImageDraw

from topo_agent.baseline import gemini_agent as hold_agent


def wall():
    img = Image.new("RGB", (800, 600), (90, 200, 200))  # teal wall
    d = ImageDraw.Draw(img)
    d.ellipse((200, 200, 260, 250), fill=(150, 60, 170))  # purple (tapped)
    d.ellipse((330, 260, 360, 285), fill=(150, 60, 170))  # purple (same problem)
    d.ellipse((400, 200, 430, 225), fill=(40, 70, 200))  # blue (another problem)
    return img


def scripted(*replies):
    """A fake model answering in a fixed order. Also keeps the received history."""
    seen = []

    def call(history):
        seen.append(list(history))
        calls = replies[min(len(seen) - 1, len(replies) - 1)]
        return {"calls": calls, "text": "", "raw": {"calls": calls}}
    return call, seen


class HoldAgentTest(unittest.TestCase):
    def test_agent_uses_tools_then_submits(self):
        tap = {"x": 0.2875, "y": 0.375}
        model, seen = scripted(
            [{"name": "inspect_hold", "args": {"x": 287, "y": 375}}],
            [{"name": "find_same_color", "args": {"x": 287, "y": 375}}],
            [{"name": "submit_holds", "args": {"holds": [{"x": 287, "y": 375, "size": 40}, {"x": 431, "y": 455},
                                                         {"x": 2000, "y": 10}], "note": "two purple holds"}}])
        holds, trace = hold_agent.run_agent(model, wall(), tap)
        self.assertEqual([t["tool"] for t in trace], ["inspect_hold", "find_same_color", "submit_holds"])
        self.assertEqual(len(holds), 2)  # coordinates outside the photo are dropped
        self.assertAlmostEqual(holds[1]["x"], 0.431)
        # tool results are passed to the next model call
        inspected = seen[1][-1]["tool_results"][0]["response"]
        self.assertEqual(inspected["color"], "purple")
        self.assertTrue(inspected["outline_found"])
        found = seen[2][-1]["tool_results"][0]["response"]["candidates"]
        self.assertEqual(len(found), 1)  # only same-color purple (no blue)
        self.assertAlmostEqual(found[0]["x"], 431, delta=15)
        self.assertTrue(all("ms" in t for t in trace))  # record the time taken

    def test_step_limit_stops_the_agent(self):
        # a model that keeps using tools stops at the limit and answers with what is known
        model, seen = scripted([{"name": "find_same_color", "args": {"x": 287, "y": 375}}])
        holds, trace = hold_agent.run_agent(model, wall(), {"x": 0.2875, "y": 0.375}, steps=3)
        ran = [t for t in trace if t["tool"] == "find_same_color" and "limit" not in t["result"]]
        self.assertEqual(len(ran), 3)  # past the limit, tools are refused without running
        self.assertEqual(trace[-1]["tool"], "(stopped at limit)")
        self.assertEqual(len(holds), 2)  # the tapped spot + 1 color-found hold
        self.assertLessEqual(len(seen), 5)

    def test_unknown_tool_and_no_answer(self):
        model, _ = scripted([{"name": "delete_everything", "args": {}}], [])
        holds, trace = hold_agent.run_agent(model, wall(), {"x": 0.2875, "y": 0.375})
        self.assertIn("Unknown tool", trace[0]["result"])
        self.assertEqual(trace[1]["tool"], "(no answer)")

    def test_clean_answer_limits(self):
        many = [{"x": 10, "y": 10}] * (hold_agent.MAX_HOLDS + 5) + [{"x": "a", "y": 1}, "x"]
        self.assertEqual(len(hold_agent.clean_answer(many)), hold_agent.MAX_HOLDS)
        self.assertEqual(hold_agent.clean_answer(None), [])

    def test_color_only_when_gemini_fails(self):
        # fallback when Gemini is silent: the tapped hold and same-color holds only (no blue)
        holds = hold_agent.color_only(wall(), {"x": 230 / 800, "y": 225 / 600})
        self.assertEqual(len(holds), 2)
        self.assertTrue(all(h["x"] < 400 / 800 for h in holds))
        self.assertEqual(hold_agent.color_only(wall(), None), [])

    def test_off_without_settings(self):
        self.assertIsNone(hold_agent.gemini_hold_finder({}))
        self.assertIsNone(hold_agent.gemini_page_checker({}))

    def test_check_finds_a_model_that_works(self):
        class Models:
            def generate_content(self, model, contents):
                if model != "gemini-2.5-flash":
                    raise RuntimeError("404 model not found")

        fake = type("Client", (), {"models": Models()})()
        real = hold_agent.gemini_client
        hold_agent.gemini_client = lambda env: fake
        try:
            found, tried = hold_agent.check({})
            self.assertEqual(found, "gemini-2.5-flash")
            self.assertEqual(len(tried), 3)  # unusable models are also listed with reasons
            self.assertEqual(hold_agent.check({"MIRAI_GEMINI_MODEL": "gemini-x"})[0], None)  # only the set model is tried
        finally:
            hold_agent.gemini_client = real
        self.assertEqual(hold_agent.check({})[0], None)  # unusable without settings

    def test_color_names(self):
        self.assertEqual(hold_agent.color_name((0.75, 0.6, 0.6)), "purple")
        self.assertEqual(hold_agent.color_name((0.0, 0.05, 0.95)), "white")
        self.assertEqual(hold_agent.color_name((0.0, 0.05, 0.1)), "black")


if __name__ == "__main__":
    unittest.main()
