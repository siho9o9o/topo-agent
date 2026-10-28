# Experiment log

Tuning experiments on the topo agent: what was tried, what was measured, and what was decided.
Check this file before re-trying an optimization. One run per case unless noted; the agent is
stochastic (`temperature` 0.2), so treat small differences with care.

## 2026-10-27: Turn-budget optimization

Goal: cut `meta` wall time on a real gym photo (956x1054, 86 SAM candidates). A trace showed
27.4 s in 3 Muse Spark turns vs 3.8 s in SAM: turns are ~90% of the time.

Tried prompt pressure, then reverted it:

| Variant | Holds | Seconds | Turns | Outcome |
|---|---|---|---:|---|
| Base prompt | 11-12 | 43-45 | 3 | Main wall only, correct |
| v1: submit in the first turn | 6 | 14.5 | 1-2 | Dropped the top group |
| v2: include every linked=yes | 22 | 17.2 | 1-2 | Included the next wall |

v1 under-picks (answers before looking far), v2 over-picks (the `linked` chain already reaches
the next wall on this photo, so "trust linked" equals a color filter). The base prompt was
restored and quality re-verified (12 holds, 43 s, 4 turns, left wall excluded).

Kept (structural caps, no policy change):

- `MAX_TURNS` 6 → 4 (typical runs use ≤4; the 90 s tail is gone).
- `MAX_LOOKS` = 2 `look_closer` calls per run (each call appends an image to every later turn).
- Tests: default turn cap, per-run look cap (`tests/test_agent.py`).

## 2026-10-27: Image diet

Knobs (all optional, defaults unchanged): `META_AGENT_MODEL`, `META_AGENT_EFFORT`,
`META_AGENT_IMAGE_SIDE` (1280), `META_AGENT_JPEG_QUALITY` (85).

Smaller first-message images, bench `meta` on tape,twins,similar × seed 11:

| Variant | Payload | Recall | Precision | Seconds/photo |
|---|---|---|---|---|
| base 1280/85 | 100% | 100% | 100% | 8.2 |
| mid 960/70 | ~60% | 100% | 100% | 9.6 |
| low 640/55 | ~30% | 100% | 96% | 10.8 |

Rejected: less detail means more turns (tape: base 2 turns + 1 `look_closer` vs low 4 turns +
3 `look_closer`, the third refused by `MAX_LOOKS`), so the run gets slower, not faster, and
low answers wobble between runs (6 vs 8 holds on tape). Payload bytes were measured without
API calls; bench numbers above used the real APIs.

## 2026-10-28: Real-photo subset bench (meta)

First real-photo scores: 8 Kaggle route cases (largest route per image), `--photos` mode,
team code (6-turn cap):

| Case | Truth | Found | Hits | Seconds |
|---|---|---|---|---|
| bh0003_r1 | 3 | 3 | 1 | 23.6 |
| bh0074_r13 | 20 | 9 | 3 | 39.0 |
| bh0075_r1 | 12 | 10 | 10 | 32.9 |
| bh0457_r1 | 18 | 7 | 7 | 33.2 |
| bh0502_r4 | 15 | 8 | 7 | 36.7 |
| bh0518_r3 | 8 | 5 | 5 | 34.6 |
| bh0530_r3 | 16 | 7 | 7 | 29.3 |
| bh0917_r7 | 12 | 11 | 11 | 36.8 |
| TOTAL | 104 | 60 | 51 | mean 33.3, recall 49%, precision 85% |

Re-runs wobble (0457: 7 → 14 hits; 0917: 11 → 9). All 8 traces saved with the overlays;
analysis in [learnings.md](learnings.md).

## 2026-10-28: SAM phrase grounding

Color- and tape-qualified concept prompts on a real gym photo (assets/119720094.2.jpg):

| Prompt | Returned | Actually matching |
|---|---|---|
| `red climbing hold` | 16 | 15 red |
| `yellow climbing hold` | 33 | 29 yellow (+4 orange) |
| `tape` | 0 | — |
| `colored tape strip` | 13 | mixed tape + bolt-hole lookalikes |

Color adjectives ground well enough for hints; tape needs the longer phrase plus
color+adjacency filtering. Detail in [learnings.md](learnings.md).

## 2026-10-28: Three-stage pipeline (meta3) vs ReAct agent (meta)

Same 8 Kaggle route cases, one run each. `meta` numbers from the earlier run (team
6-turn code); `meta3` = SAM + CV color grouping + single-pass Spark (`feat/three-stage`).
Per-case traces, prompts, raw Spark replies and found holds saved under `out/three-stage/`.

| Case | meta truth/found/hits/s | meta3 truth/found/hits/s |
|---|---|---|
| bh0003_r1 | 3/3/1/23.6 | 3/2/1/12.5 |
| bh0074_r13 | 20/9/3/39.0 | 20/3/3/13.6 |
| bh0075_r1 | 12/10/10/32.9 | 12/10/10/21.3 |
| bh0457_r1 | 18/7/7/33.2 | 18/5/4/29.4 |
| bh0502_r4 | 15/8/7/36.7 | 15/5/2/21.1 |
| bh0518_r3 | 8/5/5/34.6 | 8/16/4/13.3 |
| bh0530_r3 | 16/7/7/29.3 | 16/4/3/18.1 |
| bh0917_r7 | 12/11/11/36.8 | 12/18/12/18.2 |
| TOTAL | recall 49%, precision 85%, mean 33.3 s | recall 38%, precision 62%, mean 18.4 s |

`meta3` stage means: SAM 2.8 s (3 calls), CV colors 0.4 s, Spark 15.2 s (1 call),
post ~0 s. Spark prompts 3.3k-11.5k tokens (110 candidates on 0074), completions
1.7k-5.3k tokens, mostly reasoning (1.6k-4.8k) despite `reasoning_effort: low`.
Zero fallbacks: all 8 replies were valid JSON holding the tapped route.
Analysis in [learnings.md](learnings.md).

## 2026-10-28: Synth bench with meta3 (9 scenes x seeds 11, 12)

Same run, team 6-turn code, one run per method. Traces and raw replies saved under
`out/three-stage-synth/`.

| Method | Recall | Precision | Mean s/photo |
|---|---|---|---|
| sam_color | 90% | 86% | 1.5 |
| meta | 99% | 100% | 6.5 |
| meta3 | 85% | 96% | 12.2 |

`meta3` misses: tape-11 (1/7), tape-12 (1/7), volume-11 (2/7), busy-11 (5/7); all
other scenes 7/7. Stage means: SAM 1.5 s, CV colors 0.4 s, Spark 10.3 s.
Analysis in [learnings.md](learnings.md).

## 2026-10-28: Tap-only meta3 re-run (equal criteria, same session)

v1 asked for ALL routes (unfair task vs the agent's tap-only question, temperature 0,
no tap close-up). v2 fixes all three and re-runs both methods in one session.
Outputs under `out/three-stage-v2/` (real) and `out/three-stage-synth-v2/`.

Synth (18 problems):

| Method | Recall | Precision | Mean s/photo |
|---|---|---|---|
| sam_color | 90% | 86% | 1.6 |
| meta | 99% | 100% | 6.8 |
| meta3 | 99% | 96% | 5.1 |

`meta3` stages: SAM 1.5 s, CV 0.4 s, Spark 3.2 s (was 10.3 s). Tape 7/7 both.
Only miss vs meta: twins-12 over-pick (12 found, 7 hits).

Real (8 cases, same session):

| Case | meta truth/found/hits/s | meta3 truth/found/hits/s |
|---|---|---|
| bh0003_r1 | 3/4/2/25.4 | 3/3/1/8.7 |
| bh0074_r13 | 20/9/3/17.8 | 20/3/3/5.6 |
| bh0075_r1 | 12/10/10/25.2 | 12/5/5/7.3 |
| bh0457_r1 | 18/14/14/23.3 | 18/16/5/14.9 |
| bh0502_r4 | 15/8/7/31.9 | 15/20/5/14.9 |
| bh0518_r3 | 8/9/8/40.6 | 8/33/5/11.3 |
| bh0530_r3 | 16/7/7/28.6 | 16/9/9/12.6 |
| bh0917_r7 | 12/12/11/36.4 | 12/12/9/13.6 |
| TOTAL | recall 60%, precision 85%, mean 28.6 s | recall 40%, precision 42%, mean 11.1 s |

`meta3` stages: SAM 2.9 s, CV 0.4 s, Spark 7.8 s (was 15.2 s), zero fallbacks.
Note the baseline wobble: same code scored 49%/85%/33.3 s last session.
