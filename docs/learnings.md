# Learnings

Durable knowledge from experiments. Numbers live in [experiments.md](experiments.md); this file keeps
what we learned and what it implies. Each entry names its evidence.

## Turns dominate latency

Model turns are ~90% of wall time (31 s run: 27.4 s Spark vs 3.8 s SAM; local tools 0.02 s).
Later turns get slower as images accumulate (7.3 s → 12.6 s across 3 turns).
Implication: cut turns first; micro-optimizing anything else is noise.

## Prompt pressure swings quality both ways

"Submit in the first turn" under-picked (11 → 6 holds, top group dropped).
"Include every linked=yes" over-picked (11 → 22, next wall included).
Implication: prefer structural guardrails (`MAX_TURNS`, `MAX_LOOKS`) over prompt pressure;
prompt wording moves answers unpredictably in both directions.

## Less image detail means more turns

Halving image bytes (1280/85 → 640/55) slowed the bench (8.2 s → 10.8 s): the model
compensated with more turns (2 → 4) and more zooms (1 → 3) on tape-11, and answers wobbled
(6 vs 8 holds across runs). Implication: images are decision fuel, not overhead; do not diet them.

## The agent distrusts the candidate table

Traces show every run opening with 6–12 `look_closer` crops and 13–20-id `inspect` batches
to re-verify numbers the first message already carried, then submitting at the turn cap.
Confirmed across all 8 subset traces: 7 open with `look_closer`, 4 submit on turn 6.
Implication: exploration is the cost center. Precompute groupings deterministically and let the
model verify instead of explore (the fuse-then-verify redesign).

## The agent under-picks on dense walls

Real-photo bench: recall 49%, precision 85%, found ≤ truth in all 8 cases (worst 3/20 on the
densest wall). The 0530 trace says it outright: "excluded distant top/volume holds".
Misses are decisions, not blindness.
Implication: recall work means changing the decision policy (completeness pressure,
linked-chain handling), not better perception.

## Measured color can disagree with seen color

On 0917 the tap measured "gray" while the model saw light blue; it routed around the
mismatch with extra zoom turns. Implication: color words from measurements are hints,
and color-qualified SAM queries need a measured-distance fallback.

## SAM grounds color and tape phrases (with caveats)

- `red climbing hold`: 15/16 actually red. `yellow climbing hold`: 29/33 (+4 orange).
- `tape`: 0 results. `colored tape strip`: 13 boxes, mixed real tape and bolt-hole lookalikes.
Implication: color phrases are usable as hints; tape phrases need color+adjacency filtering.

## Turn caps bind on dense images

Two of three traced real-photo runs used all 6 turns. A 4-turn cap cuts the tail but can
truncate legitimate reasoning on dense walls. Implication: report cap interactions when
tuning limits; the caps PR trades tail latency for completeness risk.

## Synth-to-real gap is large

Same code: 99%/100%/6.6 s on drawn walls vs 49%/85%/33 s on real photos.
Implication: synth bench guards regressions; only real photos measure the product.

## Kaggle route labels are per-image

`route_id` restarts per photo (same id, different colors across images); 8 of 2273 photos
carry routes (68 photo-route cases). Some frames are misaligned (bh-phone/182) or empty
floor shots. License CC-BY-SA-4.0: eval use with attribution, no redistribution.
Implication: curate before scoring; same-color routes only, so tape/cross-wall cases still
need hand-annotated gym photos.

## Single-pass grouping is faster but blind without zoom

`meta3` (one Spark call over precomputed colors) cut real-photo latency 33.3 s to
18.4 s with zero fallbacks, but recall fell 49% to 38% and precision 85% to 62%.
Real-light color names fragment routes (a tap can be the only `white` hold on its
wall; dark greens split into `green`/`gray`), Spark answers cautiously (23
unassigned on 0530), and there is no `look_closer` to recover.
Implication: keep a verification step for uncertain holds; ask for the tapped route
only instead of all routes to cut reasoning (1.6k-4.8k tokens per call now).

## All-routes output keeps single-pass slow

One `meta3` call still averages 15.2 s because grouping the whole wall (up to 110
candidates, 11.5k prompt tokens) forces global reasoning even at `effort: low`.
Implication: latency scales with task scope, not just turn count; narrow the question.

## Union-find color groups chain across distinct colors

Pairwise-threshold grouping merged 54 holds on 0530 into 2 groups (orange/yellow/red
chained through intermediate shades). Spark split them back into routes, but the CV
fallback (tap's group) would be near-useless there.
Implication: group by distance to the tap (star) or cap chain length for fallback use.
