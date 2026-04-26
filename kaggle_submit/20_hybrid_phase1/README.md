# `20_hybrid_phase1/` — Phase-1 hybrid sweep

Each `exp_*.ipynb` is the same 0.924-baseline + 5-fold OV ensemble notebook
with **exactly three** differences from `exp_A1` baked into cell 52:

```python
HYBRID_W      # scalar weight on our predictions
PER_TAXON_W   # None or {Aves: ..., Insecta: ..., ...}
BLEND_MODE    # 'rank_arith' | 'rank_geo' | 'prob_arith'
```

## The 7 experiments

| Tag | HYBRID_W | PER_TAXON_W | BLEND_MODE | Hypothesis |
|---|---|---|---|---|
| **A0** | **0.20** | – | rank_arith | Conservative — should be very close to baseline. Tests if W=0.30 was already too aggressive. |
| **A1** | **0.30** | – | rank_arith | Current best (PB 0.925, +0.001 over baseline). |
| **A2** | **0.40** | – | rank_arith | Bumping our weight; if PB rises, our predictions are stronger than I assumed. |
| **A3** | **0.50** | – | rank_arith | Balanced 50/50; tests upper end of the W curve. |
| **B**  | 0.30 | – | **rank_geo** | Geometric blend of column ranks; better when one model is confidently wrong. |
| **C**  | 0.30* | **{Aves 0.20, Amphibia 0.40, Insecta 0.45, Mammalia 0.30, Reptilia 0.25}** | rank_arith | Up-weight us on non-Aves taxa where Perch is relatively weakest. |
| **D**  | 0.40 | **{Aves 0.30, Amphibia 0.45, Insecta 0.50, Mammalia 0.35, Reptilia 0.30}** | **rank_geo** | Combo of best knobs from A/B/C. |

\* HYBRID_W in C/D is a fallback if a class taxon is missing from the dict;
in practice all 234 classes belong to one of the 5 listed taxa, so this
fallback never fires.

## Recommended order (high information-density first)

1. **A2** (W=0.40) — does pushing our weight up help or hurt? Tells us
   whether to move along the A axis or the B/C axes.
2. **B** (rank_geo) — does the geometric blend dominate arithmetic at
   the same W? If yes, switch all subsequent runs to geo.
3. **C** (per-taxon) — does per-taxon weighting beat scalar at W=0.30?
4. **A0** (W=0.20) — only worth it if A2 shows W>0.30 hurts us
   (suggests the optimum is below 0.30).
5. **A3** (W=0.50) — only worth it if A2 shows W=0.40 helps.
6. **D** (combo) — last; combines whatever individual knobs won.

This ordering means after the first 3 submissions you know which axis
matters most and can skip the dead branches.

## What to expect

The blend space is small (1 weight + 1 mode + per-taxon dict). The
ceiling under Phase-1 alone is realistically **0.93 ± 0.005**. Anything
above 0.935 would be a surprise.

If Phase-1 caps below 0.93, the next dial is **Plan Z**: train a second
backbone (`tf_efficientnetv2_s_in21k`) on the same data, add it to the
OV ensemble. Folder `30_hybrid_phase2/` (TBD) will be where those
notebooks land.
