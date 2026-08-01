# Can the v1 transformer identify the GAM's shadow attraction `w`?

**No.** It places `w` at the BAM corner (`w ≈ 0`, i.e. `v2 ≡ 1`), slightly on the
inadmissible side of it, with a cell-level dispersion of ±0.03–0.05 whose sign
and ordering do not survive a change in how the outside option is specified.

Regenerating `v2.csv` from this probe would write **1.0000 into all 1440 cells** —
every OD clamps — which is strictly worse than the 61 distinct values the shipped
build has.

This is an *independent second* identification failure. §8 of the README shows the
ticketed-sales feed cannot identify `w` (the reconstructible offer-set axis is
rejected by the data: deviance 131,774 for dropping it vs 143,752 for the GAM at
its most favourable `w`). This note shows an individual-level choice model
trained on the same commercial data cannot either, for a structural reason that
has nothing to do with the sales feed.

Reproduce with `python probe.py` (~15 min, CPU). Raw output:
`recapture_measured.json`.

## 1. Scope — the checkpoint is a 4-city model

`api/model/vocabularies.json` carries four cities (DEL, LON, SIN, SYD) and four
points of sale (AU, GB, IN, SG); `api/new_data/Itinerary.csv`, the simulator's
live choice-set asset, is on the same network. Any other OD maps
origin/destination to `__UNKNOWN__`.

**7 of the 90 Air_NRM ODs are in-domain**: DEL–SIN, LHR–SIN, LGW–SIN, SIN–LHR,
SIN–LGW, SIN–SYD, SYD–SIN. (`SIN–DEL` is in the vocabulary but absent from
`Itinerary.csv`.) LHR and LGW share the city-level market LON, as `v1.csv`'s MIDT
share already does; they differ here through their own fares and departure times.

The checkpoint itself is healthy: 27.4M params, `d_model` 512 / 8 heads / 6
layers, GSP price head with `softplus(beta) = 2.09`. On the total-derivative test
all 16 SQ cells have **correct-sign** own-price elasticity (−0.33 to −1.13). The
pervasive wrong-sign own-price response recorded against the earlier v1
checkpoint is **not** present in this re-export.

## 2. Why a softmax choice model cannot express `w`

Three ways to close a cell, on SIN–LHR, against the analytic BAM bound
(pure MNL renormalisation, the most any `w = 0` model can recapture):

| closure operationalisation | mean recapture | verdict |
|---|---|---|
| mask `j`'s logit to `-inf`, other rows' features unchanged | 0.717 | matches the analytic bound (0.719) to 0.031 → `w = 0` **by construction** |
| drop the row, rebuild the full feature pipeline | 1.161 | above bound in 16/16 → `w < 0` |
| price `j` out (`×20`), set size unchanged | 1.093 | above bound in 16/16 → `w < 0` |

Masking is the giveaway. A softmax over a *fixed* logit vector **is** a BAM: it
renormalises the survivors in proportion, which is exactly `w = 0`. So the only
route by which the transformer can produce `w ≠ 0` is its set-dependence — and
its set-dependence has the wrong sign. `num_choices`, the within-set price
rank / percentile / vs-min / vs-mean / z-score columns and the per-PNR price
aggregates all move when a row is withdrawn, and they make the *surviving* SQ
cells more attractive. A shadow attraction is the opposite: demand that leaks
away when an option disappears.

The price-out variant is used for the main run because it leaves set size (and
therefore `num_choices`) untouched and is by far the most stable of the three.

## 3. What the probe measures

Choice sets are real, not synthetic:

* **SQ block** — the 16 Air_NRM cells at their own observed per-`(OD, window,
  product)` fares and their own real representative departure times.
* **OAL block** — up to 180 real non-SQ itineraries from `Itinerary.csv` at their
  own `OW_Amt` fares, stratified by carrier, with departure times / durations /
  stops joined from `Flight.csv` + `OW_all.csv`. This is the GAM's outside
  option: `v1.csv`'s `no_purchase` is `non_sq_pax / market_pax`, i.e. "bought a
  non-SQ carrier", which is what an OAL row is.
* **Customers** — 200 per OD, drawn from that OD's own observed lead-day
  distribution in the POS sales feed.

The scale-free quantity is the recapture **relative to each frame's own BAM
bound**,

```
s_j = 1 - r_j / r_BAM,j        (≈ w_j / v_j, and admissible only if s_j >= 0)
```

Raw `r_j` is *not* transferable: the transformer's constructed markets sit at an
SQ share of 0.81–0.99 while `v1`'s attractions imply 0.24–0.48, and a recapture
of 0.79 is "low" where the market allows 0.851 but impossible where it allows
0.46. Inverting raw `r` against `v1`'s `v`/`v_0` clamps all 112 cells to `w = 0`.

## 4. Results — all 112 measured cells

| OD | `s ≥ 0` | s min | s median | s max | anchor stability (Pearson / Spearman) |
|---|---|---|---|---|---|
| DEL–SIN | 12/16 | −0.0007 | +0.0005 | +0.0020 | +0.70 / +0.72 |
| LHR–SIN | 0/16 | −0.0756 | −0.0344 | −0.0146 | +0.94 / +0.93 |
| LGW–SIN | 0/16 | −0.0772 | −0.0362 | −0.0156 | +0.94 / +0.91 |
| SIN–LHR | 0/16 | −0.0631 | −0.0202 | −0.0058 | +0.82 / +0.61 |
| SIN–LGW | 0/16 | −0.0562 | −0.0174 | −0.0088 | +0.74 / +0.45 |
| SIN–SYD | 0/16 | −0.0890 | −0.0296 | −0.0080 | +0.86 / +0.70 |
| SYD–SIN | **16/16** | +0.0155 | +0.0448 | +0.0785 | **−0.24 / −0.22** |

Pooled: mean `s` **−0.0156**, median **−0.0188**, sd **0.0319**, admissible
**28/112**. Implied `v2 = 1 − s` spans **0.9215 … 1.0890** — i.e. 1 plus noise,
straddling the boundary.

Two stability facts pull in opposite directions:

* Across **independent customer samples** at a fixed anchor, the per-cell profile
  is reproducible to Pearson **0.999** / Spearman **0.991** (n = 120 each). The
  signal is not sampling noise.
* Across the **outside-option anchor** (OAL fares at face value vs ×0.30), it is
  not robust. SYD–SIN — the one OD with a clean admissible positive `w` — has
  Pearson **−0.235**: its cell ordering reverses. So the dispersion is
  substantially a function of an arbitrary modelling choice, and cannot be
  transferred to the other 83 ODs as a measured shape.

## 5. The one substantive number this does produce

The probe bounds `|w/v| ≤ 0.089` across every market it can see, centred on zero.
The shipped build's calibration implies `theta` of **median 0.219, max 0.564**
(`build_report.json`). Those disagree by an order of magnitude.

Either the Ja et al. (2001) 35% recapture anchor is too aggressive for these
particular markets, or the transformer understates leakage because its 16 SQ
cells are near-perfect substitutes for one another (same airline, same routing,
same nonstop leg, same departure bank within a window — the only thing separating
them is fare). Both readings are live; the probe cannot choose between them.

Worth stating in the paper either way: it is a *directional* check on the
assumption, from a source independent of the sales feed.

## 6. What is not recommended

Do not build `v2` on the measured shape. It fails the anchor-stability test in
§4, and a benchmark ground-truth file should not carry cell-level structure whose
ordering flips when an unobservable modelling choice changes.

## Files

| file | what |
|---|---|
| `choice_sets.py` | SQ + OAL choice-set construction, customer sampling, the in-vocab OD map |
| `probe.py` | the closure probe; `python probe.py` regenerates the JSON |
| `recapture_measured.json` | per-cell recapture + BAM bound, 7 ODs × 2 anchors |
