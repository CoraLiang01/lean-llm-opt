# Air-NRM — large-scale case: SQ direct 2-city markets

Real-data build of the `Air_NRM/` reference files (`flight.csv`, `od_demand.csv`,
`v1.csv`, `v2.csv`), plus `flight_capacity.csv`, at the widest scope the SIA raw
data supports. This is the **large-scale** sibling of
[`../small_scale/`](../small_scale), which holds the 3-airport toy instance the
notebooks currently run. Same schema, same source (Singapore Airlines), two
different scales — see [`../README.md`](../README.md).

## What is runnable here, and what is not

**The CSVs in this folder are the deliverable.** The experiment pipeline reads
them directly and never runs any of the scripts below. The scripts are here so
the numbers are traceable — they document how the data provider derived each
file, which is what the paper's data section needs to cite.

| Script | Runnable in this repo? |
|---|---|
| `build_queries.py` | **yes** — reads only `flight.csv`, `flight_capacity.csv`, `v2.csv` from this folder; rewrites `query_largescale_CA.csv` |
| `check_query_pairs.py` | **yes** — same, a consistency check over `query_largescale_CA.csv` vs `flight.csv` |
| `build_air_nrm_inputs.py` | **no** — provenance only |
| `w_transformer/probe.py` | **no** — provenance only |

The two marked "no" reach outside this repository, into the booking-simulator
code base:

- `build_air_nrm_inputs.py` reads the raw SIA feed at
  `app/backend/api/Raw_data/` (the 7 per-POS sales CSVs,
  `Networkplanning_raw.xlsx`, the MIDT extract). That path is resolved as
  `../../app/backend/...` relative to this script, which assumes the script
  still lives inside the simulator repo — it does not resolve here, and never
  did in this repo.
- `w_transformer/probe.py` additionally needs the trained checkpoint
  `api/model/transformer_model_v1.pt` and imports the simulator's own
  `feature_engineer` / `model_loader` modules.

Both fail with a `FileNotFoundError` / `ImportError` rather than producing
anything wrong, so nothing silently breaks — but do not expect to reproduce
`flight.csv` or the transformer verdict from this repository alone. Nothing
under `app/backend/` is ever written; the simulator code base is input only.

> **Airport and city codes are censored.** Every output carries 3-digit
> surrogates (`('473', '218')`) instead of IATA codes. The key is
> `Supplement/airport_code_map.csv` — the one file to withhold when sharing. See
> [Censoring](#censoring) for what this does and does not protect.

## Answers to the three review comments

**1. "Is `v2.csv` the attraction value? `v[product×window] = sq_pax/market_pax`,
`no_purchase = non_sq_pax/market_pax` — that should be `v1.csv`."**

Correct, and fixed. That matrix is now **`v1.csv`**, byte-for-byte unchanged —
the rename is a pure rename, and `flight.csv`, `od_demand.csv` and the
cell-status file are identical to the previous build. `v2.csv` is now a
genuinely different quantity; see [§7](#7-shadow-attraction--v2csv).

Reading the reference files back confirms the contract `v2.csv` has to satisfy.
`query_CA.csv` describes them as "attraction values in v1 and **shadow
attraction value ratios** in v2", and in the generated `SBLP_CA_Label/*.lp` the
`v2` numbers appear verbatim as the balance-constraint coefficients. Matching
that against the paper's eq. (7) fixes the units:

```
v2 cell         = vtilde_j / v_j = 1 - w_j/v_j        in [0, 1]
v2 no_purchase  = vtilde_0 / v_0 = 1 + sum_j w_j / v_0
```

using `vtilde_j = v_j - w_j` and `vtilde_0 = v_0 + W(N)` (paper p. 215).
Reconstructing the reference `v2.csv`'s `no_purchase` from its own v1/v2 by that
formula reproduces its printed value on 3 of its 4 rows to 2 dp (`('A','B')`
1.4071 vs 1.41, `('A','C')` 1.5497 vs 1.55, `('C','A')` 1.3996 vs 1.40). The
fourth, `('B','A')`, gives 1.97 against a printed 1.53 and is internally
inconsistent — worth checking on your side.

So `v2` is **not** `w/v`; it is `1 - w/v`. Both columns are invariant to
rescaling a `v1` row, so it makes no difference that our `v1` is on a share-like
scale while the reference normalises `v_0 = 1`.

**2. "To calculate `v2` we need each individual's offer set."**

Agreed, and this is the blocking problem. The offer sets we *can* reconstruct
are now shipped as `Supplement/offer_sets.csv` (5,995 OD-departure-date rows),
but they do not identify `w` — and we can show that rather than assert it.
`v2.csv` is therefore produced from an **explicit, documented assumption**, not
from a fit. Full detail in [§7](#7-shadow-attraction--v2csv); the short version:

- the ticketed-sales feed records *purchases*, never availability, so the
  fare-family (RBD) closures the GAM's upsell story turns on are unobserved;
- the one offer-set axis that *is* reconstructible — which departure banks
  operate on each date, expanded from the schedule's `Op Days` — is **rejected
  by the data**: dropping it outright fits daily sales better (deviance 131,774)
  than the GAM at its most favourable `w` (143,752). The build reruns that test
  every time and writes it to `build_report.json`.

**3. "Can you give aircraft type / economy seats per scheduled departure,
joinable to `flight.csv`'s 268 (Oneway_OD, Departure Time) combos?"**

Yes — and no haul-band approximation is needed. `Networkplanning_raw.xlsx`
already carries `Equip`, `Seats`, `First`, `Business`, `Prem Econ` and `Econ`
natively. `flight_capacity.csv` has one row per `(Oneway_OD, Departure Time)`,
**all 268 matched, 1:1, no ambiguity** (no cell has two flight numbers at the
same OD and time). See [§8](#8-capacity--flight_capacitycsv).

Real flight numbers are withheld by the censoring policy, not because they are
missing — set `CENSOR = False` and rebuild if you want the un-obfuscated key.

## Scope

| | |
|---|---|
| Carrier | SQ only |
| Itineraries | direct — `Trip OD Itinerary` names exactly 2 cities |
| Cabin | Economy (`Cabin Class == 'Y'`) |
| Departure window | **2025-01-01 → 2025-05-31** (151 days = 21.571 weeks) |
| Points of sale | AU, GB, ID, IN, KR, SG, US (the 7 POS in the raw feed) |
| Directional ODs | **90** (airport-keyed, censored to `('473', '218')`) |
| Departures | 268 distinct (OD, departure-time) pairs, 1–8 per OD |
| Products | 4 economy fare families |
| Ticketed pax | 2,045,582 |
| Offer-set sub-window | 2025-01-13 → 2025-03-29 (schedule coverage, see §6) |

### Why that departure window

It is the maximal window in which every departure has its **complete** booking
curve inside the data, so weekly demand is not truncated:

- issue dates span `2024-01-01 … 2025-05-31`;
- the observed maximum advance purchase is **365 days**;
- so a departure `D` is fully booked iff `D − 365d ≥ 2024-01-01` **and**
  `D ≤ 2025-05-31`.

The script asserts the 365-day bound at run time and fails loudly if a future
data drop breaks it. As a bonus the window sits inside the
`Networkplanning_raw.xlsx` validity range, so departure times are
contemporaneous with the tickets.

## Sources

| File | Used for |
|---|---|
| `Raw_Sales/POS *NUS Sales Data*.csv` (7 files) | SQ ticketed sales |
| `Raw_Networkplanning/Networkplanning_raw.xlsx` | departure times, equipment, seats, operating days |
| `AUG_Share_Data/All.csv` (MIDT) | non-SQ market share |
| `Airport&CityCodelist.csv` | airport → city, to join sales onto MIDT |
| `api/data_generation/simulation_sales_generator.py` | `FARE_FAMILY_POLICY` |

**The per-POS sales CSVs are the source, not the pre-combined `All.csv`.**
`load_and_clean_sales(from_raw=True)` drops `Flight Number (ALL)` when it
writes `All.csv`, and that column is the only bridge from a ticket to a
scheduled departure.

`FARE_FAMILY_POLICY` is extracted from the simulator source with `ast` (not
imported — the module pulls in Django), so the family definition here can
never drift from the code base. The script raises if the policy changes shape.

## Method

### 1. Price — `flight.csv`

Per spec, per-pax price is `Ticketed OD All-In Rev (SGD) / Ticketed Pax` with
`Ticketed Pax == 0` rows removed (108,297 rows dropped).

`Avg Price` is the **pax-weighted** mean of that ratio (`Σrev / Σpax`) within
each `(OD, departure time, product)` cell, computed **independently per OD** —
no fare is ever pooled across ODs.

Round-trip rows need no adjustment: the data is one row per directional
journey, and revenue is already apportioned per direction. Verified two ways —
forward/backward pax balance is 1.000 within each POS (e.g. POS SG `BKK-SIN`
393,730 vs 390,433), and RT per-pax price is the same order as OO, not 2×.

All 1,072 emitted rows are **observed**; no cell needed the OD-level fallback.
Prices are strictly monotone `Eco_flexi > Eco_standard > Eco_value > Eco_lite`
in all 268 `(OD, departure)` cells.

### 2. Products

From `FARE_FAMILY_POLICY`'s `(SQ, Y)` row, named most- to least-flexible —
which is also strictly descending in observed median fare:

| Product | RBDs | Median fare (SGD) |
|---|---|---|
| `Eco_flexi` | E, B, Y | 458 – 664 |
| `Eco_standard` | W, H, M | 288 – 381 |
| `Eco_value` | N, Q | 210 – 256 |
| `Eco_lite` | K, V | 124 – 150 |

RBDs **X** and **G** are not in the policy and are dropped (200,623 pax,
8.7% of in-window economy).

### 3. Departure times and windows

Tickets join the schedule on `(Orig, Dest, Flight)`. Where a flight appears on
several schedule rows, the representative departure is the `Ops/Week`-weighted
modal `Dep Time` among rows whose validity period overlaps the window.
Equipment and seat counts are carried on that *same* row, so
`flight_capacity.csv` can never disagree with `flight.csv` about which schedule
row a departure is.

**97.34% of pax match.** The 2.66% that do not are flight numbers absent from
the schedule extract — mostly SQ 611/612 (ICN–SIN) and 835/832 (PVG–SIN) — and
are dropped rather than imputed. Full list in `build_report.json`.

Windows follow the reference files: `(8am~12pm)`, `(12pm~6pm)`, `(6pm~10pm)`,
`(10pm~8am)` (wrapping midnight), on local scheduled departure time.

### 4. Non-SQ as no-purchase — `v1.csv`

MIDT gives one canonical direction per market, so SQ's economy booking share
is computed on the **undirected city pair** and applied to both directions.
All 44 city pairs match, covering **100%** of the ticketed pax.

Market size per OD is `sq_pax / sq_share`; non-SQ volume is the remainder.
Then

```
v[product × window] = sq_pax(product, window) / market_pax
no_purchase (= v_0) = non_sq_pax / market_pax
```

`no_purchase = 1 − sq_share` is fitted entirely from non-SQ sales. GAM choice
probabilities are invariant to a common per-market scale, so this is the same
model as any other normalisation — it just reports the fitted values on a
share-like scale. Dividing a row through by its own `no_purchase` recovers the
reference files' `v_0 = 1` convention; `v2.csv` is a ratio and is unaffected
either way.

Airport-keyed ODs sharing a city market (LHR/LGW → LON, NRT/HND → TYO) are
**not** double-counted: grossing each up by the shared share and summing
recovers the city market exactly (verified to 3e-11), because SQ pax is already
split across the airports.

### 5. Filling the zero cells

Before imputation, 632 of the 1,440 cells (44%) were `0` — every one of them an
`(OD, window)` pair the OD does not fly. (Availability is all-or-nothing per
window: within any served window all four products are present, so the gap is
`(OD, window)`-shaped, never product-shaped.)

**The zeros are not missing at random.** An OD's bookings pile up in whatever
bank it happens to fly, so raw pooled window shares measure *SQ's schedule*,
not passengers' time-of-day preference. Filling from those raw shares would
bake the schedule into the preference estimate.

The fill is a **quasi-independence fit on the observed support** — the standard
model for an incomplete contingency table:

```
E[pax[od, product, window]] = θ[od, product] · δ[window]      for served windows
```

`θ` absorbs each OD's own size and product mix; `δ` is the
availability-corrected window preference, identified from the 64 ODs that serve
two or more banks (every window pair co-occurs 21–35 times, and the
availability graph is connected). Fitted by alternating margin scaling —
converged in 18 iterations. Unserved cells are then filled with `θ · δ`.

The correction is material:

| Window | Fitted δ | Raw pooled (supply-confounded) |
|---|---|---|
| `(12pm~6pm)` | 0.3508 | 0.3595 |
| `(6pm~10pm)` | **0.2342** | **0.1996** |
| `(10pm~8am)` | 0.1848 | 0.1643 |
| `(8am~12pm)` | **0.2302** | **0.2765** |

**Chosen over the alternatives on hold-out evidence.** Masking one served
window from each OD that serves ≥3 (122 folds, 488 cells) and predicting it:

| Method | Pax-weighted mean \|log ratio\| | Beats uniform |
|---|---|---|
| **quasi-independence (tied δ)** | **0.6370** | 62.3% |
| quasi-independence, per-product δ | 0.6398 | 62.3% |
| raw pooled window shares | 0.6454 | 57.4% |
| uniform fill | 0.6990 | — |

A per-product `δ[product, window]` is estimable but earns nothing (spread
across products never exceeds 0.035), so the tied form is used — fewer
parameters, more robust.

#### Rows no longer sum to 1 — by design

Observed cells and `no_purchase` are left **exact**: the observed block plus
`no_purchase` still reconstructs the original share vector to 3e-6. The
imputed cells are therefore *additional*, and rows sum to more than 1 (median
1.353, p95 3.511, max 4.225). The surplus is the counterfactual attraction SQ
would add by flying the banks it currently skips: median implied share rises
from 0.403 to 0.546 if all four banks were flown.

This is the correct GAM/SBLP reading — `v` are attraction weights, not shares,
and `π(j | S) = v_j / (vtilde_0 + Σ_{k∈S} vtilde_k)` renormalises against
whatever set `S` you offer. It is also the only convention under which an
offer-set decision can change demand capture at all; rescaling the block back to
`sq_share` would hard-code "SQ's share does not depend on what SQ offers".

**Do not read a row as a share vector.** `Supplement/v1_cell_status.csv` flags
every cell `observed` or `imputed` on the same grid.

Imputed cells have no fare in `flight.csv` — that file is the real schedule and
was deliberately not extended with counterfactual departures. The natural
stand-in fare is the OD × product mean, already carried as `od_product_price`
in `Supplement/flight_price_coverage.csv`.

### 6. Demand — `od_demand.csv`

`Avg Pax` = total market economy pax (SQ + non-SQ) per departure week:

```
Avg Pax = (sq_pax / sq_share) / 21.5714
```

### 7. Offer sets — `Supplement/offer_sets.csv`

One row per `(OD, departure date)` over **2025-01-13 → 2025-03-29**, giving the
set of departure banks operating that day, how many departures, and how many
economy seats. This is the `S` of the GAM: the alternatives a customer shopping
that date can see.

It is built by expanding the schedule's `Op Days` bitmask against
`Eff Date`/`Disc Date` over the calendar. The sub-window is forced by coverage,
not by choice: `Networkplanning_raw.xlsx` is a snapshot taken 2025-01-11 — every
`Eff Date` falls in 2025-01-11…01-19 and all but a handful of `Disc Date`s land
on or before the 2025-03-29 IATA season boundary. Outside that range an "empty
offer set" would be an artefact of the extract, not a real day with no flights.

There is real variation in it: **47 of the 90 ODs** change offer set across
dates (40 have two distinct sets, 7 have three or more), over 5,995 OD-dates.

**What it is not.** Availability here is at the *departure-bank* level only. The
sales feed never records fare-family (RBD) availability, so the closures that
drive the GAM's upsell story — `Eco_lite` shutting as a flight fills — are
invisible. Within any served bank all four families are assumed open.

### 8. Shadow attraction — `v2.csv`

#### `w` is not identifiable from this data, and here is the test

Estimating `w` needs offer sets that vary (paper §2.5). Feeding the §7 offer
sets into the GAM and fitting daily SQ pax by Poisson IPF as

```
mu[i,d] = lambda_i · g[dow] · h[week] · V(S) / D(S),
D(S) = v_0 + V(S) + theta · V(Sbar)
```

against a null that drops the offer-set term `V(S)/D(S)` altogether gives:

| Model | Deviance | Pearson dispersion |
|---|---|---|
| GAM, `theta = 0` (BAM, full recapture) | 143,752 | 25.4 |
| GAM, `theta = 0.5` | 151,950 | 27.7 |
| GAM, `theta = 1` (IDM, no recapture) | 158,304 | 29.8 |
| **null — offer set dropped** | **131,774** | **22.9** |

**Dropping the offer set fits better than the GAM at its most favourable `w`.**
Deviance rises monotonically in `theta`, so an optimiser pins `theta = 0`, but
that is the boundary of the valid range being hit from outside, not an estimate:
the data wants *less* responsiveness to the offer set than even the BAM allows.
A reduced-form check agrees — regressing log daily pax on log economy seats
offered, within OD and with day-of-week and week controls, gives a pax-weighted
elasticity of **0.017**. Day-level bank availability simply does not move
day-level bookings in this feed.

Three reasons, all structural:

1. **Purchases, not availability.** The feed has no offer-set column at all; §7
   is a reconstruction from the schedule, and only at bank level.
2. **7-POS coverage.** Observed pax is 28% of the economy seats offered
   (993,419 against 3,530,860 over the offer-set window; median 0.18 per
   OD-date), so daily counts are a thin sample of each departure — Pearson
   dispersion ≈ 23 against Poisson.
3. **Wrong counterfactual even if it worked.** Day-level variation measures
   substitution when a bank is missing *on one day*, and a passenger who shifts
   to the next day's same bank shows up as recaptured. The SBLP needs the
   period-level counterfactual — a bank removed from the schedule entirely —
   where that escape route does not exist. The two are not the same number.

`assess_shadow_identifiability()` reruns this on every build and writes the
table to `build_report.json`. If a future data drop adds an availability feed,
that function is where the verdict will flip.

#### The transformer cannot identify it either — second test

The simulator's own individual-level choice model (the v1 transformer,
`api/model/transformer_model_v1.pt`) was probed directly: build the market's real
choice set, close each SQ cell, and see where its demand goes. Harness, evidence
and full numbers in [`w_transformer/verdict.md`](w_transformer/verdict.md);
reproduce with `python w_transformer/probe.py`.

It also returns `w ≈ 0`, for a reason that is structural rather than about this
data set. Masking an option in a softmax model reproduces MNL renormalisation
exactly — measured 0.717 against the analytic 0.719 — and that **is** `w = 0`.
A softmax over a fixed logit vector is a BAM. The only channel through which the
model could express `w ≠ 0` is its dependence on the composition of the offered
set, and that runs the wrong way: `num_choices`, the within-set price
rank/percentile/z-score columns and the per-PNR price aggregates make the
*surviving* SQ cells **more** attractive when a sibling is withdrawn, which is
the opposite sign to a shadow attraction. Across the 112 cells the model is
in-domain for, the implied `w/v` is **−0.0156 ± 0.0319**, admissible (`≥ 0`) in
only 28.

Two things the probe does establish:

- an independent bound, **`|w/v| ≤ 0.089`** across all seven probeable markets,
  against the `theta` of median **0.219** the calibration below implies — an
  order of magnitude apart. Either the Ja et al. anchor is too aggressive for
  these markets, or the transformer understates leakage because its 16 SQ cells
  are near-perfect substitutes for one another (same carrier, same routing, same
  nonstop leg; only fare separates them). The probe cannot choose between them;
  it is a directional check from a source independent of the sales feed.
- the checkpoint's own-price elasticity is **correct-signed in all 16 cells**
  (−0.33 to −1.13), so the pervasive wrong-sign own-price response recorded
  against the earlier v1 checkpoint is not present in the GSP re-export.

Scope: the checkpoint's vocabulary is four cities (DEL, LON, SIN, SYD) and four
points of sale, so 7 of the 90 ODs are in-domain. Retraining on the full network
would fix the coverage but not the identification — the masking result is
architectural.

#### So `v2` is an assumption — this one

With `w` unidentified from both directions, `v2.csv` is generated from the
**parsimonious GAM** (paper §2, `w_j = theta · v_j`), with `theta` solved per OD
so the market's mean recapture rate hits a target taken from the literature the
paper itself cites:

> "In markets in which there are multiple flight departures, recapture rates
> typically range between 15%–55%; see Ja et al. (2001)." — p. 213

`TARGET_RECAPTURE_RATE = 0.35`, the midpoint. Under the GAM, removing a set `R`
leaves the open products recapturing

```
r_R(theta) = (1 - theta) · (V - V_R) / (v_0 + theta·V_R + V - V_R)
```

which is 0 at `theta = 1` and maximal at `theta = 0`, and is strictly decreasing
in between — so a bisection on `theta` is exact. `RECAPTURE_SCOPE = "window"`
makes `R` a whole departure bank, matching the *cross-flight* recapture the
quoted range measures; the resulting `theta` implies a median **40.2%** recapture
at the finer scope of one fare family on one bank, which is the right ordering
(a closer substitute survives).

#### `theta` carries a per-cell shape

One `theta` per OD would put the *same* ratio in all 16 product columns of a row,
leaving 1440 cells with only 61 distinct values. The paper's GAM (eq. 2) does not
require that — it allows a free `w_j` per product — so a per-cell shape is *more*
faithful to it, not less. What the shape may not do is invent information, so it
is built only from quantities already measured elsewhere in this build:

```
theta_j = kappa_i · d_j        d_j = exp(alpha·tau_j + beta·B_j),
                               normalised to attraction-weighted mean 1 per OD
```

| driver | what | source |
|---|---|---|
| `tau_j` | time isolation — circular minutes from that bank to the nearest **served** other bank of the same OD, over 720; single-bank ODs take 1.0 | the real schedule (§3) |
| `B_j` | fare barrier — attraction-weighted mean of `max(0, log f_k − log f_j)` over the other 15 cells, i.e. how much dearer the survivors are | the observed per-cell fares (§1) |

An isolated bank has no close substitute, so its demand leaks rather than moves —
which is what the quoted Ja et al. range conditions on ("in markets in which there
are multiple flight departures"). The fare barrier is the sell-up channel `w`
exists to carry: closing `Eco_lite` forces a buy-up and loses demand, closing
`Eco_flexi` lets the passenger buy down at no cost. That ordering comes out of the
build rather than being imposed — mean `v2` by product is `Eco_flexi` 0.794 >
`Eco_standard` 0.780 > `Eco_value` 0.747 > `Eco_lite` 0.661.

Because `d` is normalised to mean 1 inside each OD, **`kappa_i` is still the
single per-OD number the recapture target pins down** — §8's calibration story is
unchanged, and the per-OD achieved recapture is still exactly 0.35. Setting
`alpha = beta = 0` collapses `d` to 1 and reproduces the previous flat build
bit-for-bit, so it is the corner of this one and an `(alpha, beta)` sweep is a
ready-made sensitivity table.

Every one of the 64 ODs that can reach the target still achieves it to within
1e-6, and the median `kappa` is **0.219** — unchanged from the flat build, as it
must be. `v2` now has **1430 distinct values in 1440 cells**, no row is constant,
and the within-row spread is 0.223 at the median.

**26 ODs cannot reach 35% even at `theta = 0`** — all 26 are single-bank ODs where
SQ is small against the no-purchase alternative, so the market structure itself
caps recapture (achieved minimum 11.8%). Rather than clamp them to the
`kappa = 0` *boundary* — which is a flat `v2 ≡ 1` row carrying no cell-level
information — they target `SHADOW_UNREACHABLE_FRACTION` (0.95) of their own
attainable maximum, which is an interior point. They are still counted in
`build_report.json` as `shadow_ods_target_unreachable`.

Six cells (0.4%) land at `v2 = 0.000000`, i.e. locally IDM — the most isolated,
hardest-to-substitute cells of the highest-`kappa` ODs, where `kappa · d` saturates
the `theta ≤ 1` bound. Lower `alpha`/`beta` to pull them off the boundary;
`shadow_theta_cells.clipped_at_1` in the report counts them.

#### Changing the assumption

Constants at the top of the script, no other edits:

| Constant | Effect |
|---|---|
| `SHADOW_MODE = "pgam_shape"` | per-OD calibration + the per-cell shape above (default) |
| `SHADOW_MODE = "pgam"` | flat `theta` per OD — the `alpha = beta = 0` corner |
| `SHADOW_MODE = "pgam_fixed"` | flat `w = PGAM_THETA · v` everywhere |
| `SHADOW_MODE = "bam"` | `theta = 0` → `v2 ≡ 1.0`, full recapture |
| `SHADOW_MODE = "idm"` | `theta = 1` → `v2 ≡ 0.0`, no recapture |
| `SHADOW_SHAPE_ALPHA` / `SHADOW_SHAPE_BETA` | weight on time isolation / fare barrier; both 0 ⇒ `"pgam"` |
| `TARGET_RECAPTURE_RATE` | the 0.35 above — sweep it for a sensitivity study |
| `RECAPTURE_SCOPE` | `"window"` (a bank) or `"cell"` (one family on one bank) |
| `SHADOW_UNREACHABLE_FRACTION` | interior target for ODs that cannot reach the rate; 1.0 restores the old boundary clamp |

`bam` and `idm` are the two bounds of Figure 1 in the paper, so sweeping
`TARGET_RECAPTURE_RATE` reproduces its `theta`-axis on real data.

#### Reading `no_purchase` in `v2`

`v2`'s `no_purchase` is `vtilde_0/v_0 = 1 + sum_j theta_j v_j / v_0`, and `V/v_0`
is large wherever SQ dominates a market — so it ranges 1.005 to 20.01
(median 1.27), well above the reference file's 1.40–1.55. That is not a scale artefact: the ratio is
invariant to rescaling a `v1` row. It says SQ's total attraction in those markets
is a large multiple of the no-purchase alternative, which is what a ~99% MIDT
share means. The SBLP balance constraint pairs it with a correspondingly small
`x_0`.

### 9. Capacity — `flight_capacity.csv`

One row per `(Oneway_OD, Departure Time)`, joinable straight onto `flight.csv`.
**All 268 cells matched**, and no cell carries two flight numbers at the same OD
and time, so the join is 1:1.

| Column | Meaning |
|---|---|
| `Aircraft Type` | `Equip` — SQ's own equipment code |
| `Total Seats` / `Y Seats` / `W Seats` | `Seats` / `Econ` / `Prem Econ` |
| `Ops/Week` | weekly frequency of that departure |
| `Y Seats/Week` | `Σ (Econ × Ops/Week)` over the schedule rows |
| `all_premium_no_Y` | flags the 2 cells with no economy cabin |
| `equip_varies`, `Y Seats min`, `Y Seats max` | equipment swap inside the cell |

**Use the right unit.** `Y Seats` is per departure — the unit a per-flight
capacity constraint wants, and the one the reference `.lp` files use with their
hard-coded `187`. `Y Seats/Week` is the unit `od_demand.csv`'s `Avg Pax` is
already in. Mixing them is the easiest mistake to make here: the reference
`SBLP_CA_Label/*.lp` constrain a single departure to 187 seats while the balance
constraint carries a *weekly* market of 38,965 pax.

Two things to know:

- **2 cells have `Y Seats = 0`, correctly.** SQ 35/36 (SIN–LAX) is the
  all-premium A350-900ULR: 161 seats, 67 J + 94 W, no economy cabin. The sales
  feed nonetheless reports 1,259 Y-cabin pax on those two departures, which is a
  question about the feed's cabin coding, not about the seat count. They are
  flagged rather than silently patched; `W Seats` (94) is the sensible
  substitute if you need a positive capacity.
- **Equipment swaps.** Where a `(flight, departure time)` cell is flown by more
  than one type across the season, `Aircraft Type` and the seat columns are the
  `Ops/Week`-weighted modal *configuration* — type and cabin taken off the same
  schedule row, never mixed. `Y Seats min`/`max` give the observed spread.

`Y Seats` by type over the 268 cells:

| Type | Y Seats | Departures |
|---|---|---|
| `787` (787-10) | 301 | 66 |
| `359` (A350-900, medium-haul) | 263 | 56 |
| `7M8` (737 MAX 8) | 144 | 48 |
| `359` (A350-900, regional) | 187 | 40 |
| `77W` (777-300ER) | 184 | 32 |
| `738` (737-800) | 150 | 12 |
| `388` (A380) | 343 | 12 |
| `359` (A350-900ULR) | 0 | 2 |

Median `Y Seats` over the 268 cells is **225**, spanning 144–343, and 27 cells
see an equipment swap across the season. The reference `.lp` files' hard-coded
`187` is real — it is the A350 regional cabin, and it is the right number for 40
of the 268 departures. For the other 228 it is off by −45% to +30%: a 787-10
departure carries 301 economy seats and an A380 343, while a 737 MAX 8 carries
only 144.

## Censoring

Every IATA airport and city code in every output is replaced by a 3-digit
surrogate drawn from `100–999`. The mapping is seeded (`CENSOR_SEED =
20250727`), so re-running reproduces it exactly and all outputs stay mutually
consistent.

- **Airports and cities share one surrogate pool**, so a code means the same
  thing wherever it appears. City codes are censored too — leaving `LON` or
  `TYO` in `od_market_size.csv` would immediately re-identify the airports
  beneath them.
- **All outputs are censored, not just the deliverables.** `od_market_size.csv`
  pairs per-OD demand with route names, so censoring only `flight.csv` would
  leave a trivial join that undoes it.
- **Flight numbers are withheld.** `build_report.json` reports schedule-join
  losses per OD instead of per flight: a public timetable turns `SQ 611` into a
  route in one lookup.
- Set `CENSOR = False` at the top of the script to rebuild with real codes;
  that also deletes any stale key file.

### What this does not protect

This is obfuscation, not anonymisation. Departure times, fare levels, market
sizes and the number of daily frequencies are all preserved *by design* —
they are the data. Any one of them can be matched against a public schedule to
recover a route; a 4-departure OD with a 01:10 and a 23:45 bank is not hard to
place. **`flight_capacity.csv` weakens this further**: aircraft type plus seat
count plus departure time is close to a fingerprint, and the two `Y Seats = 0`
rows identify SIN–LAX outright. Withholding `airport_code_map.csv` stops casual
reading and nothing stronger. Treat the outputs as commercially sensitive
regardless.

## Outputs

| File | Rows | Notes |
|---|---|---|
| `flight.csv` | 1,072 | index, `Oneway_OD`, `Departure Time`, `Oneway_Product`, `Avg Price` |
| `od_demand.csv` | 90 | index, `Oneway_OD`, `Avg Pax` (per week) |
| `v1.csv` | 90 | `OD Pairs`, 16 `product*(window)` columns, `no_purchase` — GAM attractions `v_j`, `v_0`. No zeros |
| `v2.csv` | 90 | same grid — shadow ratios `vtilde/v`; `no_purchase` is `vtilde_0/v_0`. 1430 distinct values in 1440 cells |
| `flight_capacity.csv` | 268 | aircraft type and per-cabin seats per departure |
| `build_report.json` | — | every filter count, join rate, fitted δ, θ, and the identifiability table |
| `Supplement/v1_cell_status.csv` | 90 | same grid, each cell `observed` or `imputed` |
| `Supplement/offer_sets.csv` | 5,995 | per (OD, departure date) served banks and seats |
| `Supplement/od_market_size.csv` | 90 | SQ pax, MIDT share, market size, non-SQ pax per OD |
| `Supplement/flight_price_coverage.csv` | 1,072 | per-cell pax and observed-vs-filled flag |
| `Supplement/airport_code_map.csv` | 56 | **the key — withhold when sharing** (46 airports + 10 cities) |

## Caveats

- **`v2.csv` is an assumption, not a measurement.** This is the largest caveat
  in the build and §8 is entirely about it. `v1` is estimated from data; `v2` is
  a calibrated convention with one free number in it (`TARGET_RECAPTURE_RATE`)
  plus two shape weights (`SHADOW_SHAPE_ALPHA`/`_BETA`). The *level* is the
  literature's; the *shape* is read off this build's own schedule and fares. Two
  independent tests — the offer-set fit and the transformer probe — say `w` is
  not identifiable from anything available here, so no amount of extra work on
  these inputs turns `v2` into a measurement.
- **MIDT is GDS-only.** It undercounts carriers selling mostly direct (LCCs in
  particular), so `sq_share` is likely optimistic and market size conservative.
  `od_market_size.csv` carries `sq_share_all_departures` — the same share over
  MIDT's full departure range — as a sensitivity; it moves the BKK–SIN share
  from 0.329 to 0.290.
- **MIDT booking dates stop at 2025-03-31**, so April–May 2025 departures are
  booking-truncated on both the SQ and non-SQ side. The *ratio* is far more
  robust to this than either level.
- **Market size is defined on the retained scope.** The numerator is filtered
  SQ pax (economy, in-policy RBDs, schedule-matched) while `sq_share` is over
  all SQ economy MIDT bookings, so the market is mildly understated. This is
  deliberate: it keeps dropped-RBD pax out of `no_purchase`, which must mean
  "chose a non-SQ carrier" and nothing else.
- **7 POS only.** Traffic originating outside AU/GB/ID/IN/KR/SG/US is in
  neither feed, so these are 7-POS markets, not global ones. This is also why
  observed pax is only 28% of offered economy seats — the feed is a slice of
  each departure, not the whole cabin.
- **The imputed `v1` cells are extrapolation, not measurement.** 44% of the
  matrix is filled, and for the 26 single-window ODs the fill is a 5.4×
  extrapolation from one observed bank to four. Hold-out error on *served*
  windows (0.637 mean |log ratio|) is the optimistic case — those ODs at least
  had two banks to learn from. Treat single-window ODs' imputed cells as
  indicative only; `Supplement/v1_cell_status.csv` identifies them.
- **The window effect is global.** One δ is shared by all 90 ODs. Time-of-day
  preference plausibly differs between a 1-hour regional hop and a 13-hour
  long-haul; the data would support a per-haul-band δ, but with 26 ODs
  contributing nothing to identification it was not worth the variance.
- **Offer sets cover 76 of the 151 departure days.** The schedule extract does
  not reach past the 2025-03-29 season boundary, so `offer_sets.csv` is a
  sub-window of the departure window. `v1` and `od_demand` still use all 151
  days.
- `Airport&CityCodelist.csv` is missing ICN, PVG and KNO; the script patches
  them to SEL, SHA and MES. The code list itself is untouched.
- Cross-Pacific 5th-freedom ODs (`LAX-NRT`, `NRT-LAX`, `JFK-FRA`, `FRA-JFK`,
  `IAH-MAN`, `MAN-IAH`) are in scope — they are genuine SQ 2-city directs.
