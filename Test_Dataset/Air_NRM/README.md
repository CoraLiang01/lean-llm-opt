# Air_NRM

Two cases of the same airline network-revenue-management problem, both built
from **Singapore Airlines** data, differing only in scale. They share one file
schema, so anything that reads one reads the other.

```
Air_NRM/
├── small_scale/     3-airport toy instance — what the notebooks run today
├── large_scale/     SQ direct 2-city markets, real-data build
└── A-general-attraction-model-...pdf    the SBLP/GAM paper both cases follow
```

## The two cases

| | `small_scale/` | `large_scale/` |
|---|---|---|
| Markets (ODs) | 4 | **90** |
| Departures | — | **268** (OD × departure time) |
| Price rows (`flight.csv`) | 38 | **1,072** |
| Products | 3 fare classes | 4 economy fare families |
| Queries | 15 CA + 20 NP-Flow | 15 CA |
| Ground-truth `.lp` | 15 + 21, in `SBLP_*_Label/` | not yet generated |
| Provenance | hand-built reference instance | built from the raw SIA feed, documented end to end |

## Shared schema

| File | Contents |
|---|---|
| `flight.csv` | `Oneway_OD`, `Departure Time`, `Oneway_Product`, `Avg Price` |
| `od_demand.csv` | `Oneway_OD`, `Avg Pax` (per week) |
| `v1.csv` | GAM attraction values `v_j` per product × window, plus `no_purchase` (`v_0`) |
| `v2.csv` | shadow-attraction ratios `ṽ_j / v_j`; `no_purchase` is `ṽ_0 / v_0` |
| `query_*.csv` | the natural-language problem statements fed to the pipeline |

`large_scale/` additionally carries `flight_capacity.csv` (aircraft type and
per-cabin seats per departure), a `Supplement/` folder, and the build scripts.

**The build scripts are documentation, not part of the pipeline.** No notebook
or experiment script runs them. They record how the data provider derived each
file — provenance for the paper's data section. Two of the four
(`build_air_nrm_inputs.py`, `w_transformer/probe.py`) need the raw SIA feed and
the booking-simulator code base, neither of which is in this repository; see
[`large_scale/README.md`](large_scale/README.md).

## Which one the code uses

The notebooks (`LEAN_LLM_OPT_*_Air_NRM.ipynb` and both ablations) currently
read **`small_scale/`** only. In `exp_config.yaml`:

```yaml
Air-NRM-CA:     Test_Dataset/Air_NRM/small_scale/query_CA.csv
Air-NRM-NP:     Test_Dataset/Air_NRM/small_scale/query_NP_Flow.csv
Air-NRM-CA-LS:  Test_Dataset/Air_NRM/large_scale/query_largescale_CA.csv
```

`large_scale/` is registered but not yet wired into a notebook run.

## Before you use `large_scale/`

Read [`large_scale/README.md`](large_scale/README.md) first — it is long, and
two things in it change how the numbers should be interpreted:

- **`v2.csv` is a calibrated assumption, not a measurement.** The shadow
  attraction `w` is not identifiable from the available data, and the README
  documents two independent tests showing why.
- **Airport and city codes are censored** to 3-digit surrogates. The key is
  `large_scale/Supplement/airport_code_map.csv` — the one file to withhold when
  sharing outside the team.

Also note `large_scale/`'s `.lp` ground truth does not exist yet, so the
scoring path (`score_runs.py` → `aggregate_runs.py`) has nothing to compare
against for that case.
