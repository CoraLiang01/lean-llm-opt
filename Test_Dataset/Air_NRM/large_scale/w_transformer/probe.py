"""Probe the v1 transformer for per-cell substitution structure.

For one Air_NRM OD the probe scores a baseline choice set, then closes each SQ
cell in turn and measures the **recapture rate**

    r_j = [ P_SQ\\{j}(after) - (P_SQ(before) - P_j(before)) ] / (loss of P_j)

— the share of cell ``j``'s demand the remaining SQ cells pick up rather than
lose to a non-SQ carrier.

WHAT THIS CAN AND CANNOT MEASURE
--------------------------------
Three ways to operationalise "close cell j" were tested (``verdict.md``):

* **mask** — force ``j``'s logit to ``-inf``, every other row's features
  computed on the full set. Reproduces the analytic MNL renormalisation to
  within 0.031, i.e. ``w = 0`` **by construction**. A softmax over a fixed
  logit vector *is* a BAM; it carries no information about ``w``.
* **rebuild** — drop the row and rebuild the whole feature pipeline. Recapture
  lands at 1.16 on average, above the BAM bound (0.72) in every cell, i.e.
  ``w < 0`` — outside the GAM's admissible range. The set-composition features
  (``num_choices``, the within-set price rank/percentile/z-score, the per-PNR
  price aggregates) make the survivors *more* attractive when a sibling is
  withdrawn, which is the opposite sign to a shadow attraction.
* **price-out** (used here) — multiply ``j``'s fare by ``CLOSE_PRICE_FACTOR``,
  leaving the set size and therefore ``num_choices`` untouched. Still above the
  BAM bound in 16/16 cells, but far the most stable of the three.

So the transformer does **not** identify the level of ``w``. What it does carry
is a per-cell *shape* that is reproducible to Pearson 0.999 across independent
customer samples. That shape is this probe's product; the level stays anchored
to the Ja et al. (2001) recapture range in ``build_air_nrm_inputs.py``.

SCOPE. The checkpoint's vocabulary is four cities (DEL, LON, SIN, SYD) and four
points of sale (AU, GB, IN, SG), and the live ``Itinerary.csv`` asset is on the
same network — so 7 of the 90 Air_NRM ODs are in-domain. See ``verdict.md``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parents[2] / "app" / "backend"
sys.path.insert(0, str(_BACKEND))
sys.path.insert(0, str(_HERE))

from api.model.feature_engineer import (  # noqa: E402
    build_batch_feature_pack_from_inputs,
    build_batch_model_inputs_from_feature_pack,
)
from api.model.model_loader import load_trained_model  # noqa: E402

from choice_sets import (  # noqa: E402
    IN_VOCAB_ODS, PRODUCTS, WINDOWS, AirNrmGrid, OalBlock,
    sample_customers, sq_block,
)

_MODEL_DIR = _BACKEND / "api" / "model"
MODEL_PATH = _MODEL_DIR / "transformer_model_v1.pt"
METADATA_PATH = _MODEL_DIR / "metadata.json"
VOCAB_PATH = _MODEL_DIR / "vocabularies.json"

N_CUSTOMERS = 200
MAX_OAL_ROWS = 180          # + 16 SQ rows, inside metadata's max_choice_count=300
BATCH_SIZE = 64
CLOSE_PRICE_FACTOR = 20.0   # "closed" = priced out of reach, set size unchanged
OAL_PRICE_SCALES = (1.0, 0.30)   # primary = model-native; second = sensitivity
SEED = 20250729


def _pack_inputs(cust: pd.DataFrame, choices_list: list[pd.DataFrame]) -> dict:
    pack = build_batch_feature_pack_from_inputs(cust, choices_list)
    return build_batch_model_inputs_from_feature_pack(
        pack, metadata_path=str(METADATA_PATH), vocabulary_path=str(VOCAB_PATH),
    )


def _forward(model, inputs: dict, device) -> torch.Tensor:
    b = inputs["choice_numeric"].shape[0]
    out = []
    with torch.no_grad():
        for s in range(0, b, BATCH_SIZE):
            batch = {k: v[s:s + BATCH_SIZE].to(device) for k, v in inputs.items()}
            out.append(torch.softmax(model(batch), dim=-1).cpu())
    return torch.cat(out, dim=0)


def _frames(cust: pd.DataFrame, sq: pd.DataFrame, oal: pd.DataFrame,
            oal_scale: float) -> list[pd.DataFrame]:
    o = oal.copy()
    o["Price"] = o["Price"] * oal_scale
    frame = pd.concat([sq, o], ignore_index=True)
    frame["Departure_Time_OWInbound"] = 0
    frame["StopsInbound"] = 0
    frame["Duration_OWInbound"] = 0
    frame["Routing_Inbound"] = ""
    out = []
    for pid in range(len(cust)):
        f = frame.copy()
        f["PNR_ID"] = pid
        out.append(f)
    return out


def _probs(model, cust, sq, oal, oal_scale, device):
    frames = _frames(cust, sq, oal, oal_scale)
    probs = _forward(model, _pack_inputs(cust, frames), device)
    frame = frames[0]
    return probs[:, :len(frame)].numpy(), frame


def probe_od(model, grid: AirNrmGrid, oal_src: OalBlock, od, device,
             rng: np.random.Generator, oal_scale: float) -> dict:
    market = IN_VOCAB_ODS[od]
    t0 = time.perf_counter()

    cust = sample_customers(od, market[2], N_CUSTOMERS, rng)
    sq = sq_block(grid, od)
    oal = oal_src.build(market, MAX_OAL_ROWS, rng)
    pax = cust["pax"].to_numpy(float)

    p0, f0 = _probs(model, cust, sq, oal, oal_scale, device)
    is_sq = f0["is_sq"].to_numpy(bool)
    cell = f0["cell"].to_numpy(int)
    tot0 = p0[:, is_sq].sum(axis=1)

    r = np.zeros(16)
    bam = np.zeros(16)
    cells = {}
    for j in range(16):
        pj = p0[:, cell == j].sum(axis=1)

        sq_closed = sq.copy()
        sq_closed.loc[sq_closed.cell == j, "Price"] *= CLOSE_PRICE_FACTOR
        p1, f1 = _probs(model, cust, sq_closed, oal, oal_scale, device)
        resid = p1[:, f1["cell"].to_numpy(int) == j].sum(axis=1)
        others1 = p1[:, f1["is_sq"].to_numpy(bool)].sum(axis=1) - resid

        loss = pj - resid
        gain = others1 - (tot0 - pj)
        ok = loss > 1e-9
        rc = np.zeros_like(loss)
        rc[ok] = gain[ok] / loss[ok]
        # Pure-MNL renormalisation bound: the most any BAM (w=0) can recapture.
        rb = (tot0 - pj) / (1.0 - pj)

        w = pax * loss
        ws = w.sum()
        r[j] = float((w * rc).sum() / ws) if ws > 0 else np.nan
        bam[j] = float((w * rb).sum() / ws) if ws > 0 else np.nan
        cells[f"{PRODUCTS[j // 4]}*{WINDOWS[j % 4]}"] = {
            "recapture": round(float(r[j]), 5),
            "bam_bound": round(float(bam[j]), 5),
            "admissible": bool(r[j] <= bam[j] + 1e-9),
            "baseline_prob": round(float(pj.mean()), 6),
            "residual_after_close": round(float(resid.mean()), 6),
        }

    return {
        "od": f"{od[0]}-{od[1]}",
        "market": "-".join(market),
        "oal_price_scale": oal_scale,
        "n_customers": int(len(cust)),
        "n_oal_rows": int(len(oal)),
        "baseline_sq_share": round(float(tot0.mean()), 5),
        "sq_share_midt": round(float(grid.sq_share[od]), 5),
        "recapture": [round(float(x), 6) for x in r],
        "bam_bound": [round(float(x), 6) for x in bam],
        "n_cells_admissible": int((r <= bam + 1e-9).sum()),
        "cells": cells,
        "wall_clock_s": round(time.perf_counter() - t0, 1),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(str(MODEL_PATH), device=device)
    model.eval()
    grid = AirNrmGrid()
    oal_src = OalBlock()

    results = []
    for scale in OAL_PRICE_SCALES:
        for od in IN_VOCAB_ODS:
            if od not in grid.row_of:
                continue
            rng = np.random.default_rng(SEED)   # same customers across scales
            res = probe_od(model, grid, oal_src, od, device, rng, scale)
            print(f"{res['od']:9s} scale {scale:<5} SQ share {res['baseline_sq_share']:.3f} "
                  f"(MIDT {res['sq_share_midt']:.3f})  recapture "
                  f"{np.min(res['recapture']):.3f}..{np.max(res['recapture']):.3f} "
                  f"| BAM bound {np.mean(res['bam_bound']):.3f} "
                  f"| admissible {res['n_cells_admissible']}/16  "
                  f"[{res['wall_clock_s']}s]", flush=True)
            results.append(res)

    out = _HERE / "recapture_measured.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
