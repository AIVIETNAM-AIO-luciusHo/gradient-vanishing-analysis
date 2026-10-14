"""Dot 3 analysis: Total Degradation (TD) and decay_slope from results_training.json.

No new training: reads results_training.json only (runs[i].epochs[e].gnorm = per-layer ||g||_F of
stem, blocks, head = L+2 numbers; log10_rho = log10(||g_stem|| / ||g_head||)).
  TD          = log10(||g_first|| / ||g_last||)  (== log10_rho of the training tables)
  decay_slope = OLS slope of log10(||g^[l]||) against layer index l = 0..L+1 (orders of magnitude
                per layer); slope > 0 <=> gradient grows toward the head (vanishing toward the stem).
  R^2         = R^2 of that same regression.
Elements with gnorm = 0 or non-finite are dropped from the regression (never log10(0)); the count is
reported; fewer than 3 usable points -> n/a.  Diverged runs are skipped (n=k as in the source table).
Run: python analysis_decay.py   (writes results_decay.md next to this file)
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
EPOCHS = [0, 1, 5, 10]
MIN_PTS = 3


def ms(vals):
    a = np.array(vals, dtype=np.float64)
    if len(a) == 0:
        return "n/a"
    s = f"{a.mean():.3g} ± {a.std(ddof=1):.3g}" if len(a) > 1 else f"{a.mean():.3g} ± n/a"
    return s


def ols(g):
    """slope, R^2 of log10(g) vs index over the finite positive entries; (None, None, k_dropped)."""
    idx = [i for i, v in enumerate(g) if v is not None and math.isfinite(v) and v > 0]
    dropped = len(g) - len(idx)
    if len(idx) < MIN_PTS:
        return None, None, dropped
    x = np.array(idx, dtype=np.float64)
    y = np.log10(np.array([g[i] for i in idx], dtype=np.float64))
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    slope = ((x - xm) * (y - ym)).sum() / sxx
    ss_tot = ((y - ym) ** 2).sum()
    resid = y - (ym + slope * (x - xm))
    r2 = 1.0 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(r2), dropped


def md_td_cells():
    """TD (log10(rho)) cells of results_training.md, keyed (arch, L, act, bn, epoch)."""
    cells, section = {}, None
    for line in (HERE / "results_training.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            section = line
            continue
        if section and section.startswith("## log10(rho)") and line.startswith("| ") and "---" not in line:
            p = [t.strip() for t in line.strip().strip("|").split("|")]
            if p[0] in ("plain", "resnet"):
                for e, c in zip(EPOCHS, p[4:8]):
                    cells[(p[0], int(p[1]), p[2], p[3], e)] = c
    return cells


def main():
    data = json.loads((HERE / "results_training.json").read_text(encoding="utf-8"))
    runs = data["runs"]
    # group: (arch, L, act, bn, epoch) -> list of per-run records for runs that reached the epoch
    G = defaultdict(list)
    for r in runs:
        for e in r["epochs"]:
            if e["epoch"] in EPOCHS and e["gnorm"] is not None:
                g = e["gnorm"]
                assert len(g) == r["L"] + 2, (r["arch"], r["L"], len(g))
                td = math.log10(g[0] / g[-1])
                assert abs(td - e["log10_rho"]) <= 1e-9 * max(1.0, abs(td)), (r["arch"], r["L"], r["seed"], e["epoch"])
                slope, r2, drop = ols(g)
                G[(r["arch"], r["L"], r["act"], r["bn"], e["epoch"])].append(
                    {"td": e["log10_rho"], "slope": slope, "r2": r2, "drop": drop, "seed": r["seed"]})
    print("assert OK: TD recomputed from gnorm[0]/gnorm[-1] equals log10_rho in JSON for every (run, epoch)")

    md = md_td_cells()
    nseeds = len({r["seed"] for r in runs})
    rows, nmatch, ncell = [], 0, 0
    for arch in ("plain", "resnet"):
        for L in (8, 16, 32):
            for act, bn in (("relu+kaiming", "off"), ("relu+kaiming", "on"),
                            ("sigmoid+xavier", "off"), ("sigmoid+xavier", "on")):
                actkey = act.split("+")[0]
                for ep in EPOCHS:
                    rec = G.get((arch, L, actkey, bn == "on", ep), [])
                    k = len(rec)
                    sfx = f" (n={k})" if 0 < k < nseeds else ""
                    td_s = (ms([x["td"] for x in rec]) + sfx) if k else "n/a"
                    ok_sl = [x for x in rec if x["slope"] is not None]
                    sl_s = (ms([x["slope"] for x in ok_sl]) + (f" (n={len(ok_sl)})" if 0 < len(ok_sl) < nseeds else "")) if ok_sl else "n/a"
                    r2_s = (ms([x["r2"] for x in ok_sl]) + (f" (n={len(ok_sl)})" if 0 < len(ok_sl) < nseeds else "")) if ok_sl else "n/a"
                    drop = sum(x["drop"] for x in rec)
                    ncell += 1
                    nmatch += int(td_s == md[(arch, L, act, bn, ep)])
                    assert td_s == md[(arch, L, act, bn, ep)], (arch, L, act, bn, ep, td_s, md[(arch, L, act, bn, ep)])
                    rows.append([arch, str(L), act, bn, str(ep), str(k), td_s, sl_s, r2_s, str(drop)])
    print(f"assert OK: TD cells match results_training.md: {nmatch}/{ncell}")

    cols = ["arch", "L", "act/init", "BN", "epoch", "n runs", "TD = log10(rho)", "decay_slope", "R^2",
            "dropped gnorm entries"]
    table = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(r) + " |" for r in rows]
    print("\n".join(table))
    text = (
        "# Total Degradation (TD) and decay_slope from results_training.json (mean +/- std, ddof=1, over seeds)\n\n"
        f"`source: results_training.json (no new training) | seeds={sorted({r['seed'] for r in runs})} | "
        f"epochs {EPOCHS} | numpy {np.__version__}`\n\n"
        "- TD = log10(||g_first|| / ||g_last||) = log10(rho) of results_training.md (g_first = stem, g_last = head); "
        "TD < 0: attenuation toward the first layer, TD > 0: amplification.\n"
        "- decay_slope = OLS slope of log10(||g^[l]||_F) vs layer index l = 0..L+1 (stem, blocks, head; L+2 points), "
        "in orders of magnitude per layer; slope > 0: gradient grows toward the head (vanishing toward the stem). "
        "R^2 is the R^2 of the same regression.\n"
        "- gnorm = 0 or non-finite entries are dropped from the regression (count in the last column); "
        "fewer than 3 points -> n/a. Diverged runs are skipped; (n=k) = only k seeds not diverged. "
        "TD column equals the log10(rho) table of results_training.md cell by cell (asserted).\n\n"
        + "\n".join(table) + "\n")
    (HERE / "results_decay.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
