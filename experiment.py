"""Section VI experiment: gradient norms at initialisation, 5 seeds x 3 depths.

L linear layers (L-1 hidden of width 64 + 1 output of width 1), biases = 0.
  A. Sigmoid + xavier_normal_
  B. ReLU    + kaiming_normal_(nonlinearity='relu')
One forward + backward pass at init, no optimizer step, float32, CPU.
Run: python experiment.py   (writes results.md next to this file)
"""
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SEEDS = [42, 100, 2024, 2026, 9999]
DEPTHS = [5, 10, 20]
D, N, LR = 64, 256, 0.1
EPS = torch.finfo(torch.float32).eps
CONDS = {"A: sigmoid + Xavier": "sigmoid", "B: ReLU + He": "relu"}

torch.use_deterministic_algorithms(True)
torch.set_default_dtype(torch.float32)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build(L, act):
    layers = []
    for l in range(L):
        lin = nn.Linear(D, 1 if l == L - 1 else D)
        if act == "sigmoid":
            nn.init.xavier_normal_(lin.weight)
        else:
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        if l < L - 1:  # no activation after the output layer
            layers.append(nn.Sigmoid() if act == "sigmoid" else nn.ReLU())
    return nn.Sequential(*layers)


def run(seed, L, act):
    seed_all(seed)                     # same data for both conditions
    x, y = torch.randn(N, D), torch.randn(N, 1)
    net = build(L, act)
    nn.MSELoss()(net(x), y).backward()
    Ws = [m.weight for m in net if isinstance(m, nn.Linear)]
    g = [torch.linalg.norm(W.grad).item() for W in Ws]  # Frobenius, weights only
    # conservative bound c = max sigma' * sigma_max(W), averaged over the layers
    # the backward product passes through (k = 2..L); max sigma' = 1/4 or 1
    smax = np.mean([torch.linalg.matrix_norm(W.detach(), ord=2).item() for W in Ws[1:]])
    c = (0.25 if act == "sigmoid" else 1.0) * smax
    W1 = Ws[0].detach()
    r = LR * g[0] / torch.linalg.norm(W1).item()        # relative update of layer 1
    W1new = W1 - LR * Ws[0].grad
    frozen = torch.equal(W1new, W1)                     # update fully lost to rounding
    unchanged = (W1new == W1).float().mean().item()     # fraction of entries lost
    return {"g1": g[0], "gL": g[-1], "rho": g[0] / g[-1],
            "log10rho": float(np.log10(g[0] / g[-1])), "r": r, "frozen": frozen, "c": c,
            "unchanged": unchanged}


def ms(vals):
    a = np.array(vals, dtype=np.float64)
    return f"{a.mean():.3g} \u00b1 {a.std(ddof=1):.3g}"


def main():
    # self-check: same seed twice -> identical numbers
    assert run(42, 5, "sigmoid") == run(42, 5, "sigmoid")
    assert run(42, 5, "relu") == run(42, 5, "relu")

    head = (f"torch {torch.__version__} | python {sys.version.split()[0]} | float32 CPU | "
            f"seeds={SEEDS} | D={D} N={N} lr={LR} | eps32={EPS:.3g}")
    cols = ["condition", "L", "c (bound)", "||g[1]||_F", "||g[L]||_F", "rho = ||g[1]||/||g[L]||",
            "log10(rho)", "r = lr*||g[1]||/||W[1]||", "r/eps32", "W[1] entries unchanged", "frozen (of 5)"]
    rows = []
    for name, act in CONDS.items():
        for L in DEPTHS:
            res = [run(s, L, act) for s in SEEDS]
            col = lambda k: [d[k] for d in res]
            rows.append([name, str(L), ms(col("c")), ms(col("g1")), ms(col("gL")), ms(col("rho")),
                         ms(col("log10rho")), ms(col("r")),
                         ms([v / EPS for v in col("r")]), ms(col("unchanged")),
                         f"{sum(col('frozen'))}/{len(SEEDS)}"])
            for s, d in zip(SEEDS, res):
                print(f"  seed={s:5d} L={L:2d} {act:7s} g1={d['g1']:.4e} gL={d['gL']:.4e} "
                      f"rho={d['rho']:.4e} r={d['r']:.4e} unch={d['unchanged']:.4f} frozen={d['frozen']}")

    table = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    table += ["| " + " | ".join(r) + " |" for r in rows]
    print(head)
    print("\n".join(table))
    Path(__file__).with_name("results.md").write_text(
        "# Gradient norms at initialisation (mean +/- std, ddof=1, over 5 seeds)\n\n"
        f"`{head}`\n\n" + "\n".join(table) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
