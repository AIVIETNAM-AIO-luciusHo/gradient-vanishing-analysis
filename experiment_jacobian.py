"""Dot 2 experiment: spectrum of the end-to-end Jacobian J = dz[L-1]/dz[1] at initialisation.

Same network/init/data as experiment.py (imported, not copied): L linear layers, L-1 hidden of
width 64 + a 1-d output layer, biases = 0.  A: Sigmoid + xavier_normal_, B: ReLU + kaiming_normal_.
The 1-d output layer is dropped from J (a 1 x 64 Jacobian has rank 1, so its spectrum is
meaningless):   J = W[L-1] D[L-2] ... W[2] D[1],   D[k] = diag(sigma'(z[k])),   J in R^{64x64}.
The network is INITIALISED in float32 (weights identical to results.md), then cast to float64
for the spectrum only.
Run: python experiment_jacobian.py   (writes results_jacobian.md next to this file)
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.autograd.functional import jacobian

from experiment import CONDS, D, DEPTHS, N, SEEDS, build, seed_all  # experiment.main() is guarded

NS = 16                                   # first 16 samples of x
EPS64 = torch.finfo(torch.float64).eps    # 2.2e-16
HERE = Path(__file__).parent


def act_fn(act):
    return torch.sigmoid if act == "sigmoid" else torch.relu


def dact(act, z):
    if act == "sigmoid":
        s = torch.sigmoid(z)
        return s * (1 - s)
    return (z > 0).to(z.dtype)


def forward_hidden(lins, act, z1):
    """z[1] -> z[L-1] (pre-activations), through the hidden stack only."""
    a = act_fn(act)
    z = z1
    for W in lins[1:-1]:                  # W[2] .. W[L-1]
        z = W(a(z))
    return z


def make(seed, L, act):
    """float32 init as in experiment.run, then float64 copy for the spectrum."""
    seed_all(seed)
    x, _ = torch.randn(N, D), torch.randn(N, 1)
    net = build(L, act)
    lins = [m for m in net if isinstance(m, torch.nn.Linear)]
    # c_ave exactly as experiment.run (float32): max sigma' * mean_k ||W[k]||_2, k = 2..L
    smax32 = np.mean([torch.linalg.matrix_norm(m.weight.detach(), ord=2).item() for m in lins[1:]])
    c_ave = (0.25 if act == "sigmoid" else 1.0) * smax32
    net = net.double()
    lins = [m for m in net if isinstance(m, torch.nn.Linear)]
    return lins, x[:NS].double(), c_ave


def sample_stats(lins, act, xi, L):
    """Spectrum of J at one sample, plus the conservative bound c_J (measured sigma')."""
    z1 = lins[0](xi)
    J = jacobian(lambda z: forward_hidden(lins, act, z), z1)
    sv = torch.linalg.svdvals(J)
    smax, smin = sv[0].item(), sv[-1].item()
    thr = D * EPS64 * smax
    rank = int((sv > thr).sum().item())
    # c_J = prod_{k=2}^{L-1} ||W[k]||_2 * max_i |sigma'(z[k-1]_i)|, measured sigma'
    cJ, z = 1.0, z1
    for k in range(2, L):                 # k = 2..L-1; lins[k-1] = W[k]
        with torch.no_grad():
            dmax = dact(act, z).abs().max().item()
            cJ *= torch.linalg.matrix_norm(lins[k - 1].weight, ord=2).item() * dmax
            z = lins[k - 1](act_fn(act)(z))
    out = {"smax": smax, "rms": (sv.pow(2).mean().sqrt()).item(), "rank": rank,
           "full": rank == D and smin > thr, "cJ": cJ, "J": J,
           "kappa": smax / smin if rank == D and smin > thr else None}
    return out


def run(seed, L, act):
    lins, X, c_ave = make(seed, L, act)
    S = [sample_stats(lins, act, X[i], L) for i in range(NS)]
    for s in S:
        assert s["smax"] <= s["cJ"] * (1 + 1e-9), (seed, L, act, s["smax"], s["cJ"])   # B4(a)
    full = [s for s in S if s["kappa"] is not None]
    return {"log_smax": float(np.mean([np.log10(s["smax"]) for s in S])),
            "log_rms": float(np.mean([np.log10(s["rms"]) for s in S])),
            "rank": float(np.mean([s["rank"] for s in S])),
            "n_full": len(full),
            "log_kappa": float(np.mean([np.log10(s["kappa"]) for s in full])) if full else None,
            "log_cJ": float(np.mean([np.log10(s["cJ"]) for s in S])),
            "c_ave": c_ave}


def check_vjp():
    """B4(b): autograd d(v^T z[L-1])/dz[1] == J^T v  (delta[1] = J^T delta[L-1])."""
    g = torch.Generator().manual_seed(0)
    v = torch.randn(D, generator=g, dtype=torch.float64)
    for act, (seed, L) in (("sigmoid", (42, 10)), ("relu", (42, 10))):
        lins, X, _ = make(seed, L, act)
        z1 = lins[0](X[0]).detach().requires_grad_(True)
        (v @ forward_hidden(lins, act, z1)).backward()
        J = jacobian(lambda z: forward_hidden(lins, act, z), z1.detach())
        Jtv = J.T @ v
        # explicit product D1 W2^T D2 ... D_{L-2} W_{L-1}^T v, as in the proposition
        z, zs = z1.detach(), []
        for k in range(2, L):
            zs.append(z)
            z = lins[k - 1](act_fn(act)(z))
        d = v
        for k in range(L - 1, 1, -1):     # k = L-1 .. 2
            d = dact(act, zs[k - 2]) * (lins[k - 1].weight.T @ d)
        atol = 1e-12 * Jtv.abs().max().item()
        assert torch.allclose(z1.grad, Jtv, rtol=1e-9, atol=atol), act
        assert torch.allclose(d, Jtv, rtol=1e-9, atol=atol), act
        print(f"  B4b OK: {act} seed={seed} L={L}: max|grad - J^T v|={ (z1.grad - Jtv).abs().max():.3e}, "
              f"max|D1 W2^T ... v - J^T v|={(d - Jtv).abs().max():.3e}")


def ms(vals):
    a = np.array(vals, dtype=np.float64)
    return f"{a.mean():.3g} ± {a.std(ddof=1):.3g}"


def main():
    # B4(c): determinism
    for act in ("sigmoid", "relu"):
        assert run(42, 5, act) == run(42, 5, act), act
    print("B4c OK: identical results on repeated run")
    check_vjp()

    # B4(d): reproduce the 'c (bound)' column of results.md
    ref = {}
    for line in (HERE / "results.md").read_text(encoding="utf-8").splitlines():
        p = [t.strip() for t in line.strip("|").split("|")]
        if len(p) > 3 and p[0][:2] in ("A:", "B:"):
            ref[(p[0], int(p[1]))] = p[2]

    head = (f"torch {torch.__version__} | python {sys.version.split()[0]} | init float32 CPU -> spectrum float64 | "
            f"seeds={SEEDS} | D={D} N={N} | samples per seed = {NS} (first {NS} rows of x)")
    cols = ["condition", "L", "log10 sigma_max(J)", "log10 RMS(sigma)", "numerical rank (of 64)",
            "full-rank pairs (of 80)", "log10 kappa(J)", "log10 c_J", "c_ave (reproduces results.md)"]
    rows = []
    for name, act in CONDS.items():
        for L in DEPTHS:
            res = [run(s, L, act) for s in SEEDS]
            col = lambda k: [d[k] for d in res]
            nfull = sum(col("n_full"))
            ks = [d["log_kappa"] for d in res if d["log_kappa"] is not None]
            if nfull == len(SEEDS) * NS:
                kap = ms(ks)
            elif nfull == 0:
                kap = "n/a (rank-deficient or sigma_min < 64*eps64*sigma_max)"
            else:
                kap = (ms(ks) if len(ks) > 1 else f"{ks[0]:.3g}") + f" (only {nfull}/{len(SEEDS) * NS} full-rank pairs)"
            cave = ms(col("c_ave"))
            assert cave == ref[(name, L)], (name, L, cave, ref[(name, L)])   # B4(d)
            rows.append([name, str(L), ms(col("log_smax")), ms(col("log_rms")), ms(col("rank")),
                         f"{nfull}/{len(SEEDS) * NS}", kap, ms(col("log_cJ")), cave])
            for s, d in zip(SEEDS, res):
                print(f"  seed={s:5d} L={L:2d} {act:7s} log10smax={d['log_smax']:+.4f} log10rms={d['log_rms']:+.4f} "
                      f"rank={d['rank']:.2f} full={d['n_full']:2d}/{NS} "
                      f"log10kappa={'n/a' if d['log_kappa'] is None else format(d['log_kappa'], '+.4f')} "
                      f"log10cJ={d['log_cJ']:+.4f} c_ave={d['c_ave']:.4f}")
    print("B4a OK: sigma_max(J) <= c_J (1 + 1e-9) for every sample, seed, L, condition")
    print("B4d OK: c_ave matches results.md 'c (bound)' column in all 6 rows")

    table = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    table += ["| " + " | ".join(r) + " |" for r in rows]
    print(head)
    print("\n".join(table))
    (HERE / "results_jacobian.md").write_text(
        "# End-to-end Jacobian spectrum at initialisation (mean +/- std, ddof=1, over 5 seeds)\n\n"
        f"`{head}`\n\n"
        "Jacobian trên chồng tầng ẩn 64x64 (z^[1] -> z^[L-1]); tầng ra 1 chiều bị loại.\n"
        "J = W^[L-1] D^[L-2] ... W^[2] D^[1], L-2 thừa số. Mỗi seed: trung bình của log10 trên 16 mẫu; "
        "bảng: trung bình ± độ lệch chuẩn qua 5 seed.\n"
        "Hạng số học = số sigma_i > 64*eps64*sigma_max. log10 kappa chỉ tính khi hạng = 64 "
        "(mọi sigma_i trên ngưỡng); nếu không: n/a.\n\n" + "\n".join(table) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
