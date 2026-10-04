"""Phase 2: gradient dynamics DURING training, PlainNet vs ResNet (MLP on CIFAR-10).

Model: flatten 3x32x32 -> stem Linear(3072, 256) -> Norm -> Act
       -> L hidden Linear layers of width 256 -> head Linear(256, 10).
  plain : L x [Linear -> Norm -> Act]
  resnet: L/2 x [s = x + F(x); x_next = Act(s)],  F = Linear->Norm->Act->Linear->Norm
Norm = BatchNorm1d or Identity; bias=False when BN on.
Metrics are measured on one fixed probe batch (512 train images) at epoch 0 (init)
and after every epoch, with a separate forward/backward (no optimizer step).
Run:  python experiment_training_dynamics.py            (full grid, CIFAR-10)
      python experiment_training_dynamics.py --sanity   (self-checks only)
      python experiment_training_dynamics.py --synthetic (random data, stated in header)
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
WIDTH, IN_DIM, N_CLASSES = 256, 3 * 32 * 32, 10
LR, BATCH, PROBE, SUBSET = 0.1, 128, 512, 10_000
EPS = torch.finfo(torch.float32).eps
ARCHS, DEPTHS, SEEDS = ["plain", "resnet"], [8, 16, 32], [42, 100, 2024]
ACTS = {"sigmoid": "sigmoid+xavier", "relu": "relu+kaiming"}
BNS = [False, True]
REPORT_EPOCHS = [0, 1, 5, 10]
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

torch.use_deterministic_algorithms(True)
torch.set_default_dtype(torch.float32)


class Tee:
    """Write stdout to the terminal and to a log file."""
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ------------------------------------------------------------------ data ----
def load_data(args):
    if args.synthetic:
        g = torch.Generator().manual_seed(0)
        xtr, ytr = torch.randn(1024, IN_DIM, generator=g), torch.randint(0, 10, (1024,), generator=g)
        xte, yte = torch.randn(1024, IN_DIM, generator=g), torch.randint(0, 10, (1024,), generator=g)
        return xtr, ytr, xte, yte, "SYNTHETIC (N=1024, x ~ N(0,I), random labels)"
    import torchvision  # only needed for the real dataset
    root = Path(args.data_dir).expanduser()
    tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
    te = torchvision.datasets.CIFAR10(root, train=False, download=True)

    def prep(ds):
        x = torch.tensor(ds.data, dtype=torch.float32).div_(255.0)   # N,32,32,3
        x = (x - torch.tensor(CIFAR_MEAN)) / torch.tensor(CIFAR_STD)  # per-channel
        return x.permute(0, 3, 1, 2).reshape(len(ds), -1).contiguous(), torch.tensor(ds.targets)

    xtr, ytr = prep(tr)
    xte, yte = prep(te)
    idx = torch.randperm(len(xtr), generator=torch.Generator().manual_seed(0))[:SUBSET]
    return xtr[idx], ytr[idx], xte, yte, f"CIFAR-10 (train subset {SUBSET} fixed by seed 0, full 10k test) at {root}"


# ----------------------------------------------------------------- model ----
class Net(nn.Module):
    def __init__(self, arch, L, act, bn):
        super().__init__()
        assert arch == "plain" or L % 2 == 0
        self.arch = arch
        norm = (lambda: nn.BatchNorm1d(WIDTH)) if bn else (lambda: nn.Identity())
        mk_act = (lambda: nn.Sigmoid()) if act == "sigmoid" else (lambda: nn.ReLU())
        lin = lambda i, o: nn.Linear(i, o, bias=not bn)
        self.stem = nn.Sequential(lin(IN_DIM, WIDTH), norm(), mk_act())
        if arch == "plain":
            self.body = nn.ModuleList(nn.Sequential(lin(WIDTH, WIDTH), norm(), mk_act())
                                      for _ in range(L))
        else:
            self.body = nn.ModuleList(nn.Sequential(lin(WIDTH, WIDTH), norm(), mk_act(),
                                                    lin(WIDTH, WIDTH), norm())
                                      for _ in range(L // 2))
            self.post = nn.ModuleList(mk_act() for _ in range(L // 2))
        self.head = nn.Linear(WIDTH, N_CLASSES, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if act == "sigmoid":
                    nn.init.xavier_normal_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def linears(self):
        """Weights in depth order: stem, hidden 1..L, head."""
        return [m for m in self.modules() if isinstance(m, nn.Linear)]

    def forward(self, x, record=None):
        x = self.stem(x)
        for k, blk in enumerate(self.body):
            if self.arch == "plain":
                x = blk(x)
                continue
            if record is not None:
                x.retain_grad()
            f = blk(x)
            s = x + f
            if record is not None:
                s.retain_grad()
                record.append((x, f, s))
            x = self.post[k](s)
        return self.head(x)


# ------------------------------------------------------------- measuring ----
def probe(model, xp, yp):
    """Separate forward/backward on the probe batch in train mode; BN buffers restored."""
    bufs = {k: v.clone() for k, v in model.named_buffers()}
    model.train()
    model.zero_grad(set_to_none=True)
    rec = [] if model.arch == "resnet" else None
    loss = nn.CrossEntropyLoss()(model(xp, rec), yp)
    loss.backward(retain_graph=rec is not None)
    W = model.linears()
    g = [torch.linalg.norm(m.weight.grad).item() for m in W]
    out = {"probe_loss": loss.item(), "gnorm": g,
           "log10_rho": math.log10(g[0] / g[-1]) if g[0] > 0 and g[-1] > 0 else float("-inf")}
    if rec is not None:
        # snapshot first: autograd.grad below also fires retain_grad hooks and
        # would add J_F^T dL/ds into x.grad
        grads = [(x.grad.clone(), s.grad.clone()) for x, _, s in rec]
        short, branch, recon = [], [], []
        for (x, f, s), (dLdx, dLds) in zip(rec, grads):
            (jtv,) = torch.autograd.grad(f, x, grad_outputs=dLds, retain_graph=True)
            short.append(torch.linalg.norm(dLds).item())
            branch.append(torch.linalg.norm(jtv).item())
            den = torch.linalg.norm(dLdx).item()
            err = torch.linalg.norm(dLds + jtv - dLdx).item()
            recon.append(err / den if den > 0 else (0.0 if err == 0 else float("inf")))
        out.update(shortcut=short, branch=branch, recon_rel_err=recon)
        assert max(recon) < 1e-4, f"shortcut + branch != dL/dx_l, rel err {max(recon):.3e}"
    model.zero_grad(set_to_none=True)
    with torch.no_grad():                        # restore BN running stats / counters
        for k, v in model.named_buffers():
            v.copy_(bufs[k])
    return out


@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    loss, correct = 0.0, 0
    for i in range(0, len(x), 1000):
        out = model(x[i:i + 1000])
        loss += nn.CrossEntropyLoss(reduction="sum")(out, y[i:i + 1000]).item()
        correct += (out.argmax(1) == y[i:i + 1000]).sum().item()
    return loss / len(x), correct / len(x)


def run(arch, L, act, bn, seed, data, epochs):
    xtr, ytr, xte, yte = data
    seed_all(seed)
    model = Net(arch, L, act, bn)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0, weight_decay=0)
    crit = nn.CrossEntropyLoss()
    xp, yp = xtr[:PROBE], ytr[:PROBE]            # fixed probe batch
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xtr, ytr), batch_size=BATCH, shuffle=True,
        num_workers=0, generator=torch.Generator().manual_seed(seed))
    W = model.linears()
    rec = {"arch": arch, "L": L, "act": act, "bn": bn, "seed": seed,
           "diverged": None, "epochs": []}

    def log_epoch(ep, r_steps, frozen, batch_losses):
        p = probe(model, xp, yp)
        trl, tra = evaluate(model, xtr, ytr)
        _, tea = evaluate(model, xte, yte)
        e = {"epoch": ep, **p, "train_loss_eval": trl, "train_acc": tra, "test_acc": tea,
             "train_loss_batches": float(np.mean(batch_losses)) if batch_losses else None,
             "median_r": [float(np.median(c)) for c in zip(*r_steps)] if r_steps else None,
             "stem_frozen_frac": float(np.mean(frozen)) if frozen else None}
        rec["epochs"].append(e)

    log_epoch(0, [], [], [])
    for ep in range(1, epochs + 1):
        model.train()
        r_steps, frozen, losses = [], [], []
        for step, (xb, yb) in enumerate(loader):
            before = [m.weight.detach().clone() for m in W]
            loss = crit(model(xb), yb)
            if not torch.isfinite(loss):
                rec["diverged"] = {"epoch": ep, "step": step}
                return rec
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            r_steps.append([(torch.linalg.norm(m.weight.detach() - b) / torch.linalg.norm(b)).item()
                            for m, b in zip(W, before)])
            frozen.append(torch.equal(W[0].weight.detach(), before[0]))
        log_epoch(ep, r_steps, frozen, losses)
    return rec


def all_finite(obj):
    if isinstance(obj, dict):
        return all(all_finite(v) for k, v in obj.items() if k != "diverged")
    if isinstance(obj, list):
        return all(all_finite(v) for v in obj)
    if isinstance(obj, float):
        return math.isfinite(obj)
    return True


# --------------------------------------------------------------- reports ----
def ms(vals):
    v = [x for x in vals if x is not None and math.isfinite(x)]
    if not v:
        return "n/a"
    if len(v) == 1:
        return f"{v[0]:.3g} (n=1)"
    a = np.array(v, dtype=np.float64)
    return f"{a.mean():.3g} ± {a.std(ddof=1):.3g}" + ("" if len(v) == len(vals) else f" (n={len(v)})")


def at_epoch(r, ep):
    return next((e for e in r["epochs"] if e["epoch"] == ep), None)


def write_md(runs, header, path):
    cfgs = sorted({(r["arch"], r["L"], r["act"], r["bn"]) for r in runs},
                  key=lambda c: (c[2], c[3], c[1], c[0]))
    metrics = [
        ("log10(rho) = log10(||g_stem|| / ||g_head||)", lambda e: e["log10_rho"]),
        ("stem ||g||_F", lambda e: e["gnorm"][0]),
        ("stem median r / eps32 (r = ||W_t+1 - W_t|| / ||W_t||, epoch median)",
         lambda e: e["median_r"][0] / EPS if e["median_r"] else None),
        ("stem fraction of steps with W unchanged (torch.equal)", lambda e: e["stem_frozen_frac"]),
        ("test accuracy", lambda e: e["test_acc"]),
    ]
    lines = [header, ""]
    for title, fn in metrics:
        lines += [f"## {title}", "",
                  "| arch | L | act/init | BN | " + " | ".join(f"epoch {e}" for e in REPORT_EPOCHS) + " |",
                  "|---|---|---|---|" + "---|" * len(REPORT_EPOCHS)]
        for c in cfgs:
            rs = [r for r in runs if (r["arch"], r["L"], r["act"], r["bn"]) == c]
            cells = []
            for ep in REPORT_EPOCHS:
                vals = [fn(e) if (e := at_epoch(r, ep)) else None for r in rs]
                cells.append(ms(vals) if ep > 0 or "r /" not in title and "fraction" not in title else "-")
            lines.append(f"| {c[0]} | {c[1]} | {ACTS[c[2]]} | {'on' if c[3] else 'off'} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## diverged runs", "", "| arch | L | act/init | BN | #diverged (of seeds) | where |", "|---|---|---|---|---|---|"]
    for c in cfgs:
        rs = [r for r in runs if (r["arch"], r["L"], r["act"], r["bn"]) == c]
        d = [r for r in rs if r["diverged"]]
        where = "; ".join(f"seed {r['seed']}: epoch {r['diverged']['epoch']} step {r['diverged']['step']}" for r in d) or "-"
        lines.append(f"| {c[0]} | {c[1]} | {ACTS[c[2]]} | {'on' if c[3] else 'off'} | {len(d)}/{len(rs)} | {where} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figures(runs, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir.mkdir(exist_ok=True)

    def seed_mean_heat(arch):
        rs = [r for r in runs if r["arch"] == arch and r["L"] == 32 and r["act"] == "sigmoid" and not r["bn"]]
        n_ep = max(len(r["epochs"]) for r in rs)
        M = np.full((len(rs), len(rs[0]["epochs"][0]["gnorm"]), n_ep), np.nan)
        for i, r in enumerate(rs):
            for e in r["epochs"]:
                g = np.array(e["gnorm"], dtype=np.float64)
                M[i, :, e["epoch"]] = np.log10(np.where(g > 0, g, np.nan))
        return np.nanmean(M, axis=0)

    H = {a: seed_mean_heat(a) for a in ARCHS}
    vmin = min(np.nanmin(h) for h in H.values())
    vmax = max(np.nanmax(h) for h in H.values())
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    for ax, a in zip(axes, ARCHS):
        im = ax.imshow(H[a], aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"{a}, L=32, sigmoid+xavier, BN off")
        ax.set_xlabel("epoch (0 = init)")
        ax.set_xticks(range(H[a].shape[1]))
    n_layers = H["plain"].shape[0]
    axes[0].set_ylabel("Linear layer (0 = stem, %d = head)" % (n_layers - 1))
    fig.colorbar(im, ax=axes, label="seed-mean log10 ||g^[l]||_F")
    fig.savefig(fig_dir / "heatmap_plain_vs_resnet.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for ax, act in zip(axes, ["relu", "sigmoid"]):
        for arch in ARCHS:
            for L in DEPTHS:
                rs = [r for r in runs if (r["arch"], r["L"], r["act"], r["bn"]) == (arch, L, act, False)]
                eps_, mu, sd = [], [], []
                for ep in range(0, 11):
                    v = [e["log10_rho"] for r in rs if (e := at_epoch(r, ep)) and math.isfinite(e["log10_rho"])]
                    if v:
                        eps_.append(ep)
                        mu.append(np.mean(v))
                        sd.append(np.std(v, ddof=1) if len(v) > 1 else 0.0)
                ax.errorbar(eps_, mu, yerr=sd, marker="o" if arch == "plain" else "s", ms=3, capsize=2,
                            ls="-" if arch == "plain" else "--", label=f"{arch} L={L}")
        ax.set_title(f"{ACTS[act]}, BN off")
        ax.set_xlabel("epoch (0 = init)")
        ax.set_ylabel("log10(rho), rho = ||g_stem|| / ||g_head||")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.savefig(fig_dir / "gradient_flow_residual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ main ----
def strip_time(r):
    return {k: v for k, v in r.items() if k != "run_time_s"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="~/.cache/cifar10")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--sanity", action="store_true", help="self-checks only, then exit")
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()
    sys.stdout = Tee(HERE / ("training_dynamics_sanity.log" if args.sanity else "training_dynamics.log"))

    t_start = time.time()
    *data, dataset = load_data(args)
    torch.set_num_threads(torch.get_num_threads())
    header = (f"# Phase 2: training dynamics, plain vs resnet\n\n"
              f"- torch {torch.__version__} | python {sys.version.split()[0]} | float32 CPU, "
              f"{torch.get_num_threads()} threads, deterministic algorithms on\n"
              f"- dataset: {dataset}\n- SGD lr={LR} momentum=0 wd=0 | batch {BATCH} | "
              f"{args.epochs} epochs | probe batch {PROBE} | width {WIDTH}\n"
              f"- seeds {SEEDS} | depths {DEPTHS} | eps32 = {EPS:.3g}\n"
              f"- tables: mean ± std (ddof=1) over seeds, 3 sig. figs; (n=k) = only k seeds finite/not diverged")
    print(header)

    # self-checks: same seed twice -> identical metrics; resnet decomposition inside probe()
    t0 = time.time()
    a = run("plain", 8, "relu", False, 42, data, 1)
    sanity_time = time.time() - t0
    b = run("plain", 8, "relu", False, 42, data, 1)
    assert a == b, "same seed twice gave different metrics"
    assert all_finite(a)
    c = run("resnet", 8, "relu", False, 42, data, 1)      # asserts recon_rel_err < 1e-4
    assert all_finite(c)
    print(f"self-checks passed | sanity run (plain L=8 relu BN off, 1 epoch) = {sanity_time:.1f}s | "
          f"max resnet recon rel err = {max(max(e['recon_rel_err']) for e in c['epochs']):.2e}")
    if args.sanity:
        return

    grid = [(arch, L, act, bn, s) for act in ACTS for bn in BNS for L in DEPTHS
            for arch in ARCHS for s in SEEDS]
    runs, t_grid = [], time.time()
    for k, (arch, L, act, bn, s) in enumerate(grid, 1):
        t0 = time.time()
        r = run(arch, L, act, bn, s, data, args.epochs)
        r["run_time_s"] = time.time() - t0
        if not r["diverged"]:
            assert all_finite(r), f"non-finite value in non-diverged run {arch} L={L} {act} bn={bn} seed={s}"
        runs.append(r)
        el = time.time() - t_grid
        eta = el / k * (len(grid) - k)
        last = r["epochs"][-1]
        status = (f"DIVERGED epoch {r['diverged']['epoch']} step {r['diverged']['step']}" if r["diverged"]
                  else f"test_acc={last['test_acc']:.3f} log10rho={last['log10_rho']:.2f}")
        print(f"[Run {k}/{len(grid)}] arch={arch} L={L} act={act} BN={'on' if bn else 'off'} seed={s} | "
              f"run_time={r['run_time_s']:.1f}s | elapsed={el / 60:.1f}min | ETA={eta / 60:.1f}min | {status}")

    runtime = time.time() - t_start
    header += f"\n- total runtime {runtime / 60:.1f} min (grid {(time.time() - t_grid) / 60:.1f} min)"
    (HERE / "results_training.json").write_text(json.dumps(
        {"header": header, "runs": runs}, indent=1, allow_nan=True), encoding="utf-8")
    write_md(runs, header, HERE / "results_training.md")
    make_figures(runs, HERE / "figures")
    print(f"done in {runtime / 60:.1f} min")


if __name__ == "__main__":
    main()
