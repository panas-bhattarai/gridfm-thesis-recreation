"""Training loop for GridFM pre-training and fine-tuning.
"""
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from . import dataset as ds
from .model import gridfm_loss


def make_loaders(proc_dir, grids, batch_size=64, seed=42, transform=None):
    """transform (v0.2 addition)"""
    train, val = [], []
    for g in grids:
        train += ds.load_processed(proc_dir, g, "train")
        val += ds.load_processed(proc_dir, g, "val")
    if transform is not None:
        train, val = transform(train), transform(val)
    gen = torch.Generator().manual_seed(seed)
    return (DataLoader(train, batch_size=batch_size, shuffle=True, generator=gen),
            DataLoader(val, batch_size=batch_size, shuffle=False))


def run_epoch(model, loader, device, opt=None, clip=1.0, mask_fn=None, mask_seed=None):
    """One pass over the loader. Training pass if opt is given, else evaluation."""
    training = opt is not None
    model.train() if training else model.eval()
    gen = torch.Generator().manual_seed(mask_seed) if mask_seed is not None else None
    if mask_fn is None:
        mask_fn = lambda x: ds.random_mask(x, generator=gen)

    sums, n = {"total": 0.0, "mse": 0.0, "pbe": 0.0}, 0
    for b in loader:
        # mask on CPU before moving: keeps the v0.1 mask stream bit-identical
        # and avoids CPU-generator/CUDA-tensor mismatches on GPU runs
        xm, m = mask_fn(b.x)
        b, xm, m = b.to(device), xm.to(device), m.to(device)
        with torch.set_grad_enabled(training):
            # batch marks graph boundaries; v0.2's global attention needs it,
            # v0.1 ignores it (single graphs outside a loader have none)
            pred = model(xm, b.edge_index, b.edge_attr,
                         batch=getattr(b, "batch", None))
            loss, parts = gridfm_loss(pred, b.x, m, b.ybus_index, b.ybus_g, b.ybus_b)
        if training:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        bs = b.num_graphs
        for k in sums:
            sums[k] += parts[k] * bs
        n += bs
    return {k: v / n for k, v in sums.items()}


def fit(model, train_loader, val_loader, epochs, ckpt_dir, device="cpu",
        lr=3e-4, clip=1.0, sched_factor=0.7, sched_patience=10,
        resume=True, log_every=10, mask_fn=None, val_mask_fn=None, tag="pretrain"):
    """Train with checkpoint/resume. Returns the history DataFrame.
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path = ckpt_dir / ("%s_last.pt" % tag)
    best_path = ckpt_dir / ("final_%s_best.pt" % tag)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=sched_factor, patience=sched_patience)

    history, start_epoch, best_val = [], 0, float("inf")
    if resume and last_path.exists():
        state = torch.load(last_path, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        history = state["history"]
        start_epoch = state["epoch"] + 1
        best_val = state["best_val"]
        print("resumed from epoch %d (best val %.4f)" % (start_epoch, best_val))

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        tr = run_epoch(model, train_loader, device, opt=opt, clip=clip, mask_fn=mask_fn)
        va = run_epoch(model, val_loader, device, mask_seed=1234, mask_fn=val_mask_fn)
        sched.step(va["total"])
        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"],
               "time_s": time.time() - t0}
        row.update({("train_%s" % k): v for k, v in tr.items()})
        row.update({("val_%s" % k): v for k, v in va.items()})
        history.append(row)

        if va["total"] < best_val:
            best_val = va["total"]
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val": va}, best_path)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch,
                    "history": history, "best_val": best_val}, last_path)

        if epoch % log_every == 0 or epoch == epochs - 1:
            print("epoch %3d | train %.4f (mse %.4f pbe %.4f) | val %.4f | lr %.1e | %.0fs"
                  % (epoch, tr["total"], tr["mse"], tr["pbe"], va["total"],
                     row["lr"], row["time_s"]))
    return pd.DataFrame(history)
