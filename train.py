"""Epoch-boundary resumable, matched full-precision training."""
import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F

from data import ROOT, digest, file_hash, json_write
from model import LeNet, epoch_plan, shift

CONDITIONS = "ABCD"
FORMAL = Path("artifacts/runs")


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def setup(device):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Formal runs require CUDA")


def param_hash(net):
    h = hashlib.sha256()
    for name, p in net.named_parameters():
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def better(correct, best):
    return correct > best


def tensors(device, smoke=False):
    info = json.loads(Path("results/split.json").read_text())
    x = np.load(ROOT / "images.npy", mmap_mode="r")
    y = np.load(ROOT / "labels.npy")
    mean = torch.tensor(info["mean"], device=device)[None, :, None, None]
    std = torch.tensor(info["std"], device=device)[None, :, None, None]
    parts = []
    for name in ("train", "val"):
        ids = np.load(ROOT / (name + ".npy"))
        assert digest(ids) == info[name + "_sha256"]
        if smoke:
            ids = ids[:191]
        images = torch.from_numpy(x[ids]).to(device).float().div_(255).sub_(mean).div_(std)
        parts.extend([images.contiguous(memory_format=torch.channels_last), torch.from_numpy(y[ids]).to(device)])
    return parts


class BatchStep:
    """Capture forward/backward for full GPU batches; SGD stays outside the graph."""
    def __init__(self, model, opt, x, y):
        self.model, self.opt = model, opt
        self.graph = None
        if x.is_cuda:
            self.x, self.y = x[:64].clone(), y[:64].clone()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    opt.zero_grad(set_to_none=True)
                    self.forward_backward(self.x, self.y)
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            opt.zero_grad(set_to_none=True)
            with torch.cuda.graph(self.graph):
                self.scalars = self.forward_backward(self.x, self.y)
            self.graph_grads = [p.grad for p in model.parameters()]

    def forward_backward(self, x, y):
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        return torch.stack((loss.detach() * len(y), (logits.detach().argmax(1) == y).sum()))

    def __call__(self, x, y):
        if self.graph is not None and len(y) == 64:
            self.x.copy_(x)
            self.y.copy_(y)
            for p, g in zip(self.model.parameters(), self.graph_grads):
                p.grad = g
            self.graph.replay()
            sums = self.scalars
        else:
            self.opt.zero_grad(set_to_none=True)
            sums = self.forward_backward(x, y)
        self.opt.step()
        return sums


@torch.no_grad()
def validate(net, x, y):
    logits = torch.cat([net(batch) for batch in x.split(256)])
    return F.cross_entropy(logits, y).item(), int((logits.argmax(1) == y).sum().item())


def save(path, obj):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def rng_state(device):
    return dict(torch=torch.get_rng_state(), python=random.getstate(), numpy=np.random.get_state(),
                cuda=torch.cuda.get_rng_state() if device == "cuda" else None)


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


def run(condition, seed, data, root, epochs, device, stop_after=None):
    x, y, vx, vy = data
    out = root / f"{condition}{seed}"
    out.mkdir(exist_ok=True, parents=True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = LeNet(condition in "CD")
    init = param_hash(model)
    model.to(device=device, memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=.02, momentum=.9, weight_decay=.0005,
                          fused=(device == "cuda"))
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[25, 40], gamma=.2)
    source = {f: file_hash(f) for f in ("model.py", "train.py", "data.py")}
    cfg = dict(condition=condition, seed=seed, epochs=epochs, device=device, batch=64,
               n_train=len(y), n_val=len(vy), init_hash=init, split=file_hash("results/split.json"),
               source=source, precision="FP32; TF32 disabled", momentum=.9, weight_decay=.0005,
               lr=.02, milestones=[25,40], gamma=.2, loss="cross_entropy", workers=0,
               optimizer="SGD", selection="highest clean validation correct count; earliest tie",
               python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name() if device == "cuda" else None)
    if (out / "config.json").exists():
        saved = json.loads((out / "config.json").read_text())
        if cfg != saved["config"]:
            raise ValueError(f"Configuration changed for {out}")
    else:
        json_write(out / "config.json", dict(config=cfg, git=git("rev-parse","HEAD"),
                                             workspace=git("status","--porcelain")))
    if (out / "complete.json").exists():
        record = json.loads((out / "complete.json").read_text())
        for f, h in record["sha256"].items():
            assert file_hash(out / f) == h
        return record
    best, history, best_weights = -1, [], None
    state = None
    if (out / "last.pt").exists():
        state = torch.load(out / "last.pt", weights_only=False, map_location="cpu")
        assert state["config"] == cfg
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        history, best, best_weights = state["history"], state["best"], state["best_weights"]
    step = BatchStep(model, opt, x, y)
    if state:
        restore_rng(state["rng"])
    for epoch in range(len(history)+1, epochs+1):
        start = time.perf_counter()
        model.train()
        order, offsets = epoch_plan(seed, epoch, len(y))
        ix = torch.from_numpy(order).to(device)
        epoch_x, epoch_y = x[ix], y[ix]
        if condition in "BD":
            off = torch.from_numpy(offsets[order]).to(device)
            for a, d in zip(epoch_x.split(1024), off.split(1024)):
                a.copy_(shift(a, d[:, 0], d[:, 1]))
        sums = torch.zeros(2, device=device)
        lr = opt.param_groups[0]["lr"]
        for xb, yb in zip(epoch_x.split(64), epoch_y.split(64)):
            sums.add_(step(xb, yb))
        loss, correct = sums.tolist()
        model.eval()
        val_loss, val_correct = validate(model, vx, vy)
        sched.step()
        if better(val_correct, best):
            best = val_correct
            best_weights = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        row = dict(epoch=epoch, lr=lr, train_loss=loss/len(y), train_accuracy=correct/len(y),
                   val_loss=val_loss, val_accuracy=val_correct/len(vy), val_correct=val_correct,
                   seconds=time.perf_counter()-start, order_hash=digest(order),
                   offset_hash=digest(offsets) if condition in "BD" else "")
        history.append(row)
        save(out/"last.pt", dict(config=cfg, epoch=epoch, model=model.state_dict(),
             optimizer=opt.state_dict(), scheduler=sched.state_dict(), rng=rng_state(device),
             history=history, best=best, best_weights=best_weights))
        # last.pt is authoritative if interrupted between these derivative file writes.
        log_history(out, history)
        if epoch == 1 or epoch % 10 == 0:
            print(out.name, epoch, f"val={val_correct/len(vy):.4f} seconds={row['seconds']:.2f}", flush=True)
        if stop_after == epoch and epoch < epochs:
            return
    log_history(out, history)
    best_epoch = next(r["epoch"] for r in history if r["val_correct"] == best)
    save(out/"best.pt", dict(model=best_weights, epoch=best_epoch, val_correct=best, config=cfg))
    record = dict(condition=condition, seed=seed, epochs=len(history), best_epoch=best_epoch,
                  best_val=best/len(vy), init_hash=init, seconds=sum(r["seconds"] for r in history),
                  sha256={f:file_hash(out/f) for f in ("best.pt","last.pt","history.csv","config.json")})
    json_write(out/"complete.json", record)
    return record


def log_history(out, history):
    with (out/"history.csv.tmp").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0])
        writer.writeheader()
        writer.writerows(history)
    (out/"history.csv.tmp").replace(out/"history.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["smoke","timing","all","single"])
    parser.add_argument("--condition", choices=list(CONDITIONS), default="A")
    parser.add_argument("--seed", choices=range(3), type=int, default=0)
    a = parser.parse_args()
    setup("cuda")
    data = tensors("cuda", a.mode == "smoke")
    root = FORMAL if a.mode in ("all","single") else Path("artifacts") / a.mode
    epochs = 60 if root == FORMAL else 1
    pairs = [(c,s) for s in range(3) for c in CONDITIONS] if a.mode == "all" else [(a.condition,a.seed)]
    for c,s in pairs:
        json_write("artifacts/current.json",dict(mode=a.mode,condition=c,seed=s,root=str(root)))
        run(c,s,data,root,epochs,"cuda")
    json_write("artifacts/current.json",dict(mode=a.mode,status="finished"))


if __name__ == "__main__":
    main()
