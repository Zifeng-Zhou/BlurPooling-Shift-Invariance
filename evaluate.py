"""Frozen four-direction evaluation, common validation-selected checkpoints."""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from data import archive_batches, file_hash, json_write
from model import LeNet, metrics, shift
from train import CONDITIONS, FORMAL, setup

DIRECTIONS = [(-1,0),(1,0),(0,-1),(0,1)]


@torch.inference_mode()
def predict(net, images, border, magnitude):
    clean, views = [], []
    for batch in images.split(256):
        clean.append(net(batch).argmax(1).cpu().numpy())
        predictions = [net(shift(batch, dx*magnitude, dy*magnitude, border)).argmax(1)
                       for dx,dy in DIRECTIONS]
        views.append(torch.stack(predictions,dim=1).cpu().numpy())
    return np.concatenate(clean), np.concatenate(views)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main():
    setup("cuda")
    runs = [(s,c,FORMAL/f"{c}{s}") for s in range(3) for c in CONDITIONS]
    for s,c,path in runs:
        record = json.loads((path/"complete.json").read_text())
        assert record["epochs"] == 60 and record["sha256"]["best.pt"] == file_hash(path/"best.pt")
    info = json.loads(Path("results/split.json").read_text())
    raw, y = archive_batches(["test_batch"])
    assert raw.shape == (10000,3,32,32) and np.all(np.bincount(y)==1000)
    x = torch.from_numpy(raw).cuda().float().div_(255)
    x.sub_(torch.tensor(info["mean"]).cuda()[None,:,None,None]).div_(torch.tensor(info["std"]).cuda()[None,:,None,None])
    x = x.contiguous(memory_format=torch.channels_last)
    dst = Path("artifacts/predictions")
    dst.mkdir(exist_ok=True, parents=True)
    subsets = {"full":np.arange(len(y))}
    nonpilot = Path("artifacts/overlap/nonpilot.npy")
    published_ids = Path("results/nonpilot_ids.txt")
    if nonpilot.exists():
        subsets["nonpilot"] = np.load(nonpilot)
        if published_ids.exists():
            np.testing.assert_array_equal(subsets["nonpilot"], np.loadtxt(published_ids,dtype=np.int64))
    elif published_ids.exists():
        subsets["nonpilot"] = np.loadtxt(published_ids,dtype=np.int64)
    if "nonpilot" in subsets:
        ids = subsets["nonpilot"]
        assert len(np.unique(ids)) == len(ids) and np.all((ids >= 0) & (ids < len(y)))
    rows, manifest = [], []
    for seed,c,path in runs:
        net = LeNet(c in "CD").to(device="cuda",memory_format=torch.channels_last).eval()
        net.load_state_dict(torch.load(path/"best.pt",weights_only=True)["model"])
        for border in ("reflect","mean"):
            for m in (1,2,4):
                clean, views = predict(net,x,border,m)
                p = dst/f"{c}{seed}_{border}{m}.npz"
                np.savez_compressed(p, id=np.arange(len(y)),truth=y.astype("int8"),
                                    clean=clean.astype("int8"),views=views.astype("int8"),
                                    directions=np.array(["left","right","up","down"]))
                manifest.append(dict(path=p.as_posix(),sha256=file_hash(p)))
                for subset,ids in subsets.items():
                    rows.append(dict(seed=seed, condition=c,border=border,pixels=m,subset=subset,n=len(ids),
                                     **metrics(y[ids],clean[ids],views[ids])))
    write_csv("results/evaluation.csv",rows)
    json_write("results/predictions_manifest.json",manifest)


if __name__ == "__main__":
    main()
