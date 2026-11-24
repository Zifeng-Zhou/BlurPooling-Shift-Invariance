"""Verify the 12 matched runs, archive logs and calculate paired percentage points."""
import csv
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import ROOT, file_hash, json_write
from evaluate import write_csv
from train import CONDITIONS, FORMAL

KEYS=["flip_rate","clean_accuracy","shifted_accuracy"]
FIG=Path("report/figures")
COLORS=["#34475a","#14657d","#a77835","#ce592f"]
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                     "axes.labelcolor":"#203342","text.color":"#203342","pdf.fonttype":42})


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def verify_runs():
    records=[]
    for seed in range(3):
        histories={}
        starts=[]
        for c in CONDITIONS:
            p=FORMAL/f"{c}{seed}"
            r=json.loads((p/"complete.json").read_text())
            for f,h in r["sha256"].items():
                assert file_hash(p/f)==h
            log=read_csv(p/"history.csv")
            assert [int(l["epoch"]) for l in log]==list(range(1,61))
            np.testing.assert_allclose([float(l["lr"]) for l in log],[.02]*25+[.004]*15+[.0008]*20)
            assert r["best_epoch"]==int(max(log,key=lambda l:int(l["val_correct"]))["epoch"])
            histories[c]=log
            starts.append(r["init_hash"])
            dst=Path("results/runs")/p.name
            dst.mkdir(parents=True,exist_ok=True)
            for name in ["history.csv","config.json","complete.json"]:
                shutil.copyfile(p/name,dst/name)
            records.append(r)
        assert len(set(starts))==1
        for epoch in range(60):
            assert len({histories[c][epoch]["order_hash"] for c in CONDITIONS})==1
            assert histories["B"][epoch]["offset_hash"]==histories["D"][epoch]["offset_hash"]
    json_write("results/run_manifest.json",records)


def savefig(fig, name):
    FIG.mkdir(parents=True,exist_ok=True)
    for typ in ("pdf","png"):
        fig.savefig(FIG/f"{name}.{typ}",dpi=180,bbox_inches="tight",pad_inches=.04)
    plt.close(fig)


def context_figures():
    x=np.load(ROOT/"images.npy",mmap_mode="r")
    y=np.load(ROOT/"labels.npy")
    tr=np.load(ROOT/"train.npy")
    i=next(i for i in tr if y[i]==9)
    raw=x[i].transpose(1,2,0)
    shifted=np.pad(raw,((0,0),(1,0),(0,0)),mode="reflect")[:,:32]
    diff=np.abs(raw.astype(float)-shifted).mean(-1)
    fig,ax=plt.subplots(1,3,figsize=(4.45,1.6))
    for a,img,t in zip(ax,[raw,shifted,diff],["Original truck","Right 1 pixel","Pixel change"]):
        a.imshow(img,interpolation="nearest",cmap="magma",vmin=0,vmax=100)
        a.set_title(t,fontsize=9)
        a.axis("off")
    fig.tight_layout()
    savefig(fig,"translation")
    fig,ax=plt.subplots(figsize=(4.35,1.55))
    ax.axis("off")
    table=ax.table(cellText=[["Standard max","A","B"],["Sliding max + blur","C","D"]],
                   colLabels=["Pooling rule","No shift","Random shift"],colWidths=[.43,.23,.34],
                   cellLoc="center",loc="center")
    table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.7)
    for (row,col),cell in table.get_celld().items():
        cell.set_linewidth(.5);cell.set_edgecolor("#c5d3da")
        cell.set_facecolor("#164257" if row==0 else ("#f1f5f6" if col==0 else "white"))
        if row==0:cell.set_text_props(color="white",weight="bold")
        if row==2 and col==2:cell.set_text_props(color=COLORS[3],weight="bold")
    savefig(fig,"design")


def main():
    verify_runs()
    raw=read_csv("results/evaluation.csv")
    lookup={(r["subset"],r["border"],int(r["pixels"]),int(r["seed"]),r["condition"]):r for r in raw}
    subsets=sorted({r["subset"] for r in raw})
    pairs=[]; means=[]
    for subset,b,m,comp in itertools.product(subsets,["reflect","mean"],[1,2,4],["D-B","C-A"]):
        diffs=[]
        for seed in range(3):
            high,low=[lookup[subset,b,m,seed,c] for c in comp.split("-")]
            delta=[100*(float(high[k])-float(low[k])) for k in KEYS]
            diffs.append(delta)
            pairs.append(dict(subset=subset,border=b,pixels=m,comparison=comp,seed=seed,
                              **dict(zip(KEYS,delta))))
        means.append(dict(subset=subset,border=b,pixels=m,comparison=comp,
                          **dict(zip(KEYS,np.mean(diffs,axis=0))),
                          **{k+"_sd":v for k,v in zip(KEYS,np.std(diffs,axis=0,ddof=1))}))
    write_csv("results/paired_pp.csv",pairs)
    write_csv("results/paired_mean_pp.csv",means)
    prim=[r for r in pairs if (r["subset"],r["border"],r["pixels"],r["comparison"])==("full","reflect",1,"D-B")]
    d=np.array([[r[k] for k in KEYS] for r in prim])
    json_write("results/decision.json",dict(mean_pp=dict(zip(KEYS,d.mean(0))),all_flip_lower=bool((d[:,0]<0).all()),
           passes=bool((d[:,0]<0).all() and (d.mean(0)[1:]>=-1).all())))
    context_figures()
    fig,axes=plt.subplots(1,3,figsize=(7,2.55),layout="constrained")
    for ax,k,title in zip(axes,KEYS,["Prediction flips","Clean accuracy","Shifted accuracy"]):
        for s,marker in enumerate(["o","s","^"]):
            vals=[100*float(lookup["full","reflect",1,s,c][k]) for c in "BD"]
            ax.plot([0,1],vals,marker=marker,linewidth=1.4,color=["#22647b","#bb5a35","#66568b"][s],
                    label=f"Seed {s}",markersize=5)
        ax.set(xticks=[0,1],xticklabels=["Standard","Blur"],xlim=(-.2,1.2),title=title,ylabel="Percent")
        ax.grid(axis="y",alpha=.2)
        ax.ticklabel_format(axis="y",useOffset=False)
    axes[0].legend(fontsize=7,frameon=False)
    savefig(fig,"paired")
    fig,axes=plt.subplots(1,2,figsize=(4.85,2.15),layout="constrained")
    for ax,k,title in zip(axes,["flip_rate","shifted_accuracy"],["Flip rate","Shifted accuracy"]):
        for b,color in zip(["reflect","mean"],["#186b82","#c05e35"]):
            rows=[r for r in means if r["subset"]=="full" and r["border"]==b and r["comparison"]=="D-B"]
            ax.plot([r["pixels"] for r in rows],[r[k] for r in rows],"o-",color=color,label=b.capitalize())
        ax.axhline(0,color="gray",lw=.7);ax.grid(axis="y",alpha=.2)
        ax.set(xticks=[1,2,4],xlabel="Translation (pixels)",ylabel="D minus B (pp)",title=title)
    axes[0].legend(frameon=False,fontsize=8)
    savefig(fig,"borders")


if __name__=="__main__":
    main()
