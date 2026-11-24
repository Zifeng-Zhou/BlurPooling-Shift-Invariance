"""Descriptive supplementary diagnostics from saved seed-zero predictions."""
import numpy as np
import matplotlib.pyplot as plt

from analyze import savefig
from data import archive_batches,json_write

NAMES="airplane automobile bird cat deer dog frog horse ship truck".split()


def main():
    raw,truth=archive_batches(["test_batch"])
    b,d=[np.load(f"artifacts/predictions/{c}0_reflect1.npz") for c in "BD"]
    both=(b["clean"]==truth)&(d["clean"]==truth)
    correct_b=b["views"][:,1]==truth
    correct_d=d["views"][:,1]==truth
    ids=[int(np.flatnonzero(both&(~correct_b)&correct_d)[0]),int(np.flatnonzero(both&(~correct_d))[0])]
    fig,ax=plt.subplots(2,2,figsize=(3.8,3.2))
    rows=[]
    for i,(idx,title) in enumerate(zip(ids,["Blur retains the class","Blur also changes class"])):
        clean=raw[idx].transpose(1,2,0)
        moved=np.pad(clean,((0,0),(1,0),(0,0)),mode="reflect")[:,:32]
        lab=NAMES[truth[idx]]
        cb=NAMES[b["views"][idx,1]];cd=NAMES[d["views"][idx,1]]
        for a,img,cap in zip(ax[i],[clean,moved],[f"Clean: {lab}",f"Right: B {cb}, D {cd}"]):
            a.imshow(img,interpolation="nearest");a.axis("off");a.set_title(cap,fontsize=8)
        rows.append(dict(id=idx,truth=lab,clean_B=lab,clean_D=lab,right_B=cb,right_D=cd))
    fig.tight_layout()
    savefig(fig,"examples")
    json_write("results/examples.json",rows)


if __name__=="__main__":
    main()
