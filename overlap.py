"""Find resized grayscale course images by content, keeping ambiguous IDs explicit."""
import argparse
import collections
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from data import ROOT, archive_batches, file_hash, json_write
from evaluate import write_csv
from pilot import historical_images


def signature(a):
    return torch.from_numpy(a).float().reshape(-1,7,4,7,4).mean((2,4)).flatten(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assignment", type=Path)
    args=parser.parse_args()
    torch.set_num_threads(4)
    test,test_y=archive_batches(["test_batch"])
    official=np.concatenate([np.load(ROOT/"images.npy"),test]).transpose(0,2,3,1)
    official_y=np.r_[np.load(ROOT/"labels.npy"),test_y]
    gray=[Image.fromarray(x).convert("L") for x in official]
    # Include a nearest-neighbor resize, interpolation, and a center crop.
    transforms={r.name:np.stack([np.asarray(x.resize((28,28),r)) for x in gray])
                for r in (Image.Resampling.BILINEAR,Image.Resampling.BICUBIC,Image.Resampling.NEAREST)}
    transforms["crop"]=np.stack([np.asarray(x)[2:30,2:30] for x in gray])
    sigs=[signature(t) for t in transforms.values()]
    train_set=set(np.load(ROOT/"train.npy").tolist())
    counts=collections.Counter()
    excluded=set()
    rows=[]
    for split in ("Train","Test"):
        names,old,labels=historical_images(args.assignment,split)
        sig=signature(old)
        for start in range(0,len(old),128):
            q=sig[start:start+128]
            # Search across labels and both official splits. Keep candidates from every transform.
            candidates=torch.cat([torch.cdist(q,s).topk(6,largest=False).indices for s in sigs],1).numpy()
            for j,options in enumerate(candidates):
                oi=start+j
                ids=np.unique(options)
                target=old[oi].astype(np.float32)
                residual=np.stack([np.sqrt(np.square(t[ids].astype(np.float32)-target).mean((1,2)))
                                   for t in transforms.values()])
                distances=residual.min(0)
                rank=distances.argsort()
                i=int(ids[rank[0]])
                rms=float(distances[rank[0]])
                second=float(distances[rank[1]])
                near=ids[distances<=max(2*rms,3.)]
                match=rms<=3 and (second>=max(2*rms,3.) or rms==0 and second>0)
                status="unique" if match else ("ambiguous" if rms<=3 else "unresolved")
                if status=="unique" and labels[oi]!=official_y[i]:
                    status="label_conflict"
                pool="train" if i in train_set else ("validation" if i<50000 else "test")
                if status=="unique":
                    if i>=50000: excluded.add(i-50000)
                elif status=="ambiguous":
                    excluded.update(int(k)-50000 for k in near if k>=50000)
                counts[(split,status,pool)]+=1
                rows.append(dict(course_id=names[oi],course_split=split,course_label=int(labels[oi]),
                         best_official_split=pool,best_official_id=i if i<50000 else i-50000,
                         official_label=int(official_y[i]),rmse=rms,runner_up_rmse=second,
                         transform=list(transforms)[residual[:,rank[0]].argmin()],status=status))
    path=Path("artifacts/overlap")
    path.mkdir(parents=True,exist_ok=True)
    write_csv(path/"matches.csv",rows)
    complete=not any(r["status"] in ("unresolved","label_conflict") for r in rows)
    if complete:
        np.save(path/"nonpilot.npy",np.array(sorted(set(range(10000))-excluded),dtype=np.int64))
    else:
        (path/"nonpilot.npy").unlink(missing_ok=True)
    summary=dict(counts=[dict(course_split=k[0],status=k[1],official_split=k[2],n=v)
                         for k,v in sorted(counts.items())],tested=len(rows),
                 excluded_test=len(excluded),complete=complete,max_best_rmse=max(r["rmse"] for r in rows),
                 table_sha256=file_hash(path/"matches.csv"))
    json_write("results/overlap.json",summary)
    print(summary)


if __name__=="__main__":
    main()
