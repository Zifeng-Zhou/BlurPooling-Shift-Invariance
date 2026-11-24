"""Re-evaluate the original course grayscale model on its historical test images."""
import argparse
import io
from pathlib import Path
import zipfile
import numpy as np
import torch
from PIL import Image

from data import file_hash,json_write
from evaluate import predict,write_csv
from model import LeNet,metrics


def historical_images(assignment, split):
    with zipfile.ZipFile(assignment/"CIFAR10-1.zip") as z:
        names = sorted(n for n in z.namelist() if f"/{split}/" in n and n.endswith(".png"))
        images = np.stack([np.asarray(Image.open(io.BytesIO(z.read(n))).convert("L")) for n in names])
    return names, images, np.array([int(n.split("/")[-2]) for n in names])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("assignment",type=Path)
    a=p.parse_args().assignment
    weights=a/"outputs_part1/setting_C_best.pt"
    net=LeNet(historical=True).eval()
    net.load_state_dict(torch.load(weights,map_location="cpu",weights_only=True))
    names,images,y=historical_images(a,"Test")
    assert images.shape==(1000,28,28)
    torch.set_num_threads(2)
    x=(torch.from_numpy(images[:,None]).float()/255-.48215)/.23793
    clean,views=predict(net,x,"reflect",1)
    out=Path("artifacts/pilot")
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"predictions.csv",[dict(id=n,truth=int(t),clean=int(c),
              left=int(v[0]),right=int(v[1]),up=int(v[2]),down=int(v[3]))
              for n,t,c,v in zip(names,y,clean,views)])
    result=dict(n=len(y),views=views.size,flips=int((views!=clean[:,None]).sum()),
                shifted_correct=int((views==y[:,None]).sum()),clean_correct=int((clean==y).sum()),
                checkpoint_sha256=file_hash(weights),predictions_sha256=file_hash(out/"predictions.csv"),
                **metrics(y,clean,views))
    json_write("results/pilot.json",result)
    print(result)


if __name__ == "__main__":
    main()
