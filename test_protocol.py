import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from data import ROOT, digest, train_stats
from model import LeNet, MaxBlurPool, epoch_plan, metrics, shift
from train import better, param_hash, run, setup


class Protocol(unittest.TestCase):
    def test_split_and_normalization(self):
        y=np.load(ROOT/"labels.npy")
        tr,va=[np.load(ROOT/(k+".npy")) for k in ("train","val")]
        self.assertTrue(np.array_equal(np.sort(np.r_[tr,va]), np.arange(50000)))
        for ids,n in [(tr,4500),(va,500)]:
            np.testing.assert_array_equal(np.bincount(y[ids]),np.full(10,n))
        x=np.load(ROOT/"images.npy",mmap_mode="r")
        info=json.loads(Path("results/split.json").read_text())
        m,s=train_stats(x,tr)
        np.testing.assert_allclose([m,s],[info["mean"],info["std"]])
        # Changing withheld pixels cannot change the normalization.
        a=np.array([[[[0,255]]],[[[80,100]]]],dtype=np.uint8)
        self.assertEqual(train_stats(a,[0]),([.5],[.5]))
        a[1]=255
        self.assertEqual(train_stats(a,[0]),([.5],[.5]))

    def test_shifts(self):
        x=torch.arange(9).reshape(1,1,3,3).float()
        torch.testing.assert_close(shift(x,1,0)[0,0],x[0,0][:,[1,0,1]])
        torch.testing.assert_close(shift(x,0,-1)[0,0],x[0,0][[1,2,1]])
        raw=F.pad(x,(0,1,0,1),value=4)[...,1:,1:]
        torch.testing.assert_close(shift((x-4)/2,-1,-1,"mean"),(raw-4)/2)
        batch=x.repeat(2,1,1,1)
        torch.testing.assert_close(shift(batch,[1,0],[0,-1]),torch.cat([shift(x,1,0),shift(x,0,-1)]))

    def test_metrics_and_selection(self):
        y=np.array([1,0]); clean=np.array([1,2]); v=np.array([[1,1,0,1],[0,2,2,2]])
        self.assertEqual(metrics(y,clean,v),dict(flip_rate=.25,clean_accuracy=.5,shifted_accuracy=.5))
        self.assertTrue(better(5,4)); self.assertFalse(better(5,5))
        self.assertFalse(better(4,5))

    def test_networks(self):
        hashes=[]
        for blur in (False,True):
            torch.manual_seed(17)
            net=LeNet(blur)
            hashes.append(param_hash(net))
            shapes=[]
            handles=[m.register_forward_hook(lambda m,i,o:shapes.append(tuple(o.shape[1:])))
                     for m in (net.conv1,net.pool1,net.conv2,net.pool2,net.fc1,net.fc2,net.fc3)]
            out=net(torch.rand(2,3,32,32))
            self.assertEqual(shapes,[(6,32,32),(6,16,16),(16,12,12),(16,6,6),(120,),(84,),(10,)])
            buffers={k:t.clone() for k,t in net.named_buffers()}
            out.sum().backward()
            self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in net.parameters()))
            torch.optim.SGD(net.parameters(),lr=.1).step()
            for k,t in net.named_buffers(): torch.testing.assert_close(t,buffers[k])
            for h in handles: h.remove()
        self.assertEqual(hashes[0],hashes[1])
        x=torch.rand(2,3,12,12)
        dense=F.max_pool2d(x,2,1)
        k=torch.tensor([[1.,2,1],[2,4,2],[1,2,1]])/16
        ref=F.conv2d(F.pad(dense,(1,1,1,1),"reflect"),k.repeat(3,1,1,1),groups=3)
        torch.testing.assert_close(MaxBlurPool(3)(x),ref[:,:,::2,::2])

    def test_pairing_and_resume(self):
        setup("cpu")
        for seed in range(3):
            order, offsets = epoch_plan(seed,2,131)
            for _ in range(4):
                a,b=epoch_plan(seed,2,131)
                np.testing.assert_array_equal(a,order); np.testing.assert_array_equal(b,offsets)
            self.assertEqual(len(np.unique(order)),131)
            self.assertTrue(np.all(abs(offsets)<=2))
            self.assertNotEqual(digest(order),digest(epoch_plan(seed,1,131)[0]))
        torch.manual_seed(99)
        data=[torch.rand(131,3,32,32),torch.arange(131)%10,torch.rand(37,3,32,32),torch.arange(37)%10]
        with tempfile.TemporaryDirectory() as t:
            a,b=Path(t)/"continuous",Path(t)/"resume"
            run("D",1,data,a,3,"cpu")
            run("D",1,data,b,3,"cpu",stop_after=1)
            run("D",1,data,b,3,"cpu")
            l,r=[torch.load(p/"D1/last.pt",weights_only=False) for p in (a,b)]
            for k in l["model"]: torch.testing.assert_close(l["model"][k],r["model"][k],rtol=0,atol=0)
            self.assertEqual(l["scheduler"],r["scheduler"])
            for x,y in zip(l["history"],r["history"]):
                self.assertEqual({k:v for k,v in x.items() if k!="seconds"},{k:v for k,v in y.items() if k!="seconds"})


if __name__=="__main__":
    unittest.main()
