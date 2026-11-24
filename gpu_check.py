"""Compare graph replay against ordinary SGD and interrupt/resume a small CUDA run."""
from pathlib import Path
import tempfile
import torch
from torch.nn import functional as F
from model import LeNet
from train import setup,BatchStep,run


def main():
    setup("cuda")
    torch.manual_seed(9)
    x=torch.randn(131,3,32,32,device="cuda").contiguous(memory_format=torch.channels_last)
    y=torch.arange(131,device="cuda")%10
    nets=[LeNet(True).to(device="cuda",memory_format=torch.channels_last) for _ in range(2)]
    nets[1].load_state_dict(nets[0].state_dict())
    opts=[torch.optim.SGD(n.parameters(),lr=.02,momentum=.9,weight_decay=.0005,fused=True) for n in nets]
    step=BatchStep(nets[0],opts[0],x,y)
    for _ in range(2):
        for a,b in zip(x.split(64),y.split(64)):
            step(a,b)
            opts[1].zero_grad(set_to_none=True)
            F.cross_entropy(nets[1](a),b).backward()
            opts[1].step()
    for a,b in zip(nets[0].parameters(),nets[1].parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    with tempfile.TemporaryDirectory() as t:
        p,q=Path(t)/"continuous",Path(t)/"resume"
        data=[x,y,x[:31],y[:31]]
        run("D",0,data,p,3,"cuda")
        run("D",0,data,q,3,"cuda",stop_after=1)
        run("D",0,data,q,3,"cuda")
        models=[torch.load(root/"D0/last.pt",map_location="cpu",weights_only=False)["model"] for root in (p,q)]
        for k in models[0]:torch.testing.assert_close(models[0][k],models[1][k],rtol=0,atol=0)
    print("PASS. Graph and eager updates match; uninterrupted and resumed CUDA weights match exactly.")


if __name__=="__main__":
    main()
