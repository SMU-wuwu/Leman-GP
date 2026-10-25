"""Independent KL and accumulation checks for the new training objective."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'leman_gp_rdrop_v17'))
from consistency import span_symmetric_kl, rdrop_backward


def main():
    torch.manual_seed(19)
    first = torch.randn(3,2,5,5,requires_grad=True)
    second = torch.randn(3,2,5,5,requires_grad=True)
    mask = torch.tensor([[1,1,1,1,1],[1,1,0,0,0],[0,0,0,0,0]],dtype=torch.bool)
    actual = span_symmetric_kl(first,second,mask,3)
    terms = []
    for row in range(3):
        for typ in range(2):
            coords = [(i,j) for i in range(5) for j in range(i,min(5,i+3))
                      if mask[row,i] and mask[row,j]]
            a = torch.stack([first[row,typ,i,j].double() for i,j in coords]+[torch.tensor(0.)])
            b = torch.stack([second[row,typ,i,j].double() for i,j in coords]+[torch.tensor(0.)])
            p,q = a.softmax(0),b.softmax(0)
            terms.append(((p*(p.log()-q.log())).sum()+(q*(q.log()-p.log())).sum())/2)
    reference = torch.stack(terms).mean()
    torch.testing.assert_close(actual.double(),reference,rtol=3e-6,atol=3e-7)
    actual.backward()
    assert all(torch.isfinite(t.grad).all() for t in (first,second))
    for row in range(3):
        for i in range(5):
            for j in range(5):
                if not (mask[row,i] and mask[row,j] and 0 <= j-i < 3):
                    assert first.grad[row,:,i,j].eq(0).all() and second.grad[row,:,i,j].eq(0).all()
    assert span_symmetric_kl(first,first,mask,3).item() == 0
    assert span_symmetric_kl(first,second,torch.zeros_like(mask),3).item() == 0

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.3))
            self.cfg = SimpleNamespace(max_span=3)
        def forward(self,input_ids,char_mask):
            scores = input_ids.float()*self.weight
            return scores.square().flatten(1).mean(1).mean(),scores
    # Uneven last microbatch: production sample-count weights must equal one
    # full-window gradient. Dropout disabled here to isolate accumulation math.
    source = torch.randn(7,2,5,5)
    masks = torch.ones(7,5,dtype=torch.bool)
    whole, split = Toy(), Toy()
    class Wrapped:
        def __init__(self,model): self.module = model
        def __call__(self,**kwargs): return self.module(**kwargs)
    rdrop_backward(Wrapped(whole),{'input_ids':source,'char_mask':masks},weight=1)
    for a,b in ((0,3),(3,6),(6,7)):
        rdrop_backward(Wrapped(split),{'input_ids':source[a:b],'char_mask':masks[a:b]},
                       weight=1,loss_scale=(b-a)/7)
    torch.testing.assert_close(whole.weight.grad,split.weight.grad,rtol=1e-6,atol=1e-7)
    micro, accumulation, size, count = 7,4,3,20
    windows = []
    for i in range(micro):
        start = i//accumulation*accumulation
        end = min(start+accumulation,micro)
        denominator = min(end*size,count)-start*size
        windows.append((i,denominator,i+1 == end))
    assert [d for _,d,last in windows if last] == [12,8]
    assert sum(last for _,_,last in windows) == 2
    result = {'status':'PASS','independent_KL':float(reference.detach()),
              'implementation_KL':float(actual.detach()),'padding_gradient_zero':True,
              'empty_type_null_finite':True,'uneven_accumulation_gradient_equal':True}
    (Path(__file__).parent/'rdrop_math_qa.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
