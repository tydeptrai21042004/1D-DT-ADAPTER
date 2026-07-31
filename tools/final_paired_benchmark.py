from __future__ import annotations
import importlib.util, json, math, random, statistics, time
from pathlib import Path
import torch
from torchvision.models import resnet18

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('dt1d_final_pair', ROOT/'models'/'hcc_adapter.py')
mod=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(mod)
DT1DAdapter=mod.DT1DAdapter

def median_ci(vals, seed=123, samples=4000):
    rng=random.Random(seed); n=len(vals); meds=[]
    for _ in range(samples): meds.append(statistics.median(vals[rng.randrange(n)] for _ in range(n)))
    meds.sort(); return [meds[int(.025*samples)], meds[int(.975*samples)-1]]

def bench_call(fn, iters):
    t=time.perf_counter()
    for _ in range(iters): fn()
    return (time.perf_counter()-t)*1000/iters

def adapter_pair():
    kw=dict(C=64,M=1,dilations=(1,2,4),scale_adaptive=True,axis='hw',alpha_group=16,no_pw=False,pw_ratio=8,pw_groups=4,gate_init=.2,padding_mode='reflect',separate_axis_kernels=True)
    b=DT1DAdapter(**kw,exact_cost_realization=False); a=DT1DAdapter(**kw,exact_cost_realization=True); a.load_state_dict(b.state_dict()); return b,a

class Net(torch.nn.Module):
    def __init__(self,opt):
        super().__init__(); self.net=resnet18(weights=None); self.net.fc=torch.nn.Linear(512,10)
        cs=[block.conv2.out_channels for layer in (self.net.layer1,self.net.layer2,self.net.layer3,self.net.layer4) for block in layer]
        self.adapters=torch.nn.ModuleList([DT1DAdapter(C=c,M=1,dilations=(1,2,4),scale_adaptive=True,axis='hw',alpha_group=16,no_pw=False,pw_ratio=32,pw_groups=4,gate_init=.2,padding_mode='reflect',separate_axis_kernels=True,exact_cost_realization=opt) for c in cs])
    def forward(self,x):
        n=self.net; x=n.maxpool(n.relu(n.bn1(n.conv1(x)))); i=0
        for layer in (n.layer1,n.layer2,n.layer3,n.layer4):
            for block in layer: x=self.adapters[i](block(x)); i+=1
        return n.fc(torch.flatten(n.avgpool(x),1))

def paired(name,before,after,x,target=None,rounds=31,iters=5,train_rounds=17):
    rng=random.Random(20260731)
    before.eval(); after.eval()
    with torch.inference_mode():
        for _ in range(5): before(x); after(x)
        yb=before(x); ya=after(x)
        fwd=[]
        for r in range(rounds):
            order=['before','after']; rng.shuffle(order); d={}
            for key in order: d[key]=bench_call((lambda: before(x)) if key=='before' else (lambda: after(x)), iters)
            fwd.append({'before_ms':d['before'],'after_ms':d['after'],'speedup':d['before']/d['after']})
    train=[]
    if target is not None:
        before.train(); after.train()
        def step(m):
            m.zero_grad(set_to_none=True); out=m(x)
            loss=torch.nn.functional.cross_entropy(out,target) if out.ndim==2 else out.square().mean()
            loss.backward()
        for _ in range(2): step(before); step(after)
        for r in range(train_rounds):
            order=['before','after']; rng.shuffle(order); d={}
            for key in order: d[key]=bench_call((lambda: step(before)) if key=='before' else (lambda: step(after)), 1)
            train.append({'before_ms':d['before'],'after_ms':d['after'],'speedup':d['before']/d['after']})
    def summarize(rows):
        sp=[r['speedup'] for r in rows]; b=[r['before_ms'] for r in rows]; a=[r['after_ms'] for r in rows]
        return {'rounds':len(rows),'median_before_ms':statistics.median(b),'median_after_ms':statistics.median(a),'median_paired_speedup':statistics.median(sp),'paired_speedup_95pct_bootstrap_ci':median_ci(sp),'after_faster_round_fraction':sum(v>1 for v in sp)/len(sp),'all_paired_speedups':sp}
    return {'name':name,'max_abs_output_diff':float((yb-ya).abs().max()),'top1_or_sign_agreement':float((yb.argmax(1)==ya.argmax(1)).float().mean()) if yb.ndim==2 else float(((yb>=0)==(ya>=0)).float().mean()),'inference':summarize(fwd),'forward_backward':summarize(train) if train else None}

def main():
    torch.set_num_threads(4); torch.manual_seed(7777)
    ab,aa=adapter_pair(); x1=torch.randn(4,64,56,56)
    # add dummy classifier to use CE for adapter training timing
    class Wrap(torch.nn.Module):
        def __init__(self,m): super().__init__(); self.m=m; self.fc=torch.nn.Linear(64,10)
        def forward(self,x): return self.fc(self.m(x).mean((2,3)))
    wb,wa=Wrap(ab),Wrap(aa); wa.fc.load_state_dict(wb.fc.state_dict()); y1=torch.randint(0,10,(4,))
    nb=Net(False); na=Net(True); na.load_state_dict(nb.state_dict());
    for m in (nb,na):
        for n,p in m.named_parameters(): p.requires_grad=('adapters' in n or 'net.fc' in n)
    x2=torch.randn(2,3,224,224); y2=torch.randint(0,10,(2,))
    out={'torch':torch.__version__,'threads':torch.get_num_threads(),'fairness':{'interleaved_random_order':True,'same_class':True,'same_weights':True,'same_input':True,'cache':False,'only_switch':'exact_cost_realization'},'adapter':paired('adapter_64x56',wb,wa,x1,y1,rounds=21,iters=6,train_rounds=11),'resnet18':paired('resnet18_b2_224',nb,na,x2,y2,rounds=15,iters=2,train_rounds=7)}
    (ROOT/'final_paired_timing_audit.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
