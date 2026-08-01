from __future__ import annotations
import copy, importlib.util, json, random, statistics, time
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('dt1d_latency',ROOT/'models'/'dt1d_adapter.py')
mod=importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(mod)
DT1DAdapter=mod.DT1DAdapter

def kwargs(C,closed):
 return dict(C=C,M=1,h=1,axis='hw',alpha_group=16,no_pw=False,pw_ratio=32,pw_groups=4,use_bn=False,residual_scale=1.0,gate_init=.01,padding_mode='reflect',dilations=(1,2,4),scale_adaptive=True,separate_axis_kernels=True,gate_temperature=1.0,exact_cost_realization=True,closed_form_dyadic_realization=closed)

def paired_times(a,b,x,iters,rounds,seed):
 rng=random.Random(seed); out={'before':[],'after':[]}
 for _ in range(rounds):
  order=['before','after'];rng.shuffle(order)
  for name in order:
   m=a if name=='before' else b
   st=time.perf_counter()
   for _ in range(iters):m(x)
   out[name].append((time.perf_counter()-st)*1000/iters)
 return out

def bootstrap_ratio(before,after,n=4000,seed=0):
 rng=random.Random(seed); vals=[];N=len(before)
 for _ in range(n):
  idx=[rng.randrange(N) for _ in range(N)]
  vals.append(statistics.median(before[i] for i in idx)/statistics.median(after[i] for i in idx))
 vals.sort();return [vals[int(.025*n)],vals[int(.975*n)]]

@torch.no_grad()
def adapter_bench():
 rows=[]
 for C,H,iters in [(64,56,12),(128,28,18),(256,14,25),(512,7,30)]:
  a=DT1DAdapter(**kwargs(C,False)).eval();b=DT1DAdapter(**kwargs(C,True)).eval();b.load_state_dict(copy.deepcopy(a.state_dict()))
  x=torch.randn(4,C,H,H)
  for _ in range(8):a(x);b(x)
  t=paired_times(a,b,x,iters,7,100+C)
  mb=statistics.median(t['before']);ma=statistics.median(t['after'])
  rows.append({'C':C,'H':H,'batch':4,'before_ms':mb,'after_ms':ma,'speedup':mb/ma,'speedup_ci95':bootstrap_ratio(t['before'],t['after'],seed=C),'max_output_abs_diff':float((a(x)-b(x)).abs().max())})
 return rows

def attach(model,closed):
 for layer in [model.layer1,model.layer2,model.layer3,model.layer4]:
  for block in layer:
   ad=DT1DAdapter(**kwargs(block.conv2.out_channels,closed));block.pet_adapter=ad;block.register_forward_hook(lambda m,i,o,ad=ad:ad(o))
 return model

@torch.no_grad()
def resnet_bench():
 import torchvision.models as tvm
 a=attach(tvm.resnet18(weights=None).eval(),False);b=attach(tvm.resnet18(weights=None).eval(),True);b.load_state_dict(copy.deepcopy(a.state_dict()))
 x=torch.randn(2,3,224,224)
 for _ in range(5):a(x);b(x)
 t=paired_times(a,b,x,3,7,2026);mb=statistics.median(t['before']);ma=statistics.median(t['after'])
 return {'batch':2,'input_size':224,'before_ms':mb,'after_ms':ma,'speedup':mb/ma,'speedup_ci95':bootstrap_ratio(t['before'],t['after'],seed=2026),'max_logit_abs_diff':float((a(x)-b(x)).abs().max()),'top1_agreement':float((a(x).argmax(1)==b(x).argmax(1)).float().mean())}

def main():
 torch.set_num_threads(4);torch.manual_seed(20260801)
 report={'environment':{'torch':torch.__version__,'device':'cpu','threads':torch.get_num_threads()},'adapter':adapter_bench(),'resnet18':resnet_bench()}
 (ROOT/'benchmark_closed_form_latency_cpu.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
