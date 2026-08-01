from __future__ import annotations
import copy, importlib.util, json
from pathlib import Path
import torch
from torch.profiler import profile, ProfilerActivity
ROOT=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location('m',ROOT/'models'/'dt1d_adapter.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);DT=m.DT1DAdapter

def kw(C,closed):return dict(C=C,M=1,h=1,axis='hw',alpha_group=16,no_pw=False,pw_ratio=32,pw_groups=4,use_bn=False,residual_scale=1.,gate_init=.01,padding_mode='reflect',dilations=(1,2,4),scale_adaptive=True,separate_axis_kernels=True,gate_temperature=1.,exact_cost_realization=True,closed_form_dyadic_realization=closed)
def counts(model,x):
 with profile(activities=[ProfilerActivity.CPU],record_shapes=False,profile_memory=True,with_flops=True) as p:model(x)
 d={e.key:{'calls':e.count,'cpu_time_total_us':e.cpu_time_total,'flops':e.flops,'self_cpu_memory_usage':e.self_cpu_memory_usage} for e in p.key_averages()}
 keys=['aten::conv2d','aten::convolution','aten::repeat','aten::index_add','aten::stack','aten::pad','aten::softmax','aten::mul','aten::add']
 return {k:d.get(k,{'calls':0,'cpu_time_total_us':0,'flops':0,'self_cpu_memory_usage':0}) for k in keys}
def attach(closed):
 import torchvision.models as tvm
 z=tvm.resnet18(weights=None).eval()
 for l in [z.layer1,z.layer2,z.layer3,z.layer4]:
  for b in l:
   ad=DT(**kw(b.conv2.out_channels,closed));b.pet_adapter=ad;b.register_forward_hook(lambda mm,i,o,ad=ad:ad(o))
 return z
@torch.no_grad()
def main():
 torch.set_num_threads(4);torch.manual_seed(1)
 a=DT(**kw(64,False)).eval();b=DT(**kw(64,True)).eval();b.load_state_dict(copy.deepcopy(a.state_dict()));x=torch.randn(4,64,56,56)
 ra=attach(False);rb=attach(True);rb.load_state_dict(copy.deepcopy(ra.state_dict()));xr=torch.randn(2,3,224,224)
 report={'adapter_C64_H56_batch4':{'before':counts(a,x),'after':counts(b,x),'max_output_abs_diff':float((a(x)-b(x)).abs().max())},'resnet18_batch2_224':{'before':counts(ra,xr),'after':counts(rb,xr),'max_logit_abs_diff':float((ra(xr)-rb(xr)).abs().max())}}
 (ROOT/'profile_closed_form_ops.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
