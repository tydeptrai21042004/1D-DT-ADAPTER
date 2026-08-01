from __future__ import annotations
import copy, importlib.util, json, random, statistics, time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location('base',ROOT/'models'/'dt1d_adapter.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);Base=m.DT1DAdapter

class GroupSpace(Base):
    def _kg(self,ai,si,device,dtype):
        K=2*self.M+3;c=self.M+1;aa=ai if self.separate_axis_kernels else 0;a=self.alpha[aa,si].to(device=device,dtype=dtype);wg=torch.zeros(self.num_alpha_groups,K,device=device,dtype=dtype)
        for mm in range(-self.M,self.M+1):
            v=a[:,abs(mm)]
            for r in (-(mm+1),-(mm-1),(mm+1),(mm-1)):wg[:,c+r]+=0.5*v
        return wg/wg.abs().sum(1,keepdim=True).clamp_min(1e-6)
    def _build_weighted_dt1d_kernel_1d(self,ai,si,device,dtype):return self._expand_group_kernel_to_channels(self._kg(ai,si,device,dtype))
    def _build_exact_group_kernel_1d(self,ai,inds,w,device,dtype):
        rad=self.M+1;mo=rad*max(self.dilations[i] for i in inds);f=torch.zeros(self.num_alpha_groups,2*mo+1,device=device,dtype=dtype);off=torch.arange(-rad,rad+1,device=device)
        for si in inds:f=f.index_add(1,mo+self.dilations[si]*off,(w[ai,si]*self._kg(ai,si,device,dtype)).to(dtype))
        return self._expand_group_kernel_to_channels(f)

class LaunchFirst(GroupSpace):
    def _kernel(self,ai,inds,w,device,dtype):
        rad=self.M+1;mo=rad*max(self.dilations[i] for i in inds);f=torch.zeros(self.num_alpha_groups,2*mo+1,device=device,dtype=dtype);off=torch.arange(-rad,rad+1,device=device)
        for si in inds:f=f.index_add(1,mo+self.dilations[si]*off,(w[ai,si]*self._kg(ai,si,device,dtype)).to(dtype))
        return self._expand_group_kernel_to_channels(f)
    def forward(self,x):
        if not self.scale_adaptive:return super().forward(x)
        w=self._compute_axis_scale_weights(x.device,x.dtype);y=torch.zeros_like(x);rad=self.M+1
        for ai,axis in enumerate(self.axis_names):
            if self.padding_mode=='reflect':
                spatial=x.shape[-2] if axis=='h' else x.shape[-1]
                a=tuple(i for i,d in enumerate(self.dilations) if rad*d<spatial);b=tuple(i for i,d in enumerate(self.dilations) if rad*d>=spatial);classes=tuple(g for g in (a,b) if g)
            else:classes=(tuple(range(self.num_scales)),)
            for inds in classes:y=y+self._conv_axis(x,axis,self._kernel(ai,inds,w,x.device,x.dtype),1)
        y=self.pw(y);return x+self.residual_scale*self.gate*y

def kw(C,closed=False):return dict(C=C,M=1,h=1,axis='hw',alpha_group=16,no_pw=False,pw_ratio=32,pw_groups=4,use_bn=False,residual_scale=1.,gate_init=.01,padding_mode='reflect',dilations=(1,2,4),scale_adaptive=True,separate_axis_kernels=True,gate_temperature=1.,exact_cost_realization=True,closed_form_dyadic_realization=closed)
def attach(cls,closed=False):
 import torchvision.models as tvm
 model=tvm.resnet18(weights=None).eval()
 for layer in [model.layer1,model.layer2,model.layer3,model.layer4]:
  for block in layer:
   ad=cls(**kw(block.conv2.out_channels,closed));block.pet_adapter=ad;block.register_forward_hook(lambda mm,i,o,ad=ad:ad(o))
 return model
@torch.no_grad()
def main():
 torch.set_num_threads(4);torch.manual_seed(123)
 names=['dynamic_tap_first','group_space_tap_first','closed_form_tap_first','launch_first_full_fusion']
 models=[attach(Base,False),attach(GroupSpace,False),attach(Base,True),attach(LaunchFirst,False)]
 state=copy.deepcopy(models[0].state_dict())
 for z in models[1:]:z.load_state_dict(state)
 x=torch.randn(2,3,224,224)
 for _ in range(4):
  for z in models:z(x)
 vals={n:[] for n in names};rng=random.Random(99)
 for _ in range(7):
  order=list(range(len(models)));rng.shuffle(order)
  for i in order:
   st=time.perf_counter()
   for _ in range(3):models[i](x)
   vals[names[i]].append((time.perf_counter()-st)*1000/3)
 ref=models[0](x)
 rows=[]
 for n,z in zip(names,models):
  med=statistics.median(vals[n]);rows.append({'candidate':n,'median_ms':med,'speedup_vs_dynamic':statistics.median(vals[names[0]])/med,'max_logit_abs_diff':float((ref-z(x)).abs().max()),'top1_agreement':float((ref.argmax(1)==z(x).argmax(1)).float().mean())})
 report={'environment':{'torch':torch.__version__,'threads':4,'device':'cpu'},'resnet18_batch2_224':rows,'selection':'closed_form_tap_first','reason':'best exact speed without increasing dense-tap/FLOP cost; launch-first is exact but increases dense support; group-space is generic but slower than the closed form.'}
 (ROOT/'latency_candidate_comparison_cpu.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
