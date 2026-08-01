import copy
import importlib.util
from pathlib import Path
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('dt1d_latency_test',ROOT/'models'/'dt1d_adapter.py')
mod=importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(mod)
DT1DAdapter=mod.DT1DAdapter

def make(closed, **overrides):
 kw=dict(C=16,M=1,h=1,axis='hw',alpha_group=16,no_pw=False,pw_ratio=32,pw_groups=4,use_bn=False,residual_scale=1.0,gate_init=.01,padding_mode='reflect',dilations=(1,2,4),scale_adaptive=True,separate_axis_kernels=True,gate_temperature=1.0,exact_cost_realization=True,closed_form_dyadic_realization=closed)
 kw.update(overrides);return DT1DAdapter(**kw)

@pytest.mark.parametrize('padding',['zeros','replicate','reflect'])
@pytest.mark.parametrize('shape',[(2,3),(3,4),(7,8),(19,23)])
@pytest.mark.parametrize('axis',['h','w','hw'])
def test_closed_form_matches_dynamic_exact(padding,shape,axis):
 torch.manual_seed(12)
 a=make(False,padding_mode=padding,axis=axis).double();b=make(True,padding_mode=padding,axis=axis).double();b.load_state_dict(copy.deepcopy(a.state_dict()))
 x0=torch.randn(2,16,*shape,dtype=torch.float64);xa=x0.clone().requires_grad_();xb=x0.clone().requires_grad_();ya=a(xa);yb=b(xb)
 torch.testing.assert_close(yb,ya,rtol=1e-10,atol=1e-11)
 (ya.square().mean()).backward();(yb.square().mean()).backward();torch.testing.assert_close(xb.grad,xa.grad,rtol=1e-9,atol=1e-11)
 for pa,pb in zip(a.parameters(),b.parameters()):
  if pa.grad is not None:torch.testing.assert_close(pb.grad,pa.grad,rtol=1e-8,atol=1e-11)

def test_parameters_and_checkpoint_keys_unchanged():
 a=make(False);b=make(True)
 assert tuple(a.state_dict())==tuple(b.state_dict())
 assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())

def test_fallback_for_other_configurations():
 a=make(False,M=2,dilations=(1,3,5));b=make(True,M=2,dilations=(1,3,5));b.load_state_dict(a.state_dict());x=torch.randn(2,16,17,19)
 torch.testing.assert_close(a(x),b(x),rtol=0,atol=0)
