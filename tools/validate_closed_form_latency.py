from __future__ import annotations
import copy, importlib.util, itertools, json, math, statistics, time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
modpath=ROOT/'models'/'dt1d_adapter.py'
spec=importlib.util.spec_from_file_location('dt1d_latency_mod',modpath)
mod=importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(mod)
DT1DAdapter=mod.DT1DAdapter

def make(C=16,axis='hw',padding='reflect',no_pw=False,use_bn=False,separate=True,closed=False,dtype=torch.float64):
    m=DT1DAdapter(C=C,M=1,h=1,axis=axis,alpha_group=16,tie_sym=True,no_pw=no_pw,pw_ratio=32,pw_groups=4,use_bn=use_bn,residual_scale=1.0,gate_init=0.01,padding_mode=padding,dilations=(1,2,4),scale_adaptive=True,separate_axis_kernels=separate,gate_temperature=1.0,exact_cost_realization=True,closed_form_dyadic_realization=closed)
    return m.to(dtype=dtype)

def main():
    torch.manual_seed(20260801)
    max_out=max_xg=max_pg=max_loss=0.0
    n=0; min_agree=1.0
    configs=[]
    for C,axis,padding,shape,dtype,no_pw,use_bn,separate in itertools.product(
        (15,16,32,64),('h','w','hw'),('zeros','replicate','reflect'),((2,3),(3,4),(7,8),(19,23)),(torch.float32,torch.float64),(False,True),(False,True),(False,True)):
        # keep runtime bounded while retaining all critical dimensions
        if C>16 and (shape!=(7,8) or axis!='hw' or padding!='reflect' or dtype!=torch.float32 or no_pw or use_bn or not separate):
            continue
        a=make(C,axis,padding,no_pw,use_bn,separate,False,dtype)
        b=make(C,axis,padding,no_pw,use_bn,separate,True,dtype)
        b.load_state_dict(copy.deepcopy(a.state_dict()),strict=True)
        a.train(); b.train()
        x0=torch.randn(2,C,*shape,dtype=dtype)
        xa=x0.clone().requires_grad_(True); xb=x0.clone().requires_grad_(True)
        ya=a(xa); yb=b(xb)
        target=torch.randn_like(ya)
        la=F.mse_loss(ya,target); lb=F.mse_loss(yb,target)
        la.backward(); lb.backward()
        max_out=max(max_out,float((ya-yb).abs().max()))
        max_xg=max(max_xg,float((xa.grad-xb.grad).abs().max()))
        max_loss=max(max_loss,float((la-lb).abs()))
        for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
            assert na==nb
            if pa.grad is not None:
                max_pg=max(max_pg,float((pa.grad-pb.grad).abs().max()))
        agree=float((ya.argmax(1)==yb.argmax(1)).float().mean())
        min_agree=min(min_agree,agree)
        assert tuple(a.state_dict())==tuple(b.state_dict())
        assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
        configs.append(dict(C=C,axis=axis,padding=padding,shape=shape,dtype=str(dtype),no_pw=no_pw,use_bn=use_bn,separate=separate))
        n+=1
    # short trajectory
    a=make(16,'hw','reflect',False,True,True,False,torch.float32)
    b=make(16,'hw','reflect',False,True,True,True,torch.float32); b.load_state_dict(copy.deepcopy(a.state_dict()))
    oa=torch.optim.AdamW(a.parameters(),lr=1e-3); ob=torch.optim.AdamW(b.parameters(),lr=1e-3)
    max_step_loss=max_step_param=0.0; min_step_agree=1.0
    for step in range(100):
        torch.manual_seed(9000+step)
        x=torch.randn(4,16,19,23); t=torch.randn_like(x)
        oa.zero_grad(); ob.zero_grad(); ya=a(x); yb=b(x); la=F.mse_loss(ya,t); lb=F.mse_loss(yb,t); la.backward(); lb.backward(); oa.step(); ob.step()
        max_step_loss=max(max_step_loss,abs(float(la-lb)))
        min_step_agree=min(min_step_agree,float((ya.argmax(1)==yb.argmax(1)).float().mean()))
        for pa,pb in zip(a.parameters(),b.parameters()): max_step_param=max(max_step_param,float((pa-pb).abs().max()))
    report={
        'configurations':n,'max_output_abs_diff':max_out,'max_input_grad_abs_diff':max_xg,'max_parameter_grad_abs_diff':max_pg,'max_loss_abs_diff':max_loss,'minimum_prediction_agreement':min_agree,
        'trajectory_steps':100,'trajectory_max_loss_diff':max_step_loss,'trajectory_max_parameter_diff':max_step_param,'trajectory_min_prediction_agreement':min_step_agree,
        'state_dict_keys_identical':True,'trainable_parameters_identical':True,
    }
    out=ROOT/'validation_closed_form_latency.json'; out.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
