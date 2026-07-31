"""Exhaustive numerical regression of the exact mathematical realization."""
from __future__ import annotations
import importlib.util, json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=Path('/mnt/data/dt1d_repo/1D-DT-ADAPTER-main/models/hcc_adapter.py')

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(mod); return mod.DT1DAdapter
Old=load(ORIGINAL,'old_dt1d_validation'); New=load(ROOT/'models'/'hcc_adapter.py','new_dt1d_validation')

def build(cls,m,dils,pad,dtype,no_pw,opt=None):
    kw=dict(C=5,M=m,dilations=dils,scale_adaptive=True,axis='hw',alpha_group=2,no_pw=no_pw,pw_ratio=4,pw_groups=1,gate_init=0.27,padding_mode=pad,separate_axis_kernels=True)
    if opt is not None: kw['exact_cost_realization']=opt
    return cls(**kw).to(dtype=dtype)

def main():
    torch.set_num_threads(4)
    Ms=(0,1,2,3)
    dilation_sets=((1,2),(1,2,4),(1,3,5),(2,4),(1,2,3,4))
    paddings=('zeros','replicate','reflect')
    shapes=((19,23),(7,8),(3,4))
    dtypes=(torch.float32,torch.float64)
    no_pws=(True,False)
    maxima={'output_abs':0.0,'input_grad_abs':0.0,'parameter_grad_abs':0.0,'loss_abs':0.0}
    min_prediction_agreement=1.0
    total=0
    cost_violations=[]
    for m in Ms:
      for dils in dilation_sets:
       for pad in paddings:
        for shape in shapes:
         for dtype in dtypes:
          for no_pw in no_pws:
            total+=1
            seed=100000+total
            torch.manual_seed(seed)
            old=build(Old,m,dils,pad,dtype,no_pw)
            new=build(New,m,dils,pad,dtype,no_pw,True)
            with torch.no_grad():
                old.alpha.normal_(); old.axis_scale_logits.normal_(); old.gate.fill_(0.31)
                for p in old.pw.parameters(): p.normal_(0,0.1)
            new.load_state_dict(old.state_dict(),strict=True)
            xo=torch.randn(2,5,*shape,dtype=dtype,requires_grad=True)
            xn=xo.detach().clone().requires_grad_(True)
            yo=old(xo); yn=new(xn)
            maxima['output_abs']=max(maxima['output_abs'],float((yo-yn).abs().max().detach()))
            pred_o=yo.mean((2,3)).argmax(1); pred_n=yn.mean((2,3)).argmax(1)
            min_prediction_agreement=min(min_prediction_agreement,float((pred_o==pred_n).float().mean()))
            target=torch.randn_like(yo)
            lo=torch.nn.functional.mse_loss(yo,target); ln=torch.nn.functional.mse_loss(yn,target)
            maxima['loss_abs']=max(maxima['loss_abs'],float((lo-ln).abs().detach()))
            lo.backward(); ln.backward()
            maxima['input_grad_abs']=max(maxima['input_grad_abs'],float((xo.grad-xn.grad).abs().max().detach()))
            for (_,po),(_,pn) in zip(old.named_parameters(),new.named_parameters()):
                if po.grad is not None:
                    maxima['parameter_grad_abs']=max(maxima['parameter_grad_abs'],float((po.grad-pn.grad).abs().max().detach()))
            for axis in ('h','w'):
                c=new.exact_realization_cost(xn,axis)
                if c['after_calls']>c['before_calls'] or c['after_dense_taps']>c['before_dense_taps']:
                    cost_violations.append({'m':m,'dilations':dils,'padding':pad,'shape':shape,'axis':axis,'cost':c})
    # Deterministic metric-level check over 1000 synthetic examples.
    torch.manual_seed(999)
    old=build(Old,1,(1,2,4),'reflect',torch.float32,False)
    new=build(New,1,(1,2,4),'reflect',torch.float32,False,True)
    with torch.no_grad():
        old.alpha.normal_(); old.axis_scale_logits.normal_(); old.gate.fill_(0.31)
        for p in old.pw.parameters(): p.normal_(0,0.1)
    new.load_state_dict(old.state_dict(),strict=True)
    head=torch.nn.Linear(5,7)
    x=torch.randn(1000,5,9,11); labels=torch.randint(0,7,(1000,))
    with torch.no_grad():
        logits_old=head(old(x).mean((2,3))); logits_new=head(new(x).mean((2,3)))
        metric={
          'old_accuracy':float((logits_old.argmax(1)==labels).float().mean()),
          'new_accuracy':float((logits_new.argmax(1)==labels).float().mean()),
          'old_cross_entropy':float(torch.nn.functional.cross_entropy(logits_old,labels)),
          'new_cross_entropy':float(torch.nn.functional.cross_entropy(logits_new,labels)),
          'top1_agreement':float((logits_old.argmax(1)==logits_new.argmax(1)).float().mean()),
          'max_logit_abs_difference':float((logits_old-logits_new).abs().max()),
        }
    # Fifty-step optimizer trajectory: same minibatches, head, loss, and optimizer.
    torch.manual_seed(12345)
    old_train=build(Old,1,(1,2,4),'reflect',torch.float32,False)
    new_train=build(New,1,(1,2,4),'reflect',torch.float32,False,True)
    with torch.no_grad():
        old_train.alpha.normal_(); old_train.axis_scale_logits.normal_(); old_train.gate.fill_(0.31)
        for p in old_train.pw.parameters(): p.normal_(0,0.1)
    new_train.load_state_dict(old_train.state_dict(),strict=True)
    head_old=torch.nn.Linear(5,7); head_new=torch.nn.Linear(5,7); head_new.load_state_dict(head_old.state_dict())
    optimizer_old=torch.optim.AdamW(list(old_train.parameters())+list(head_old.parameters()),lr=1e-3)
    optimizer_new=torch.optim.AdamW(list(new_train.parameters())+list(head_new.parameters()),lr=1e-3)
    max_train_loss_diff=0.0; max_train_logit_diff=0.0; min_train_agreement=1.0
    for step in range(50):
        generator=torch.Generator().manual_seed(50000+step)
        xb=torch.randn(8,5,13,15,generator=generator); yb=torch.randint(0,7,(8,),generator=generator)
        records=[]
        for model,head_model,optimizer in ((old_train,head_old,optimizer_old),(new_train,head_new,optimizer_new)):
            optimizer.zero_grad(set_to_none=True); logits=head_model(model(xb).mean((2,3))); loss=torch.nn.functional.cross_entropy(logits,yb); loss.backward(); optimizer.step(); records.append((loss.detach(),logits.detach()))
        max_train_loss_diff=max(max_train_loss_diff,float((records[0][0]-records[1][0]).abs()))
        max_train_logit_diff=max(max_train_logit_diff,float((records[0][1]-records[1][1]).abs().max()))
        min_train_agreement=min(min_train_agreement,float((records[0][1].argmax(1)==records[1][1].argmax(1)).float().mean()))
    max_train_parameter_diff=0.0
    for po,pn in zip(list(old_train.parameters())+list(head_old.parameters()),list(new_train.parameters())+list(head_new.parameters())):
        max_train_parameter_diff=max(max_train_parameter_diff,float((po-pn).abs().max().detach()))
    training_trajectory={
      'steps':50,'optimizer':'AdamW','max_loss_abs_difference':max_train_loss_diff,
      'max_logit_abs_difference':max_train_logit_diff,'minimum_top1_agreement':min_train_agreement,
      'max_parameter_abs_difference_after_50_steps':max_train_parameter_diff,
    }

    result={
      'configuration_count':total,
      'dimensions':{'M':list(Ms),'dilation_sets':[list(x) for x in dilation_sets],'paddings':list(paddings),'shapes':[list(x) for x in shapes],'dtypes':['float32','float64'],'pointwise':[False,True]},
      'max_absolute_differences':maxima,
      'minimum_prediction_agreement':min_prediction_agreement,
      'cost_violation_count':len(cost_violations),
      'cost_violations':cost_violations,
      'synthetic_metric_regression':metric,
      'training_trajectory':training_trajectory,
      'checkpoint_keys_identical':tuple(old.state_dict().keys())==tuple(new.state_dict().keys()),
      'trainable_parameters_identical':sum(p.numel() for p in old.parameters())==sum(p.numel() for p in new.parameters()),
    }
    out=ROOT/'validation_dt1d_math_realization.json'; out.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
