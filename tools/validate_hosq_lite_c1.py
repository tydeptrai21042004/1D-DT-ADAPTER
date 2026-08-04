#!/usr/bin/env python3
"""Validate the mathematics and ablations of HOSQ-Lite-C1-Orth."""
from __future__ import annotations
import json, math
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.dt1d_adapter import DT1DAdapter
from models.hosq_lite_c1_adapter import HOSQLiteC1Adapter
OUT=ROOT/'outputs'/'hosq_lite_c1_validation'

def legacy(C=64,no_pw=True):
    return DT1DAdapter(C=C,M=1,axis='hw',alpha_group=16,no_pw=no_pw,gate_init=.17,
        padding_mode='replicate',dilations=(1,2,4),scale_adaptive=True,separate_axis_kernels=True)
def lite(C=64,basis='orth',components='both'):
    return HOSQLiteC1Adapter(C=C,axis='hw',alpha_group=16,gate_init=.17,padding_mode='replicate',
        detail_basis=basis,detail_components=components,contrast_split=8)

def main():
    torch.manual_seed(20260804)
    a,b=legacy(),lite()
    with torch.no_grad():
        a.alpha.normal_(0,.25);a.axis_scale_logits.normal_(0,.35)
    b.initialize_from_dt1d(a)
    warm={}
    for size in (7,8,14,28,56):
        x=torch.randn(2,64,size,size)
        warm[str(size)]=float((a(x)-b(x)).abs().max())
    orth=lite().spectral_atoms.double();raw=lite(basis='raw').spectral_atoms.double()
    gram_orth=orth@orth.T;gram_raw=raw@raw.T
    # Synthetic spectral ablation in the exact detail space.
    target=.7*orth[0]-.4*orth[1]
    errors={}
    for name,atoms in {
        'without_detail':torch.empty(0,17,dtype=torch.float64),
        'offset4_only':orth[:1],
        'offset8_only':orth[1:],
        'raw_both':raw,
        'orth_both':orth,
    }.items():
        if atoms.numel()==0: fit=torch.zeros_like(target)
        else:
            coef=torch.linalg.lstsq(atoms.T,target).solution
            fit=coef@atoms
        errors[name]={'mse':float((fit-target).square().mean()),'max_abs':float((fit-target).abs().max())}
    stable=lite()
    with torch.no_grad():stable.quotient_beta.normal_(0,3);stable.detail_eta.normal_(0,3)
    k=stable.build_kernels(torch.device('cpu'),torch.float64).squeeze(2)
    channels=(64,64,128,128,256,256,512,512)
    params=sum(sum(p.numel() for p in lite(c).parameters() if p.requires_grad) for c in channels)
    result={
      'method':'HOSQ-Lite-C1-Orth','legacy_warm_start_max_abs':warm,
      'orthogonal_gram':gram_orth.tolist(),'raw_gram':gram_raw.tolist(),
      'orthogonal_gram_condition':float(torch.linalg.cond(gram_orth)),
      'raw_gram_condition':float(torch.linalg.cond(gram_raw)),
      'spectral_ablation':errors,
      'max_joint_axis_l1':float(k.abs().sum(-1).sum(0).max()),
      'resnet18_adapter_parameters':params,'convolutions_per_adapter':2,
      'checks':{
        'warm_start':max(warm.values())<3e-6,
        'orthonormal':bool(torch.allclose(gram_orth,torch.eye(2,dtype=torch.float64),atol=2e-7,rtol=0)),
        'nonexpansive':float(k.abs().sum(-1).sum(0).max())<=1+1e-12,
        'orth_both_exact':errors['orth_both']['max_abs']<1e-10,
      }
    }
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'validation.json';path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not all(result['checks'].values()):raise SystemExit(1)
    print(f'Wrote {path}')
if __name__=='__main__':main()
