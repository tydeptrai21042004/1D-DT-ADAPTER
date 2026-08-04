#!/usr/bin/env python3
"""Paired CPU latency benchmark: original DT1D and HOSQ-Lite ablations."""
from __future__ import annotations
import argparse,json,statistics,time
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.dt1d_adapter import DT1DAdapter
from models.hosq_lite_c1_adapter import HOSQLiteC1Adapter
SHAPES=((64,56,56),(64,56,56),(128,28,28),(128,28,28),(256,14,14),(256,14,14),(512,7,7),(512,7,7))

def build(name):
  mods=[]
  for c,_,_ in SHAPES:
    if name=='original_dt1d_pointwise':m=DT1DAdapter(C=c,M=1,axis='hw',alpha_group=16,no_pw=False,gate_init=.01,padding_mode='replicate',dilations=(1,2,4),scale_adaptive=True,separate_axis_kernels=True)
    elif name=='original_dt1d_core':m=DT1DAdapter(C=c,M=1,axis='hw',alpha_group=16,no_pw=True,gate_init=.01,padding_mode='replicate',dilations=(1,2,4),scale_adaptive=True,separate_axis_kernels=True)
    else:
      basis='raw' if name=='raw_atoms' else 'orth';comp={'without_detail':'none','offset4_only':'offset4','offset8_only':'offset8'}.get(name,'both')
      m=HOSQLiteC1Adapter(C=c,axis='hw',alpha_group=16,gate_init=.01,padding_mode='replicate',detail_basis=basis,detail_components=comp)
    mods.append(m.eval())
  return torch.nn.ModuleList(mods)

def main():
 p=argparse.ArgumentParser();p.add_argument('--batch-size',type=int,default=2);p.add_argument('--warmup',type=int,default=3);p.add_argument('--iters',type=int,default=15);p.add_argument('--threads',type=int,default=4);a=p.parse_args();torch.set_num_threads(a.threads)
 inputs=[torch.randn(a.batch_size,c,h,w) for c,h,w in SHAPES]
 names=['original_dt1d_pointwise','original_dt1d_core','without_detail','offset4_only','offset8_only','raw_atoms','hosq_lite_c1_orth']
 result={'environment':{'torch':torch.__version__,'threads':a.threads,'batch_size':a.batch_size},'variants':{}}
 with torch.inference_mode():
  for name in names:
   mods=build(name)
   for _ in range(a.warmup):
    for m,x in zip(mods,inputs):m(x)
   samples=[]
   for _ in range(a.iters):
    t=time.perf_counter()
    for m,x in zip(mods,inputs):m(x)
    samples.append((time.perf_counter()-t)*1000/a.batch_size)
   result['variants'][name]={'median_ms_per_image':statistics.median(samples),'mean_ms_per_image':statistics.mean(samples),'parameters':sum(p.numel() for p in mods.parameters()),'conv_calls_per_adapter':8 if name=='original_dt1d_pointwise' else (6 if name=='original_dt1d_core' else 2)}
 base=result['variants']['original_dt1d_pointwise']['median_ms_per_image'];final=result['variants']['hosq_lite_c1_orth']['median_ms_per_image']
 result['final_vs_original_pointwise_latency_reduction_percent']=100*(base-final)/base
 out=ROOT/'outputs'/'hosq_lite_c1_validation';out.mkdir(parents=True,exist_ok=True);path=out/'latency.json';path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));print(f'Wrote {path}')
if __name__=='__main__':main()
