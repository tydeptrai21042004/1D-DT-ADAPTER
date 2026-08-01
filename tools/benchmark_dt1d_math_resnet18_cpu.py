"""Controlled end-to-end ResNet-18 timing with only DT1D realization changed."""
from __future__ import annotations
import argparse, importlib.util, json, statistics, time
from pathlib import Path
import torch
from torchvision.models import resnet18

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('dt1d_math_resnet', ROOT/'models'/'dt1d_adapter.py')
MOD=importlib.util.module_from_spec(SPEC); assert SPEC.loader is not None; SPEC.loader.exec_module(MOD)
DT1DAdapter=MOD.DT1DAdapter

class Model(torch.nn.Module):
    def __init__(self, optimized: bool, classes=10):
        super().__init__()
        self.net=resnet18(weights=None)
        self.net.fc=torch.nn.Linear(512, classes)
        channels=[block.conv2.out_channels for layer in (self.net.layer1,self.net.layer2,self.net.layer3,self.net.layer4) for block in layer]
        self.adapters=torch.nn.ModuleList([
            DT1DAdapter(C=c,M=1,dilations=(1,2,4),scale_adaptive=True,axis='hw',alpha_group=16,
                        no_pw=False,pw_ratio=32,pw_groups=4,gate_init=0.2,padding_mode='reflect',
                        separate_axis_kernels=True,exact_cost_realization=optimized)
            for c in channels])
    def forward(self,x):
        n=self.net
        x=n.maxpool(n.relu(n.bn1(n.conv1(x))))
        idx=0
        for layer in (n.layer1,n.layer2,n.layer3,n.layer4):
            for block in layer:
                x=self.adapters[idx](block(x)); idx+=1
        return n.fc(torch.flatten(n.avgpool(x),1))

def samples(fn,warmup,iterations,repeats):
    for _ in range(warmup): fn()
    vals=[]
    for _ in range(repeats):
        t=time.perf_counter()
        for _ in range(iterations): fn()
        vals.append((time.perf_counter()-t)*1000/iterations)
    return {'median_ms':statistics.median(vals),'samples_ms':vals,'min_ms':min(vals),'max_ms':max(vals)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='benchmark_dt1d_math_resnet18_cpu.json'); ap.add_argument('--threads',type=int,default=4); ap.add_argument('--batch-size',type=int,default=2); ap.add_argument('--input-size',type=int,default=224); args=ap.parse_args()
    torch.set_num_threads(args.threads); torch.manual_seed(77)
    before=Model(False); after=Model(True); after.load_state_dict(before.state_dict(), strict=True)
    for model in (before,after):
        for name,p in model.named_parameters(): p.requires_grad=('adapters' in name or 'net.fc' in name)
    x=torch.randn(args.batch_size,3,args.input_size,args.input_size); target=torch.randint(0,10,(args.batch_size,))
    before.eval(); after.eval()
    with torch.inference_mode():
        lb=before(x); la=after(x)
        tb=samples(lambda:before(x),3,5,7); ta=samples(lambda:after(x),3,5,7)
    before.train(); after.train()
    def step(m):
        m.zero_grad(set_to_none=True); torch.nn.functional.cross_entropy(m(x),target).backward()
    sb=samples(lambda:step(before),1,2,5); sa=samples(lambda:step(after),1,2,5)
    out={
      'fairness_controls': {'same_model_class':True,'same_weights':True,'same_backbone':True,'same_pointwise':True,'cache_used':False,'only_switch':'exact_cost_realization'},
      'torch_version':torch.__version__,'device':'cpu','threads':torch.get_num_threads(),'batch_size':args.batch_size,'input_size':args.input_size,
      'max_logit_abs_difference':float((lb-la).abs().max()),'top1_agreement':float((lb.argmax(1)==la.argmax(1)).float().mean()),
      'before_inference':tb,'after_inference':ta,'inference_speedup':tb['median_ms']/ta['median_ms'],
      'before_forward_backward':sb,'after_forward_backward':sa,'forward_backward_speedup':sb['median_ms']/sa['median_ms']}
    Path(args.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
