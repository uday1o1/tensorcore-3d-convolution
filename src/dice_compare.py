import torch, torch.nn as nn, numpy as np, pickle, copy, json, sys
from nnunet_mednext.network_architecture.generic_UNet import Generic_UNet
from winconv import WindowedConv3d
torch.backends.cudnn.benchmark=True

PRE='/workspace/nnUNet_preprocessed/Task003_Liver/nnUNetData_plans_v2.1_trgSp_1x1x1_stage1/'
CK='/workspace/RESULTS_FOLDER/nnUNet/3d_fullres/Task003_Liver/nnUNetTrainerV2_150ep__nnUNetPlansv2.1_trgSp_1x1x1/fold_0/model_best.model'
PATCH=(128,128,128)

def build():
    return Generic_UNet(1,30,3,5,2,2, nn.Conv3d, nn.InstanceNorm3d, {'eps':1e-5,'affine':True},
        nn.Dropout3d, {'p':0,'inplace':True}, nn.LeakyReLU, {'negative_slope':1e-2,'inplace':True},
        False, False, lambda x:x, None, [[2,2,2]]*5, [[3,3,3]]*6, False, True, True).cuda().eval()

def swap(m):
    n=0
    for _,mod in m.named_modules():
        for cn,ch in list(mod.named_children()):
            if isinstance(ch, nn.Conv3d):
                w=WindowedConv3d(ch.in_channels,ch.out_channels,ch.kernel_size,ch.stride,
                                 ch.padding,ch.dilation,ch.groups,ch.bias is not None).cuda()
                w.weight.data=ch.weight.data.clone()
                if ch.bias is not None: w.bias.data=ch.bias.data.clone()
                setattr(mod,cn,w); n+= 1 if w.use_windowed else 0
    return n

@torch.no_grad()
def sliding(model, vol):
    '''identical procedure for both kernels: non-overlapping-ish tiling with edge clamp'''
    C,D,H,W = vol.shape
    pd,ph,pw = PATCH
    D2,H2,W2 = max(D,pd), max(H,ph), max(W,pw)
    pad = (0,W2-W, 0,H2-H, 0,D2-D)
    x = torch.nn.functional.pad(vol.unsqueeze(0), pad)
    out = torch.zeros((1,3,D2,H2,W2), device='cuda', dtype=torch.float32)
    cnt = torch.zeros((1,1,D2,H2,W2), device='cuda', dtype=torch.float32)
    step = [p//2 for p in PATCH]
    zs=list(range(0,max(D2-pd,0)+1,step[0])) or [0]
    ys=list(range(0,max(H2-ph,0)+1,step[1])) or [0]
    xs=list(range(0,max(W2-pw,0)+1,step[2])) or [0]
    if zs[-1]!=D2-pd: zs.append(D2-pd)
    if ys[-1]!=H2-ph: ys.append(H2-ph)
    if xs[-1]!=W2-pw: xs.append(W2-pw)
    for z in zs:
        for y in ys:
            for xx in xs:
                patch = x[:,:,z:z+pd, y:y+ph, xx:xx+pw].cuda()
                with torch.autocast('cuda',dtype=torch.float16):
                    p = model(patch)
                out[:,:,z:z+pd,y:y+ph,xx:xx+pw] += p.float()
                cnt[:,:,z:z+pd,y:y+ph,xx:xx+pw] += 1
    out = out/cnt.clamp(min=1)
    return out[:,:,:D,:H,:W].argmax(1)[0]

def dice(pred, gt, cls):
    p=(pred==cls); g=(gt==cls)
    inter=(p&g).sum().item(); s=p.sum().item()+g.sum().item()
    return (2*inter/s) if s>0 else float('nan')

split=pickle.load(open('/workspace/nnUNet_preprocessed/Task003_Liver/splits_final.pkl','rb'))
val=[str(v) for v in split[0]['val']]
ck=torch.load(CK,map_location='cpu',weights_only=False)['state_dict']

ref=build(); ref.load_state_dict(ck)
win=copy.deepcopy(ref); nsw=swap(win)
print(f'windowed layers active: {nsw}', flush=True)

rows=[]
N=int(sys.argv[1]) if len(sys.argv)>1 else len(val)
for i,case in enumerate(val[:N]):
    d=np.load(PRE+case+'.npz')['data']
    vol=torch.from_numpy(d[:-1]).float().cuda()
    gt=torch.from_numpy(d[-1]).cuda()
    pr=sliding(ref,vol); pw_=sliding(win,vol)
    agree=(pr==pw_).float().mean().item()
    r={'case':case,
       'cudnn_liver':dice(pr,gt,1),'cudnn_tumor':dice(pr,gt,2),
       'win_liver':dice(pw_,gt,1),'win_tumor':dice(pw_,gt,2),
       'voxel_agreement':agree}
    rows.append(r)
    print(f"{i+1}/{N} {case}: cuDNN L{r['cudnn_liver']:.4f} T{r['cudnn_tumor']:.4f} | win L{r['win_liver']:.4f} T{r['win_tumor']:.4f} | agree {agree:.6f}", flush=True)
    del vol,gt,pr,pw_; torch.cuda.empty_cache()
json.dump(rows, open('/workspace/win3d/dice_results.json','w'), indent=1)
print('saved')
