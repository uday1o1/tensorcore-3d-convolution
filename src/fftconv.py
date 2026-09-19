import torch, torch.nn.functional as F

def fft_conv3d(x, w, stride=1):
    '''FFT-based 3D convolution, valid padding, stride 1 only.
    Cost is O(n log n) in the volume and INDEPENDENT of kernel size,
    which is the property that should make it win at large k.'''
    assert stride == 1, 'FFT path implemented for stride 1'
    B, C, D, H, W = x.shape
    Co, _, kd, kh, kw = w.shape
    # linear (not circular) convolution needs padding to at least N+K-1
    sd, sh, sw = D + kd - 1, H + kh - 1, W + kw - 1
    Xf = torch.fft.rfftn(x.float(), s=(sd, sh, sw), dim=(2,3,4))          # (B,C,.,.,.)
    # correlation, not convolution: flip the kernel
    wf = torch.flip(w.float(), dims=(2,3,4))
    Wf = torch.fft.rfftn(wf, s=(sd, sh, sw), dim=(2,3,4))                 # (Co,C,.,.,.)
    # sum over input channels: (B,1,C,...) * (1,Co,C,...) -> (B,Co,...)
    out = torch.einsum('bcxyz,ocxyz->boxyz', Xf, Wf)
    y = torch.fft.irfftn(out, s=(sd, sh, sw), dim=(2,3,4))
    # valid region of a correlation starts at (k-1)
    return y[:, :, kd-1:kd-1+D-kd+1, kh-1:kh-1+H-kh+1, kw-1:kw-1+W-kw+1].contiguous()
