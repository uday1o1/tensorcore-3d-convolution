import torch, torch.nn.functional as F

def im2win_conv2d(x, w, stride=1):
    '''Windowed convolution, 2D. Materializes H-axis overlap only (expansion k),
    then slides along W via conv1d. Faithful to im2win's layout.
    x: (B,C,H,W)  w: (Co,C,k,k)  -> (B,Co,oh,ow), valid padding.'''
    B, C, H, W = x.shape
    Co, _, k, k2 = w.shape
    assert k == k2
    oh = (H - k) // stride + 1
    # unfold H: (B,C,oh,W,k) -> window rows
    u = x.unfold(2, k, stride)                       # (B,C,oh,W,k)
    u = u.permute(0, 2, 1, 4, 3).contiguous()        # (B,oh,C,k,W)
    u = u.reshape(B * oh, C * k, W)                  # (B*oh, C*k, W)
    # filter: (Co,C,k,k) -> (Co, C*k, k) matching (c_kernelrow, channel) grouping
    w1 = w.reshape(Co, C * k, k)                     # (Co, C*k, kw), matches (c,kh) grouping
    o = F.conv1d(u, w1, stride=stride)               # (B*oh, Co, ow)
    ow = o.shape[-1]
    return o.reshape(B, oh, Co, ow).permute(0, 2, 1, 3).contiguous()

def im2win_conv3d(x, w, stride=1):
    '''Windowed convolution, 3D. Materializes D and H overlap (expansion k^2),
    slides along W via conv1d.
    x: (B,C,D,H,W)  w: (Co,C,k,k,k) -> (B,Co,od,oh,ow), valid padding.'''
    B, C, D, H, W = x.shape
    Co = w.shape[0]; k = w.shape[2]
    od = (D - k) // stride + 1
    oh = (H - k) // stride + 1
    u = x.unfold(2, k, stride).unfold(3, k, stride)   # (B,C,od,oh,W,kd,kh)
    u = u.permute(0, 2, 3, 1, 5, 6, 4).contiguous()   # (B,od,oh,C,kd,kh,W)
    u = u.reshape(B * od * oh, C * k * k, W)
    w1 = w.reshape(Co, C * k * k, k)
    o = F.conv1d(u, w1, stride=stride)
    ow = o.shape[-1]
    return o.reshape(B, od, oh, Co, ow).permute(0, 3, 1, 2, 4).contiguous()

def im2win_conv3d_minmat(x, w, stride=1):
    '''3D windowed conv, minimal materialization: unfold D only (expansion k),
    then slide H and W via conv2d. Closer in spirit to im2win 2D than the
    double-unfold variant.'''
    import torch.nn.functional as F
    B, C, D, H, W = x.shape
    Co = w.shape[0]; k = w.shape[2]
    od = (D - k) // stride + 1
    u = x.unfold(2, k, stride)                      # (B,C,od,H,W,kd)
    u = u.permute(0, 2, 1, 5, 3, 4).contiguous()    # (B,od,C,kd,H,W)
    u = u.reshape(B * od, C * k, H, W)
    w2 = w.reshape(Co, C * k, k, k)                 # (Co, C*kd, kh, kw)
    o = F.conv2d(u, w2, stride=stride)
    oh, ow = o.shape[-2], o.shape[-1]
    return o.reshape(B, od, Co, oh, ow).permute(0, 2, 1, 3, 4).contiguous()


def im2win_conv3d_depthwise(x, w, stride=1):
    '''3D windowed conv, depthwise (groups == channels).

    Same minimal materialization as im2win_conv3d_minmat: unfold D, slide H and
    W. The unfold orders the materialized channel axis as (c, kd), so the k
    channels belonging to input channel c are contiguous and a grouped conv2d
    with groups=C picks out exactly the right ones.

    This exists to test the method on the layer type large-kernel 3D
    architectures actually use. Depthwise reduces over one input channel rather
    than C of them, so the materialization is unchanged while the compute it
    feeds drops by a factor of C. The intensity account predicts this is the
    worst case for any materializing method, and that prediction should be
    measured rather than assumed.

    x: (B,C,D,H,W)  w: (C,1,k,k,k) -> (B,C,od,oh,ow), valid padding.'''
    B, C, D, H, W = x.shape
    assert w.shape[0] == C and w.shape[1] == 1, "depthwise expects (C,1,k,k,k)"
    k = w.shape[2]
    od = (D - k) // stride + 1
    u = x.unfold(2, k, stride)                      # (B,C,od,H,W,kd)
    u = u.permute(0, 2, 1, 5, 3, 4).contiguous()    # (B,od,C,kd,H,W)
    u = u.reshape(B * od, C * k, H, W)
    w2 = w.reshape(C, k, k, k)                      # (C_out, kd, kh, kw)
    o = F.conv2d(u, w2, stride=stride, groups=C)
    oh, ow = o.shape[-2], o.shape[-1]
    return o.reshape(B, od, C, oh, ow).permute(0, 2, 1, 3, 4).contiguous()


import torch.nn as nn

class WindowedConv3d(nn.Module):
    '''Drop-in replacement for nn.Conv3d using windowed convolution.
    Supports the subset nnU-Net uses: symmetric padding, square kernels, bias.
    Falls back to cuDNN for shapes the windowed path cannot serve
    (kernel_size 1, or grouped conv), matching the paper's stated scope.'''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        if isinstance(kernel_size,(tuple,list)): kernel_size=kernel_size[0]
        if isinstance(stride,(tuple,list)): stride=stride[0]
        if isinstance(padding,(tuple,list)): padding=padding[0]
        if isinstance(dilation,(tuple,list)): dilation=dilation[0]
        if isinstance(groups,(tuple,list)): groups=groups[0]
        self.in_channels=in_channels; self.out_channels=out_channels
        self.kernel_size=kernel_size; self.stride=stride; self.padding=padding
        self.groups=groups; self.dilation=dilation
        # windowed path serves dense and fully-depthwise conv, non-dilated,
        # kernel>1. Partially grouped conv (1 < groups < C) falls back.
        self.depthwise = (groups>1 and groups==in_channels==out_channels)
        self.use_windowed = (dilation==1 and kernel_size>1
                             and (groups==1 or self.depthwise))
        self.weight=nn.Parameter(torch.empty(out_channels,in_channels//groups,
                                             kernel_size,kernel_size,kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self.bias=nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self,x):
        if not self.use_windowed:
            return F.conv3d(x,self.weight,self.bias,self.stride,self.padding,
                            self.dilation,self.groups)
        if self.padding>0:
            p=self.padding; x=F.pad(x,(p,p,p,p,p,p))
        if self.depthwise:
            o=im2win_conv3d_depthwise(x,self.weight,stride=self.stride)
        else:
            o=im2win_conv3d_minmat(x,self.weight,stride=self.stride)
        if self.bias is not None:
            o=o+self.bias.view(1,-1,1,1,1)
        return o
