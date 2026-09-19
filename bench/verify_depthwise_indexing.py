"""Validates the depthwise windowed indexing scheme without torch.

Replicates im2win_conv3d_depthwise's transform in numpy and compares against a
naive depthwise conv3d. The risk being tested is the channel grouping: the
unfold must order the materialized axis as (c, kd) so that a grouped conv2d
with groups=C, which consumes input channels [c*k, (c+1)*k), picks out exactly
channel c's kd slices. Getting (kd, c) instead is silently wrong, which is how
the dense path broke earlier in this project.
"""
import numpy as np
rng = np.random.default_rng(0)


def naive_dw(x, w, s=1):
    B, C, D, H, W = x.shape
    k = w.shape[2]
    od, oh, ow = [(n - k) // s + 1 for n in (D, H, W)]
    o = np.zeros((B, C, od, oh, ow))
    for b in range(B):
        for c in range(C):
            for i in range(od):
                for j in range(oh):
                    for l in range(ow):
                        o[b, c, i, j, l] = np.sum(
                            x[b, c, i*s:i*s+k, j*s:j*s+k, l*s:l*s+k] * w[c, 0])
    return o


def windowed_dw(x, w, s=1):
    """Mirrors winconv.im2win_conv3d_depthwise exactly."""
    B, C, D, H, W = x.shape
    k = w.shape[2]
    od = (D - k) // s + 1
    # x.unfold(2,k,s) then permute(0,2,1,5,3,4) -> (B,od,C,kd,H,W)
    u = np.stack([x[:, :, i*s:i*s+k] for i in range(od)], axis=2)  # (B,C,od,kd,H,W)
    u = u.transpose(0, 2, 1, 3, 4, 5)                              # (B,od,C,kd,H,W)
    u = u.reshape(B * od, C * k, H, W)
    w2 = w.reshape(C, k, k, k)                                     # (Cout,kd,kh,kw)
    oh = (H - k) // s + 1
    ow = (W - k) // s + 1
    o = np.zeros((B * od, C, oh, ow))
    for c in range(C):
        grp = u[:, c*k:(c+1)*k]                                    # groups=C slice
        for j in range(oh):
            for l in range(ow):
                o[:, c, j, l] = np.einsum(
                    'nahw,ahw->n', grp[:, :, j*s:j*s+k, l*s:l*s+k], w2[c])
    return o.reshape(B, od, C, oh, ow).transpose(0, 2, 1, 3, 4)


ok = True
for B, C, D, k, s in [(2, 5, 8, 3, 1), (1, 4, 9, 3, 2), (2, 3, 10, 5, 1), (1, 6, 11, 3, 1)]:
    x = rng.standard_normal((B, C, D, D, D))
    w = rng.standard_normal((C, 1, k, k, k))
    a, b = windowed_dw(x, w, s), naive_dw(x, w, s)
    e = np.abs(a - b).max() / np.abs(b).max()
    print(f"B{B} C{C} D{D} k{k} s{s}: shape {a.shape} == {b.shape} rel {e:.2e}")
    ok &= a.shape == b.shape and e < 1e-12

# The wrong grouping must actually produce a different buffer, or this test
# would pass for the wrong reason.
B, C, D, k = 1, 4, 8, 3
x = rng.standard_normal((B, C, D, D, D))
st = np.stack([x[:, :, i:i+k] for i in range(D - k + 1)], axis=2)
right = st.transpose(0, 2, 1, 3, 4, 5).reshape(B * (D-k+1), C*k, D, D)
wrong = st.transpose(0, 2, 3, 1, 4, 5).reshape(B * (D-k+1), C*k, D, D)
distinguishes = not np.allclose(right, wrong)
print(f"\n(c,kd) ordering is distinguishable from (kd,c): {distinguishes}")
ok &= distinguishes

print("ALL CORRECT" if ok else "FAILURES")
