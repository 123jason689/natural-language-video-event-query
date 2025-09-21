import kornia as K
import kornia.filters as KF
import kornia.enhance as KE
import torch.nn.functional as F
import torch

def _as_BTCHW(x):
    if x.ndim == 5:  # (B,T,C,H,W)
        B,T,C,H,W = x.shape
        return x.view(B*T, C, H, W), (B,T)
    elif x.ndim == 4:  # (T,C,H,W)
        T,C,H,W = x.shape
        return x.view(T, C, H, W), (1, T)
    else:
        raise ValueError("Expected (T,C,H,W) or (B,T,C,H,W)")

def _back_from_BTCHW(x, bt_shape):
    B,T = bt_shape
    N,C,H,W = x.shape
    if B == 1:  # original was (T,C,H,W)
        return x.view(T, C, H, W)
    else:
        return x.view(B, T, C, H, W)

def _gdino_resize(x):  # x: (N,C,H,W)
    # Grounding DINO policy: short side = 800; long side <= 1333; keep aspect
    N,C,H,W = x.shape
    short, long = (H, W) if H < W else (W, H)
    scale = 800.0 / short
    if scale * long > 1333:
        scale = 1333.0 / long
    H2 = int(round(H * scale))
    W2 = int(round(W * scale))
    return KF.resize(x, (H2, W2), antialias=True)

def _ema_1d(params, t_axis=0, alpha=0.6):
    # params: (T, K) tensor -> EMA along time
    # alpha ~ 0.6 keeps it responsive without flicker
    y = params.clone()
    for t in range(1, y.shape[0]):
        y[t] = alpha * y[t] + (1 - alpha) * y[t-1]
    return y

@torch.inference_mode()
def adaptive_enhance_for_detection(video, device=None, out_dtype=torch.float16):
    """
    Adaptive, parameter-free preprocessing for CCTV-like footage before detection.
    Input:
      video: (T,C,H,W) or (B,T,C,H,W), uint8 or float [0,1]
    Output:
      enhanced, resized to Grounding-DINO policy; same leading dims as input
    """
    if device is None:
        device = video.device
    x, bt = _as_BTCHW(video)  # (N,C,H,W) where N=B*T
    N,C,H,W = x.shape

    # to float [0,1]
    if x.dtype == torch.uint8:
        x = x.to(device=device, dtype=torch.float32) / 255.0
    else:
        x = x.to(device=device)

    # --------- quick per-frame diagnostics (cheap & differentiable) ---------
    # Luminance & contrast
    ycbcr = K.color.rgb_to_ycbcr(x)
    Y = ycbcr[:, :1]  # (N,1,H,W)
    Y_mean = Y.mean(dim=(2,3))                     # (N,1)
    Y_std  = Y.std(dim=(2,3)) + 1e-8               # (N,1)

    # Sharpness proxy: variance of Laplacian
    lap = KF.laplacian(Y, kernel_size=3)
    sharp = lap.pow(2).mean(dim=(2,3))             # (N,1)

    # Color cast: gray-world deviation (per-channel means)
    means = x.mean(dim=(2,3))                      # (N,C)
    m_avg = means.mean(dim=1, keepdim=True)        # (N,1)
    wb_gain = (m_avg / (means + 1e-6)).clamp(0.6, 1.6)  # conservative clamp

    # Noise proxy: high-frequency energy (image - blur)
    blur = KF.gaussian_blur2d(x, (3,3), (1.0,1.0))
    hf = (x - blur)
    noise_level = hf.abs().mean(dim=(1,2,3), keepdim=True)  # (N,1)

    # --------- convert N to (B,T) to smooth parameters over time -----------
    B,T = bt
    # pack diagnostics per-frame: [Y_mean, Y_std, sharp, noise, r_gain, g_gain, b_gain]
    diag = torch.cat([
        Y_mean, Y_std, sharp, noise_level,
        wb_gain[:,0:1], wb_gain[:,1:2], wb_gain[:,2:3]
    ], dim=1).view(B, T, -1)

    # EMA over time to avoid flicker
    diag_smoothed = torch.stack([_ema_1d(diag[b]) for b in range(B)], dim=0)  # (B,T,K)
    # unpack
    Y_mean_s, Y_std_s, sharp_s, noise_s, r_g, g_g, b_g = [diag_smoothed[:,:,i] for i in range(7)]

    # --------- derive adaptive parameters (piecewise but smooth) -----------
    # Gamma: brighten if dark; compress if overly bright (rare in CCTV)
    gamma = torch.where(Y_mean_s < 0.22, torch.full_like(Y_mean_s, 0.70),
             torch.where(Y_mean_s < 0.35, torch.full_like(Y_mean_s, 0.80),
             torch.where(Y_mean_s < 0.55, torch.full_like(Y_mean_s, 0.90),
                         torch.full_like(Y_mean_s, 1.00))))
    # CLAHE clip limit: push when contrast is low
    clahe = torch.where(Y_std_s < 0.05, torch.full_like(Y_std_s, 3.5),
             torch.where(Y_std_s < 0.08, torch.full_like(Y_std_s, 2.5),
             torch.where(Y_std_s < 0.12, torch.full_like(Y_std_s, 1.8),
                         torch.full_like(Y_std_s, 1.2))))
    # Denoise kernel: higher noise -> bigger median kernel
    ksize = torch.where(noise_s > 0.06, torch.full_like(noise_s, 5.0),
             torch.where(noise_s > 0.035, torch.full_like(noise_s, 3.0),
                         torch.full_like(noise_s, 0.0)))  # 0 = skip
    # Unsharp amount: if sharpness low, increase; if noisy, be conservative
    base_sharp = torch.where(sharp_s < 0.002, 0.9,
                   torch.where(sharp_s < 0.006, 0.7, 0.5)).to(x.dtype)
    noise_penalty = (noise_s.squeeze(-1).clamp(0, 0.08) / 0.08) * 0.4  # reduce if noisy
    usm_amount = (base_sharp - noise_penalty).clamp(0.3, 1.0)  # (B,T)

    # White-balance gains (already smoothed & clamped)
    wb = torch.stack([r_g, g_g, b_g], dim=-1)  # (B,T,3)

    # --------- apply per-frame with time-smoothed params ----------
    x = x.view(B, T, C, H, W)

    # Denoise (median when requested)
    if ksize.max() > 0:
        # build mask for frames with k=3 or k=5 to keep it efficient
        def _apply_median(x_btchw, mask, k):
            if mask.any():
                idx = mask.nonzero(as_tuple=True)
                x_sel = x_btchw[idx]  # (N_sel, C, H, W)
                x_sel = KF.median_blur(x_sel, (int(k), int(k)))
                x_btchw[idx] = x_sel
            return x_btchw
        k3 = (ksize.squeeze(-1) == 3).view(B,T)
        k5 = (ksize.squeeze(-1) == 5).view(B,T)
        x = _apply_median(x, k3, 3)
        x = _apply_median(x, k5, 5)

    # White balance (per-frame channel gains)
    gains = wb.view(B, T, 3, 1, 1).to(x.dtype)
    x = (x * gains).clamp(0,1)

    # Gamma (per-frame)
    g = gamma.view(B, T, 1, 1, 1).to(x.dtype)
    x = (x.clamp(0,1) + 1e-8).pow(g.reciprocal().clamp(1.0, 2.0))  # adjust_gamma

    # CLAHE on luminance
    x_flat = x.view(B*T, C, H, W)
    ycbcr = K.color.rgb_to_ycbcr(x_flat)
    Y = ycbcr[:, :1]
    cl = clahe.view(B*T, 1)
    # Kornia's CLAHE takes a scalar clip_limit; loop in small chunks to keep mem low
    chunk = 128
    Ys = []
    for s in range(0, B*T, chunk):
        e = min(s+chunk, B*T)
        Ys.append(KE.equalize_clahe(
            Y[s:e], clip_limit=float(cl[s].item()), grid_size=(8,8)))
    Y2 = torch.cat(Ys, dim=0)
    x_flat = K.color.ycbcr_to_rgb(torch.cat([Y2, ycbcr[:,1:],], dim=1)).clamp(0,1)

    # Unsharp mask (per-frame amount)
    usm = usm_amount.view(B*T, 1)
    out = []
    for s in range(0, B*T, chunk):
        e = min(s+chunk, B*T)
        out.append(KE.unsharp_mask(
            x_flat[s:e], kernel_size=(5,5), sigma=(1.5,1.5),
            amount=float(usm[s].item()), threshold=0.0))
    x_flat = torch.cat(out, dim=0).clamp(0,1)

    # Grounding-DINO resize (antialias helps when shrinking)
    x_flat = _gdino_resize(x_flat)

    # back to original leading dims
    enhanced = _back_from_BTCHW(x_flat, bt)

    # optional: fp16 to save VRAM
    if out_dtype in (torch.float16, torch.bfloat16):
        enhanced = enhanced.to(out_dtype)

    return enhanced