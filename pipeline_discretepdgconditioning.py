#!/usr/bin/env python3
"""
Unified particle generation pipeline with DDPM.

Modes:
    train    - Train the diffusion model
    sample   - Generate synthetic events from trained model
    evaluate - Compare real vs generated distributions

Usage:
    python particle_diffusion.py train --data_path mc_gen1.npy --outdir results
    python particle_diffusion.py sample --outdir results --n_events 1000
    python particle_diffusion.py evaluate --real_path mc_gen1.npy --gen_path results/generated_events.npy
"""

from asyncio import events
import os
import math
import argparse
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.colors as mcolors





# ============================================================
# UTILITIES
# ============================================================
def set_seed(seed: int):
    """Make runs reproducible."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices(n, val_frac, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(val_frac * n))
    return idx[n_val:], idx[:n_val]


def make_linear_beta_schedule(T: int, beta_start: float, beta_end: float, device: str):
    """Create DDPM linear noise schedule."""
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    acp = torch.cumprod(alphas, dim=0)
    acp_prev = torch.cat([torch.ones(1, device=device), acp[:-1]])
    return betas, alphas, acp, acp_prev

def beta_to_w(beta_xyz: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    beta (N,3) with ||beta||<1  -> w (N,3) in R^3
    Inverse of w_to_beta using: beta = w/(1+||w||)
    """
    beta = np.asarray(beta_xyz, dtype=np.float32)
    bmag = np.linalg.norm(beta, axis=1, keepdims=True)
    # Clamp bmag into [0, 1-eps] to avoid blow-ups and sign issues
    bmag = np.clip(bmag, 0.0, 1.0 - eps)

    scale = 1.0 / (1.0 - bmag)
    return beta * scale

def w_to_beta(w_xyz: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    w (N,3) in R^3 -> beta (N,3) with ||beta||<1
    """
    w = np.asarray(w_xyz, dtype=np.float32)
    wmag = np.linalg.norm(w, axis=1, keepdims=True)
    return w / (1.0 + np.maximum(wmag, eps))


def torch_w_to_beta(w: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # w: (B,K,3)
    wmag = torch.linalg.norm(w, dim=-1, keepdim=True)  # (B,K,1)
    return w / (1.0 + torch.clamp(wmag, min=eps))

def torch_beta_mag_from_w(w: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    beta = torch_w_to_beta(w, eps=eps)
    return torch.linalg.norm(beta, dim=-1)  # (B,K)


# ============================================================
# CONFIG
# ============================================================
@dataclass
class CFG:
    data_path: str = "guineapig_raw_trimmed.npy"
    outdir: str = "new_8"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    max_particles: int = 1300
    min_particles: int = 1
    keep_fraction: float = 1.0
    
    T: int = 200
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    
    d_model = 64
    nhead = 4
    num_layers = 2
    dropout = 0.1
    
    batch_size: int = 4
    lr: float = 2e-4
    epochs: int = 15
    num_workers: int = 4
    grad_clip: float = 1.0
    seed: int = 123

    frac_range = 0.80

    feat_dim: int = 7
    me: float = 0.00051099895069 # GeV

    lambda_mass: float = 0.1   # start small; tune 0.01–1.0

    lambda_sign: float = 0.05   # try 0.01–0.2
    sign_margin: float = 0.0    # 0 is fine; >0 makes it stricter



# ============================================================
# DATASET
# ============================================================
class MCPDataset(Dataset):
    """
    Diffuses 7 features directly, WITHOUT PDG conditioning:
      x = [E_signed, betax, betay, betaz, x, y, z]

    Uses a single global normalisation (mean/std per feature).
    """
    def __init__(self, path, max_particles=64, min_particles=1, keep_fraction=1.0):
        raw = np.load(path, allow_pickle=True)

        if keep_fraction < 1.0:
            raw = raw[: int(len(raw) * keep_fraction)]

        events = []
        for ev in raw:
            if ev is None:
                continue
            ev = np.asarray(ev)
            if ev.ndim != 2 or ev.shape[1] < 7:
                continue
            if len(ev) < min_particles:
                continue

            ev = ev.astype(np.float32)

            # Convert beta -> w ONCE and store it
            beta = ev[:, 1:4]
            ev[:, 1:4] = beta_to_w(beta)

            events.append(ev)


        if len(events) == 0:
            raise RuntimeError("No events left after filtering")

        self.events = events
        self.max_particles = max_particles
        self.feat_dim = 7

        all_feats = np.concatenate([ev[:, :7] for ev in events], axis=0)


        self.feat_mean = all_feats.mean(axis=0).astype(np.float32)
        self.feat_std  = all_feats.std(axis=0).astype(np.float32)
        self.feat_std  = np.maximum(self.feat_std, 1e-6)

        self.multiplicities = np.array([len(ev) for ev in events], dtype=np.int64)

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        ev = self.events[idx]
        N = int(len(ev))
        Kmax = self.max_particles

        if N <= Kmax:
            chosen = np.arange(N)
        else:
            chosen = torch.randperm(N)[:Kmax].numpy()

        rows = ev[chosen]          # (K, >=7)
        K = rows.shape[0]

        feats = rows[:, :7].astype(np.float32)
        feats_norm = (feats - self.feat_mean) / self.feat_std



        x0   = np.zeros((Kmax, self.feat_dim), dtype=np.float32)
        mask = np.zeros((Kmax,), dtype=np.bool_)

        x0[:K] = feats_norm
        mask[:K] = True

        return torch.from_numpy(x0), torch.from_numpy(mask)




# ============================================================
# MODEL COMPONENTS
# ============================================================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
    
    def forward(self, t: torch.Tensor):
        device = t.device
        half = self.d_model // 2
        
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, device=device).float() / max(half - 1, 1)
        )

        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class ParticleDenoiser(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.time_emb = SinusoidalTimeEmbedding(d_model)
        self.mom_proj = nn.Linear(7, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output = nn.Linear(d_model, 7)

        self.t_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # NEW: multiplicity conditioning MLP
        self.k_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.skip_alpha = nn.Parameter(torch.tensor(0.2))   # start < 1

    def forward(self, x_t, t, mask):
        B, K, _ = x_t.shape

        t_emb = self.time_emb(t)          # (B, d_model)
        t_emb = self.t_mlp(t_emb)         # (B, d_model)
        t_emb = t_emb.unsqueeze(1).expand(B, K, self.d_model)

        mom_emb = self.mom_proj(x_t)      # (B, K, d_model)

        h = t_emb + mom_emb

        # NEW: multiplicity conditioning from mask
        K_event = mask.sum(dim=1)                                  # (B,)
        k = torch.log(K_event.float().clamp(min=1)).unsqueeze(-1)   # (B,1)
        k_emb = self.k_mlp(k).unsqueeze(1)                          # (B,1,d_model)
        h = h + k_emb                                               # broadcast to (B,K,d_model)


        src_key_padding_mask = ~mask

        h_in = h
        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)
        h = h + self.skip_alpha * h_in
        h = h * mask.unsqueeze(-1)


        eps_hat = self.output(h)
        eps_hat = eps_hat * mask.unsqueeze(-1)
        return eps_hat


        


# ============================================================
# DDPM WRAPPER
# ============================================================
class DDPM:
    def __init__(self, model, T, beta_start, beta_end, device):
        self.model = model
        self.T = T
        self.device = device
        
        betas, alphas, acp, acp_prev = make_linear_beta_schedule(T, beta_start, beta_end, device)
        
        self.betas = betas
        self.alphas = alphas
        self.acp = acp
        self.acp_prev = acp_prev
        
        self.sqrt_acp = torch.sqrt(acp)
        self.sqrt_1m_acp = torch.sqrt(1.0 - acp)
        self.posterior_variance = betas * (1.0 - acp_prev) / (1.0 - acp)
    
    def q_sample(self, x0, t, noise):
        B = x0.shape[0]
        a = self.sqrt_acp[t].view(B, 1, 1)
        b = self.sqrt_1m_acp[t].view(B, 1, 1)
        return a * x0 + b * noise
    
    def p_sample(self, x_t, t, mask):
        B = x_t.shape[0]

        eps_hat = self.model(x_t, t, mask)

        beta_t  = self.betas[t].view(B, 1, 1)
        alpha_t = self.alphas[t].view(B, 1, 1)
        acp_t   = self.acp[t].view(B, 1, 1)

        mu = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1.0 - acp_t)) * eps_hat)
        var = self.posterior_variance[t].view(B, 1, 1)

        if t[0].item() == 0:
            z = torch.zeros_like(x_t)
        else:
            z = torch.randn_like(x_t)

        x_prev = mu + torch.sqrt(var) * z
        return x_prev * mask.unsqueeze(-1)

    
    @torch.no_grad()
    def sample(self, mask):
        B, K = mask.shape
        x = torch.randn((B, K, 7), device=self.device) * mask.unsqueeze(-1)

        for ti in reversed(range(self.T)):
            t = torch.full((B,), ti, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t, mask)

        return x



# ============================================================
# TRAINING
# ============================================================
def train(args):
    cfg = CFG()

    print("Using device:", cfg.device)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

    
    if args.data_path:
        cfg.data_path = args.data_path
    if args.outdir:
        cfg.outdir = args.outdir
    if args.max_particles:
        cfg.max_particles = args.max_particles
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.T:
        cfg.T = args.T
    if args.seed:
        cfg.seed = args.seed
    
    set_seed(cfg.seed)
    os.makedirs(cfg.outdir, exist_ok=True)
    
    val_frac = 0.1
    
    ds_full = MCPDataset(
        cfg.data_path,
        max_particles=cfg.max_particles,
        min_particles=cfg.min_particles,
        keep_fraction=cfg.keep_fraction,
    )
    
    train_idx, val_idx = split_indices(len(ds_full), val_frac, cfg.seed)
    
    ds_train = torch.utils.data.Subset(ds_full, train_idx)
    ds_val   = torch.utils.data.Subset(ds_full, val_idx)
    
    dl_train = DataLoader(
        ds_train, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
    )
    
    dl_val = DataLoader(
        ds_val, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
    )
    
    model = ParticleDenoiser(
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(cfg.device)

    
    ddpm = DDPM(model, cfg.T, cfg.beta_start, cfg.beta_end, cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)


    feat_mean_t = torch.as_tensor(ds_full.feat_mean, device=cfg.device).view(1, 1, 7)
    feat_std_t  = torch.as_tensor(ds_full.feat_std,  device=cfg.device).view(1, 1, 7)

    
    meta = {
        "multiplicities": ds_full.multiplicities,
        "feat_mean": ds_full.feat_mean,
        "feat_std": ds_full.feat_std,
        "feat_dim": ds_full.feat_dim,
        "me": cfg.me,


        "max_particles": cfg.max_particles,
        "T": cfg.T,
        "beta_start": cfg.beta_start,
        "beta_end": cfg.beta_end,
        "d_model": cfg.d_model,
        "nhead": cfg.nhead,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
        "val_frac": val_frac,
        "seed": cfg.seed,
        "n_events": len(ds_full),
        "n_train_events": len(train_idx),
        "n_val_events": len(val_idx),
    }
    torch.save(meta, os.path.join(cfg.outdir, "meta.pt"))
    
    train_losses = []
    val_losses = []
    
    for epoch in range(cfg.epochs):
        # TRAIN
        model.train()
        total_train = 0.0
        n_train = 0
        
        pbar = tqdm(dl_train, desc=f"Epoch {epoch+1:03d}/{cfg.epochs} [train]", leave=False)
        for x0, mask in pbar:
            x0   = x0.to(cfg.device)
            mask = mask.to(cfg.device)

            B = x0.shape[0]
            t = torch.randint(0, cfg.T, (B,), device=cfg.device)

            noise = torch.randn_like(x0) * mask.unsqueeze(-1)
            x_t = ddpm.q_sample(x0, t, noise)

            eps_hat = model(x_t, t, mask)


            # --- reconstruct x0 estimate from predicted noise ---
            # x0_hat = (x_t - sqrt(1-acp_t)*eps_hat) / sqrt(acp_t)
            sqrt_acp_t    = ddpm.sqrt_acp[t].view(B, 1, 1)
            sqrt_1m_acp_t = ddpm.sqrt_1m_acp[t].view(B, 1, 1)
            x0_hat = (x_t - sqrt_1m_acp_t * eps_hat) / torch.clamp(sqrt_acp_t, min=1e-12)

            # de-normalise to physical feature space
            feats_hat = x0_hat * feat_std_t + feat_mean_t  # (B,K,7)

            # true signed energy (physical space)
            feats_true = x0 * feat_std_t + feat_mean_t
            E_true = feats_true[..., 0]          # (B,K)

            # predicted signed energy
            E_hat = feats_hat[..., 0]            # (B,K)

            # target: 1 for positive, 0 for negative
            y_sign = (E_true > 0).float()

            # classify sign using E_hat as the logit
            sign_loss = F.binary_cross_entropy_with_logits(E_hat, y_sign, reduction="none")
            sign_loss = sign_loss[mask].mean() if mask.any() else torch.tensor(0.0, device=cfg.device)



            E_signed = feats_hat[..., 0]
            Eabs = torch.clamp(E_signed.abs(), min=cfg.me)  # treat as total energy magnitude

            w = feats_hat[..., 1:4]                         # (B,K,3)
            beta = torch_w_to_beta(w)                       # (B,K,3)
            beta2 = (beta ** 2).sum(dim=-1)                 # (B,K)

            # mass-shell: m^2 = E^2 (1 - beta^2)
            m2_hat = (Eabs ** 2) * torch.clamp(1.0 - beta2, min=0.0)
            m2_true = (cfg.me ** 2)

            # only apply on real particles
            mse_m = ((m2_hat - m2_true) / (m2_true + 1e-12)) ** 2 #can be huge at high energies, make it relative so it doesn't dominate
            mass_loss = mse_m[mask].mean() if mask.any() else torch.tensor(0.0, device=cfg.device)

            # your original diffusion loss
            mse = (eps_hat - noise).pow(2).sum(dim=-1)
            diff_loss = mse[mask].mean() if mask.any() else torch.tensor(0.0, device=cfg.device)

            loss = diff_loss + cfg.lambda_mass * mass_loss + cfg.lambda_sign * sign_loss


            '''        
            mse = (eps_hat - noise).pow(2).sum(dim=-1)
            masked = mse[mask]
            loss = masked.mean() if masked.numel() > 0 else torch.tensor(0.0, device=cfg.device)
            '''

            
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            
            total_train += loss.item()
            n_train += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        
        train_loss = total_train / max(n_train, 1)
        
        # VALIDATION
        model.eval()
        total_val = 0.0
        n_val = 0
        
        with torch.no_grad():
            pbarv = tqdm(dl_val, desc=f"Epoch {epoch+1:03d}/{cfg.epochs} [val]", leave=False)
            for x0, mask in pbarv:
                x0   = x0.to(cfg.device)
                mask = mask.to(cfg.device)

                B = x0.shape[0]
                t = torch.randint(0, cfg.T, (B,), device=cfg.device)

                noise = torch.randn_like(x0) * mask.unsqueeze(-1)
                x_t = ddpm.q_sample(x0, t, noise)

                eps_hat = model(x_t, t, mask)



                # --- reconstruct x0 estimate from predicted noise ---
                # x0_hat = (x_t - sqrt(1-acp_t)*eps_hat) / sqrt(acp_t)
                sqrt_acp_t    = ddpm.sqrt_acp[t].view(B, 1, 1)
                sqrt_1m_acp_t = ddpm.sqrt_1m_acp[t].view(B, 1, 1)
                x0_hat = (x_t - sqrt_1m_acp_t * eps_hat) / torch.clamp(sqrt_acp_t, min=1e-12)

                # de-normalise to physical feature space
                feats_hat = x0_hat * feat_std_t + feat_mean_t  # (B,K,7)

                # true signed energy (physical space)
                feats_true = x0 * feat_std_t + feat_mean_t
                E_true = feats_true[..., 0]          # (B,K)

                # predicted signed energy
                E_hat = feats_hat[..., 0]            # (B,K)

                # target: 1 for positive, 0 for negative
                y_sign = (E_true > 0).float()

                # classify sign using E_hat as the logit
                sign_loss = F.binary_cross_entropy_with_logits(E_hat, y_sign, reduction="none")
                sign_loss = sign_loss[mask].mean() if mask.any() else torch.tensor(0.0, device=cfg.device)



                E_signed = feats_hat[..., 0]
                Eabs = torch.clamp(E_signed.abs(), min=cfg.me)  # treat as total energy magnitude

                w = feats_hat[..., 1:4]                         # (B,K,3)
                beta = torch_w_to_beta(w)                       # (B,K,3)
                beta2 = (beta ** 2).sum(dim=-1)                 # (B,K)

                # mass-shell: m^2 = E^2 (1 - beta^2)
                m2_hat = (Eabs ** 2) * torch.clamp(1.0 - beta2, min=0.0)
                m2_true = (cfg.me ** 2)

                # only apply on real particles
                mse_m = ((m2_hat - m2_true) / (m2_true + 1e-12)) ** 2 #can be huge at high energies, make it relative so it doesn't dominate
                mass_loss = mse_m[mask].mean() if mask.any() else torch.tensor(0.0, device=cfg.device)

                # your original diffusion loss
                mse = (eps_hat - noise).pow(2).sum(dim=-1)
                diff_loss = mse[mask].mean() if mask.any() else torch.tensor(0.0, device=cfg.device)

                loss = diff_loss + cfg.lambda_mass * mass_loss + cfg.lambda_sign * sign_loss
                

                '''
                mse = (eps_hat - noise).pow(2).sum(dim=-1)
                masked = mse[mask]                          # (N_kept,)
                loss = masked.mean() if masked.numel() > 0 else torch.tensor(0.0, device=cfg.device)
                '''




                
                total_val += loss.item()
                n_val += 1

                pbarv.set_postfix(loss=f"{loss.item():.4f}")

        
        val_loss = total_val / max(n_val, 1)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1:03d}/{cfg.epochs} | train={train_loss:.6f} | val={val_loss:.6f}")
        
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            os.path.join(cfg.outdir, "ckpt_last.pt"),
        )
    
    np.save(os.path.join(cfg.outdir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(cfg.outdir, "val_losses.npy"), np.array(val_losses))
    
    print("Training complete. Outputs saved to:", cfg.outdir)


# ============================================================
# SAMPLING
# ============================================================
def load_meta_and_model(outdir: str, device: str):
    meta_path = os.path.join(outdir, "meta.pt")
    meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    
    model = ParticleDenoiser(
        d_model=meta["d_model"],
        nhead=meta["nhead"],
        num_layers=meta["num_layers"],
        dropout=meta["dropout"],
    ).to(device)

    
    ckpt_path = os.path.join(outdir, "ckpt_last.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ddpm = DDPM(
        model=model,
        T=int(meta["T"]),
        beta_start=float(meta["beta_start"]),
        beta_end=float(meta["beta_end"]),
        device=device,
    )

    
    return meta, model, ddpm


def sample_one_event(meta: dict, ddpm: DDPM, device: str):
    multiplicities = np.asarray(meta["multiplicities"], dtype=np.int64)
    K = int(np.random.choice(multiplicities))

    Kmax = int(meta["max_particles"])
    K = max(1, min(K, Kmax))

    # mask only (no PDGs)
    mask = np.zeros((Kmax,), dtype=np.bool_)
    mask[:K] = True
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)

    with torch.no_grad():
        x_norm = ddpm.sample(mask_t)[0].detach().cpu().numpy()  # (Kmax,7)
    x_norm = x_norm[:K]  # (K,7)

    mean = np.asarray(meta["feat_mean"], dtype=np.float32)
    std  = np.asarray(meta["feat_std"], dtype=np.float32)
    feats = x_norm * std + mean   # (K,7)

    w = feats[:, 1:4].astype(np.float32)
    beta = w_to_beta(w)
    feats[:, 1:4] = beta


    # just return the de-normalised model output
    event = feats.astype(np.float32, copy=False)
    return event






def sample(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = args.outdir
    n_events = args.n_events
    
    meta, model, ddpm = load_meta_and_model(outdir, device)
    
    from tqdm import trange
    events = []
    for _ in trange(n_events, desc="Generating events"):
        events.append(sample_one_event(meta, ddpm, device))
    
    out_path = os.path.join(outdir, "generated_events.npy")
    np.save(out_path, np.array(events, dtype=object))
    print("Saved:", out_path)


# ============================================================
# EVALUATION
# ============================================================
try:
    from scipy.stats import wasserstein_distance
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ---- Corner plots (optional dependency) ----
try:
    import corner as _corner
    HAVE_CORNER = True
except Exception:
    HAVE_CORNER = False


def _subsample_rows(X, max_points=40000, seed=123):
    X = np.asarray(X, dtype=np.float64)
    X = X[np.all(np.isfinite(X), axis=1)]
    if X.shape[0] <= max_points:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=max_points, replace=False)
    return X[idx]


def _clip_cols_quantile(X, q_lo=0.01, q_hi=0.99):
    """Clip each column to [q_lo, q_hi] to stop huge tails wrecking the corner plot."""
    X = np.asarray(X, dtype=np.float64)
    out = X.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        lo = np.quantile(col, q_lo)
        hi = np.quantile(col, q_hi)
        out[:, j] = np.clip(col, lo, hi)
    return out

def clamp_pairs_to_ranges(x, y, xlo, xhi, ylo, yhi):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    m2 = (x >= xlo) & (x <= xhi) & (y >= ylo) & (y <= yhi)
    return x[m2], y[m2]


def robust_range_1d(z, q_lo=0.01, q_hi=0.99):
    z = np.asarray(z, dtype=np.float64)
    z = z[np.isfinite(z)]
    if z.size < 2:
        return None
    lo = float(np.quantile(z, q_lo))
    hi = float(np.quantile(z, q_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return None
    return (lo, hi)


def plot_corner_overlay(
    real_dict, gen_dict,
    keys, labels,
    outpath,
    title="",
    max_points=40000,
    q_lo=0.01, q_hi=0.99,
    seed=123,
    bins=70,
):
    # Build paired matrices (N, D) with matched rows
    cols_r = [np.asarray(real_dict[k], dtype=np.float64) for k in keys]
    cols_g = [np.asarray(gen_dict[k],  dtype=np.float64) for k in keys]
    R = np.stack(cols_r, axis=1)
    G = np.stack(cols_g, axis=1)

    # keep finite rows only
    R = R[np.all(np.isfinite(R), axis=1)]
    G = G[np.all(np.isfinite(G), axis=1)]

    if R.shape[0] == 0 or G.shape[0] == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data for corner plot", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        savefig(fig, outpath)
        return

    # subsample (keeps it fast)
    rng = np.random.default_rng(seed)
    if R.shape[0] > max_points:
        R = R[rng.choice(R.shape[0], size=max_points, replace=False)]
    if G.shape[0] > max_points:
        G = G[rng.choice(G.shape[0], size=max_points, replace=False)]

    D = len(keys)

    # per-dim robust ranges shared across real+gen
    ranges = []
    for d in range(D):
        z = np.concatenate([R[:, d], G[:, d]])
        r = robust_range_1d(z, q_lo=q_lo, q_hi=q_hi)
        if r is None:
            lo, hi = float(np.min(z)), float(np.max(z))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                lo, hi = -1.0, 1.0
            ranges.append((lo, hi))
        else:
            ranges.append(r)

    fig, axes = plt.subplots(D, D, figsize=(2.4 * D, 2.4 * D), constrained_layout=True)

    for i in range(D):
        for j in range(D):
            ax = axes[i, j]

            # upper triangle off
            if j > i:
                ax.set_axis_off()
                continue

            # diagonal: 1D hist overlay
            if i == j:
                lo, hi = ranges[i]
                xr = R[:, i]
                xg = G[:, i]
                xr = xr[(xr >= lo) & (xr <= hi)]
                xg = xg[(xg >= lo) & (xg <= hi)]

                ax.hist(xr, bins=bins, range=(lo, hi), density=True, alpha=0.55, label="Real")
                ax.hist(xg, bins=bins, range=(lo, hi), density=True, histtype="step", linewidth=1.8, label="Gen")

                if i == 0:
                    ax.legend(loc="upper right", fontsize=9)

            # lower triangle: real density + gen contours (lognorm)
            else:
                xlo, xhi = ranges[j]
                ylo, yhi = ranges[i]

                xr, yr = clamp_pairs_to_ranges(R[:, j], R[:, i], xlo, xhi, ylo, yhi)
                xg, yg = clamp_pairs_to_ranges(G[:, j], G[:, i], xlo, xhi, ylo, yhi)

                if xr.size == 0 or xg.size == 0:
                    ax.set_axis_off()
                    continue

                ax.hist2d(
                    xr, yr,
                    bins=bins,
                    range=[[xlo, xhi], [ylo, yhi]],
                    density=True,
                    norm=mcolors.LogNorm()
                )

                H, xedges, yedges = np.histogram2d(
                    xg, yg,
                    bins=bins,
                    range=[[xlo, xhi], [ylo, yhi]],
                    density=True
                )
                H = H.T
                H[H <= 0] = np.nan

                vmax = np.nanmax(H)
                if np.isfinite(vmax) and vmax > 0:
                    levels = np.geomspace(vmax * 1e-3, vmax * 0.8, 6)
                    xc = 0.5 * (xedges[:-1] + xedges[1:])
                    yc = 0.5 * (yedges[:-1] + yedges[1:])
                    X, Y = np.meshgrid(xc, yc, indexing="xy")
                    ax.contour(X, Y, H, levels=levels, linewidths=1.3)

            # labels only on left + bottom
            if i == D - 1:
                ax.set_xlabel(labels[j], fontsize=10)
            else:
                ax.set_xticklabels([])

            if j == 0 and i != 0:
                ax.set_ylabel(labels[i], fontsize=10)
            elif j != 0:
                ax.set_yticklabels([])

    if title:
        fig.suptitle(title, y=1.02)

    savefig(fig, outpath)


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "#cccccc",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "axes.grid": False,
    })


def savefig(fig, path, dpi=200):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.06, facecolor=fig.get_facecolor())
    plt.close(fig)


def load_events(path: str):
    arr = np.load(path, allow_pickle=True)
    
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return list(arr)
    
    if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[-1] >= 4:
        return [arr[i] for i in range(arr.shape[0])]
    
    raise ValueError(f"Unrecognized format in {path}")


def sanitize_event(ev, me=0.000511):
    """
    Supports:
      - (K,4): [pdg, px, py, pz]
      - (K,7+): [E_signed, betax, betay, betaz, x, y, z, ...]
    Always returns 13 arrays:
      pdg, px, py, pz, Eabs, E_signed, beta_mag, x, y, z, betax, betay, betaz
    """
    empty_f = np.array([], dtype=np.float64)
    empty_i = np.array([], dtype=np.int64)

    if ev is None:
        return (empty_i, empty_f, empty_f, empty_f, empty_f, empty_f, empty_f,
                empty_f, empty_f, empty_f, empty_f, empty_f, empty_f)

    ev = np.asarray(ev)
    if ev.size == 0:
        return (empty_i, empty_f, empty_f, empty_f, empty_f, empty_f, empty_f,
                empty_f, empty_f, empty_f, empty_f, empty_f, empty_f)

    # ---- Old format: [pdg, px, py, pz] ----
    if ev.ndim == 2 and ev.shape[1] >= 4 and ev.shape[1] < 7:
        pdg = ev[:, 0].astype(np.int64, copy=False)
        px  = ev[:, 1].astype(np.float64, copy=False)
        py  = ev[:, 2].astype(np.float64, copy=False)
        pz  = ev[:, 3].astype(np.float64, copy=False)

        p2 = px*px + py*py + pz*pz
        Eabs = np.sqrt(p2 + me*me)

        # no sign info in this format -> just treat as positive
        E_signed = Eabs.copy()

        beta_mag = np.sqrt(p2) / np.maximum(Eabs, 1e-12)

        return (pdg, px, py, pz, Eabs, E_signed, beta_mag,
                empty_f, empty_f, empty_f, empty_f, empty_f, empty_f)

    # ---- New format: [E_signed, betax, betay, betaz, x, y, z, ...] ----
    if ev.ndim == 2 and ev.shape[1] >= 7:
        E_signed = ev[:, 0].astype(np.float64, copy=False)
        betax = ev[:, 1].astype(np.float64, copy=False)
        betay = ev[:, 2].astype(np.float64, copy=False)
        betaz = ev[:, 3].astype(np.float64, copy=False)

        x = ev[:, 4].astype(np.float64, copy=False)
        y = ev[:, 5].astype(np.float64, copy=False)
        z = ev[:, 6].astype(np.float64, copy=False)

        # Infer "charge species" from sign so we can plot e− vs e+
        # Convention: E_signed < 0 => e− (PDG 11), E_signed > 0 => e+ (PDG -11)
        pdg = np.zeros(len(E_signed), dtype=np.int64)
        pdg[E_signed > 0] = 11
        pdg[E_signed < 0] = -11

        Eabs = np.maximum(np.abs(E_signed), me)

        beta = np.stack([betax, betay, betaz], axis=1)
        beta_mag = np.linalg.norm(beta, axis=1)

        # Momentum from (E, beta): p = Eabs * beta (keeps magnitude physical)
        pvec = Eabs[:, None] * beta
        px, py, pz = pvec[:, 0], pvec[:, 1], pvec[:, 2]

        return (pdg, px, py, pz, Eabs, E_signed, beta_mag,
                x, y, z, betax, betay, betaz)

    raise ValueError(f"Unrecognized event shape {ev.shape}")




def extract_species(events, pdgs=None, me=0.000511):
    mult = np.zeros(len(events), dtype=np.int64)
    px_list, py_list, pz_list = [], [], []
    E_list, Esigned_list, bmag_list = [], [], []
    x_list, y_list, z_list = [], [], []
    bx_list, by_list, bz_list = [], [], []

    for i, ev in enumerate(events):
        pdg, px, py, pz, Eabs, E_signed, bmag, x, y, z, betax, betay, betaz = sanitize_event(ev, me=me)

        if pdgs is None:
            sel = np.ones(len(px), dtype=bool)
        else:
            sel = np.zeros(len(px), dtype=bool)
            for code in pdgs:
                sel |= (pdg == code)

        mult[i] = int(np.sum(sel))

        if np.any(sel):
            px_list.append(px[sel]); py_list.append(py[sel]); pz_list.append(pz[sel])
            E_list.append(Eabs[sel])
            Esigned_list.append(E_signed[sel])
            bmag_list.append(bmag[sel])
            bx_list.append(betax[sel]); by_list.append(betay[sel]); bz_list.append(betaz[sel])

            if x.size:
                x_list.append(x[sel]); y_list.append(y[sel]); z_list.append(z[sel])

    def cat_or_empty(lst):
        return np.concatenate(lst) if len(lst) else np.array([], dtype=np.float64)

    px_all = cat_or_empty(px_list)
    py_all = cat_or_empty(py_list)
    pz_all = cat_or_empty(pz_list)
    p_all  = np.sqrt(px_all**2 + py_all**2 + pz_all**2)
    pt_all = np.sqrt(px_all**2 + py_all**2)

    E_signed_all = cat_or_empty(Esigned_list)

    return {
        "mult": mult,
        "px": px_all, "py": py_all, "pz": pz_all, "p": p_all, "pt": pt_all,
        "E": cat_or_empty(E_list),                 # this is already |E| (Eabs)
        "E_abs": cat_or_empty(E_list),             # alias, for clarity in plotting
        "E_signed": E_signed_all,
        "beta_mag": cat_or_empty(bmag_list),
        "x": cat_or_empty(x_list), "y": cat_or_empty(y_list), "z": cat_or_empty(z_list),
        "betax": cat_or_empty(bx_list),
        "betay": cat_or_empty(by_list),
        "betaz": cat_or_empty(bz_list),
    }





def peak_centered_range(x, y, bins=400, frac=0.995, min_width=1e-12):
    vals = []
    for arr in (x, y):
        arr = np.asarray(arr, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            vals.append(arr)
    if not vals:
        return None
    
    z = np.concatenate(vals)
    if z.size < 2:
        return None
    
    zmin = float(np.min(z))
    zmax = float(np.max(z))
    if not np.isfinite(zmin) or not np.isfinite(zmax) or (zmax - zmin) < min_width:
        return None
    
    counts, edges = np.histogram(z, bins=bins, range=(zmin, zmax))
    tot = counts.sum()
    if tot == 0:
        return None
    
    i0 = int(np.argmax(counts))
    lo_i = hi_i = i0
    cum = int(counts[i0])
    
    target = frac * tot
    while cum < target and (lo_i > 0 or hi_i < len(counts) - 1):
        if lo_i > 0 and hi_i < len(counts) - 1:
            if counts[lo_i - 1] > counts[hi_i + 1]:
                lo_i -= 1
                cum += counts[lo_i]
            else:
                hi_i += 1
                cum += counts[hi_i]
        elif lo_i > 0:
            lo_i -= 1
            cum += counts[lo_i]
        else:
            hi_i += 1
            cum += counts[hi_i]
    
    lo = float(edges[lo_i])
    hi = float(edges[hi_i + 1])
    return (lo, hi)


def robust_range(x, y, q_lo=0.005, q_hi=0.995):
    vals = []
    for arr in (x, y):
        arr = np.asarray(arr, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            vals.append(arr)
    if not vals:
        return None
    
    z = np.concatenate(vals)
    if z.size < 2:
        return None
    
    lo = float(np.quantile(z, q_lo))
    hi = float(np.quantile(z, q_hi))
    
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return None
    
    return (lo, hi)


def clamp_to_range(arr, lo, hi):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return arr[(arr >= lo) & (arr <= hi)]


def kl_divergence_from_counts(p_counts, q_counts, eps=1e-12):
    p = np.asarray(p_counts, dtype=np.float64)
    q = np.asarray(q_counts, dtype=np.float64)
    
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    
    return float(np.sum(p * np.log(p / q)))


def wasserstein_1d(x, y):
    if len(x) == 0 or len(y) == 0:
        return np.nan
    if HAVE_SCIPY:
        return float(wasserstein_distance(x, y))
    
    xs = np.sort(np.asarray(x, dtype=np.float64))
    ys = np.sort(np.asarray(y, dtype=np.float64))
    q = np.linspace(0.0, 1.0, 400)
    xq = np.interp(q, np.linspace(0.0, 1.0, len(xs)), xs)
    yq = np.interp(q, np.linspace(0.0, 1.0, len(ys)), ys)
    return float(np.mean(np.abs(xq - yq)))


def plot_multiplicity(real_mult, gen_mult, outpath, species_name, n_real, n_gen, bins=50, logy=False):
    fig, ax = plt.subplots(1, 1, figsize=(7.6, 4.8), constrained_layout=True)
    
    rng = robust_range(real_mult, gen_mult, q_lo=0.0, q_hi=1.0)
    if rng is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        savefig(fig, outpath)
        return
    
    lo, hi = rng
    real_mult = clamp_to_range(real_mult, lo, hi)
    gen_mult  = clamp_to_range(gen_mult,  lo, hi)
    
    ax.hist(real_mult, bins=bins, range=(lo, hi), density=True, alpha=0.55, label="Real")
    ax.hist(gen_mult,  bins=bins, range=(lo, hi), density=True, histtype="step",
            linewidth=1.8, label="Generated")
    
    ax.set_title(
        f"Multiplicity of {species_name} | real={n_real} gen={n_gen}",
        pad=10
    )    
    ax.set_xlabel(f"Multiplicity N({species_name}) per event")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left")
    if logy:
        ax.set_yscale("log")
    
    savefig(fig, outpath)


def two_panel_dist(real, gen, outpath, xlabel, title, species_name, bins=80, ratio_min_count=10, frac_range=0.80):
    fig = plt.figure(figsize=(7.6, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
    
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)
    ax_top.tick_params(labelbottom=False)
    
    rng = peak_centered_range(real, gen, bins=600, frac=frac_range)
    if rng is None:
        ax_top.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_top.transAxes)
        ax_top.set_axis_off()
        ax_bot.set_axis_off()
        savefig(fig, outpath)
        return
    
    lo, hi = rng
    
    real_use = clamp_to_range(real, lo, hi)
    gen_use  = clamp_to_range(gen,  lo, hi)
    
    if real_use.size == 0 or gen_use.size == 0:
        ax_top.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_top.transAxes)
        ax_top.set_axis_off()
        ax_bot.set_axis_off()
        savefig(fig, outpath)
        return
    
    r_counts, edges = np.histogram(real_use, bins=bins, range=(lo, hi), density=False)
    g_counts, _     = np.histogram(gen_use,  bins=bins, range=(lo, hi), density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    
    ax_top.hist(real_use, bins=bins, range=(lo, hi), density=True, alpha=0.55, label="Real Data")
    ax_top.hist(gen_use,  bins=bins, range=(lo, hi), density=True, histtype="step",
                linewidth=1.8, label="Generated Data")
    
    ax_top.set_title(f"{title} — {species_name}", pad=10)
    ax_top.set_ylabel("Density", labelpad=10)
    ax_bot.set_ylabel("Frac. diff.", labelpad=10)
    ax_top.legend(loc="upper left")
    
    kl = kl_divergence_from_counts(r_counts, g_counts)
    wd = wasserstein_1d(real_use, gen_use)
    
    ax_top.text(
        0.98, 0.95, f"KL: {kl:.4f}\nW1: {wd:.4f}",
        transform=ax_top.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#bbbbbb", alpha=0.95),
    )
    
    r_sum = r_counts.sum()
    g_sum = g_counts.sum()
    
    r_prob = r_counts / max(r_sum, 1)
    g_prob = g_counts / max(g_sum, 1)
    
    mask = r_counts >= ratio_min_count
    
    frac = np.full_like(r_prob, np.nan, dtype=np.float64)
    frac[mask] = (g_prob[mask] - r_prob[mask]) / r_prob[mask]
    
    frac_err = np.full_like(r_prob, np.nan, dtype=np.float64)
    frac_err[mask] = 1.0 / np.sqrt(r_counts[mask])
    
    ax_bot.axhline(0.0, linewidth=1.0, color='black')
    ax_bot.axhspan(-0.1, 0.1, color="gray", alpha=0.15, zorder=0)
    
    ax_bot.errorbar(centers[mask], frac[mask], yerr=frac_err[mask],
                    fmt="o", markersize=3, linewidth=1.0, capsize=0)
    
    ax_bot.set_xlabel(xlabel)
    ax_bot.set_ylim(-1.0, 1.0)
    
    ax_bot.text(0.98, 0.85, "±10% band", transform=ax_bot.transAxes,
                ha="right", va="center", fontsize=10)
    
    savefig(fig, outpath)


def evaluate(args):
    setup_style()
    cfg = CFG()

    if args.outdir is None:
        args.outdir = os.path.dirname(args.gen_path)
    os.makedirs(args.outdir, exist_ok=True)

    real_events = load_events(args.real_path)
    gen_events  = load_events(args.gen_path)

    n_real = len(real_events)
    n_gen  = len(gen_events)
    print(f"Loaded {n_real} real events and {n_gen} generated events")

    # ALWAYS do e-, e+, and total (all)
    species_list = [
        {"name": "e−",  "pdgs": [11],   "tag": "eminus"},
        {"name": "e+",  "pdgs": [-11],  "tag": "eplus"},
        {"name": "all", "pdgs": None,   "tag": "all"},
    ]


    # What to plot (key -> xlabel)
    # These are computed in extract_species() from sanitize_event()
    plot_keys_all = [
        ("E_signed", "E (signed) [GeV]"),
        ("betax", r"$\beta_x$"),
        ("betay", r"$\beta_y$"),
        ("betaz", r"$\beta_z$"),
        ("x", "x [nm]"),
        ("y", "y [nm]"),
        ("z", "z [nm]"),
        ("px", "p_x [GeV]"),
        ("py", "p_y [GeV]"),
        ("pz", "p_z [GeV]"),
        ("pt", "p_T [GeV]"),
        ("p",  "|p| [GeV]"),
    ]

    plot_keys_charge = [
        ("E_abs", "|E| [GeV]"),
        ("betax", r"$\beta_x$"),
        ("betay", r"$\beta_y$"),
        ("betaz", r"$\beta_z$"),
        ("x", "x [nm]"),
        ("y", "y [nm]"),
        ("z", "z [nm]"),
        ("px", "p_x [GeV]"),
        ("py", "p_y [GeV]"),
        ("pz", "p_z [GeV]"),
        ("pt", "p_T [GeV]"),
        ("p",  "|p| [GeV]"),
    ]



    for sp in species_list:
        print(f"Processing species: {sp['name']}")
        real_sp = extract_species(real_events, sp["pdgs"], me=cfg.me)
        gen_sp  = extract_species(gen_events,  sp["pdgs"], me=cfg.me)

        # Corner plots (overlay real vs generated)
        # Keep these small + informative. Subsampled so it won’t explode.
        corner_sets = [
            (["px", "py", "pz"], ["p_x [GeV]", "p_y [GeV]", "p_z [GeV]"], "p_xyz"),
            (["betax", "betay", "betaz"], [r"$\beta_x$", r"$\beta_y$", r"$\beta_z$"], "beta_xyz"),
        ]

        # Only do position corner if positions exist in the inputs
        if real_sp["x"].size and gen_sp["x"].size:
            corner_sets.append((["x", "y", "z"], ["x [nm]", "y [nm]", "z [nm]"], "xyz"))

        for keys, labels, tag2 in corner_sets:
            # skip if any component empty
            if any(real_sp[k].size == 0 or gen_sp[k].size == 0 for k in keys):
                continue

            plot_corner_overlay(
                real_dict=real_sp,
                gen_dict=gen_sp,
                keys=keys,
                labels=labels,
                outpath=os.path.join(args.outdir, f"corner_{tag2}_{sp['tag']}.png"),
                title=f"Corner: {tag2} — {sp['name']} | real={n_real} gen={n_gen}",
                max_points=30000,   # tune if you want
                q_lo=0.01, q_hi=0.99,
                seed=cfg.seed,
            )

        # Multiplicity
        plot_multiplicity(
            real_sp["mult"], gen_sp["mult"],
            outpath=os.path.join(args.outdir, f"multiplicity_{sp['tag']}.png"),
            species_name=sp["name"],
            n_real=n_real, n_gen=n_gen,
            bins=args.mult_bins,
            logy=False,
        )

        # Distributions (all requested metrics)
        plot_keys = plot_keys_all if sp["tag"] == "all" else plot_keys_charge
        for key, xlabel in plot_keys:
            # positions may be missing if input format is (K,4)
            if key not in real_sp or key not in gen_sp:
                continue
            if real_sp[key].size == 0 or gen_sp[key].size == 0:
                continue

            # Use a wider frac_range for positions (tails can be huge)
            frac = cfg.frac_range
            if key in ("x", "y", "z"):
                frac = 0.98

            two_panel_dist(
                real_sp[key], gen_sp[key],
                outpath=os.path.join(args.outdir, f"{key}_{sp['tag']}.png"),
                xlabel=xlabel,
                title=f"Comparison of {key} | real={n_real} gen={n_gen}",
                species_name=sp["name"],
                bins=args.mom_bins,
                ratio_min_count=args.ratio_min_count,
                frac_range=frac,
            )

    print(f"Evaluation complete. Plots saved to: {args.outdir}/")



# ============================================================
# MAIN
# ============================================================
def main():
    cfg = CFG()

    parser = argparse.ArgumentParser(
        description="Particle diffusion model - train, sample, or evaluate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train:    python particle_diffusion.py train --data_path mc_gen1.npy --outdir results --epochs 100
  Sample:   python particle_diffusion.py sample --outdir results --n_events 1000
  Evaluate: python particle_diffusion.py evaluate --real_path mc_gen1.npy --gen_path results/generated_events.npy
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Mode of operation')
    
    # Train
    train_parser = subparsers.add_parser('train', help='Train the diffusion model')
    train_parser.add_argument('--data_path', type=str, help='Path to training data (.npy)')
    train_parser.add_argument('--outdir', type=str, help='Output directory')
    train_parser.add_argument('--max_particles', type=int, help='Max particles per event')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs')
    train_parser.add_argument('--batch_size', type=int, help='Batch size')
    train_parser.add_argument('--T', type=int, help='Diffusion steps')
    train_parser.add_argument('--seed', type=int, help='Random seed')
    
    # Sample
    sample_parser = subparsers.add_parser('sample', help='Generate synthetic events')
    sample_parser.add_argument('--outdir', type=str, default=cfg.outdir, help='Model directory')
    sample_parser.add_argument('--n_events', type=int, default=1247, help='Number of events')
    
    # Evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate distributions')
    eval_parser.add_argument('--real_path', type=str, default=cfg.data_path, help='Real data path')
    eval_parser.add_argument('--gen_path', type=str, default=os.path.join(cfg.outdir, "generated_events.npy"), help='Generated data path')
    eval_parser.add_argument('--outdir', type=str, default=os.path.join(cfg.outdir, "plots"), help='Output directory (default: same as gen_path)')
    eval_parser.add_argument('--include_all', action='store_true', help='Include all species')
    eval_parser.add_argument('--mult_bins', type=int, default=50, help='Multiplicity bins')
    eval_parser.add_argument('--mom_bins', type=int, default=80, help='Momentum bins')
    eval_parser.add_argument('--ratio_min_count', type=int, default=10, help='Min counts for ratio')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'sample':
        sample(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()