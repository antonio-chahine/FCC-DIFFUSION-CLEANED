#!/usr/bin/env python3
"""
Unified particle generation pipeline with DDPM.

Modes:
    train    - Train the diffusion model
    sample   - Generate synthetic events from trained model
    evaluate - Compare real vs generated distributions

Usage:
    python particle_diffusion.py train --data_path mc_gen1.npy --outdir results
    python particle_diffusion.py sample --outdir results --n_events 1000 --num_steps 200
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
import time
import subprocess
import threading


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


def make_cosine_beta_schedule(T: int, s: float = 0.008, device: str = "cpu"):
    steps = torch.arange(T + 1, device=device) / T
    alphas_cumprod = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clamp(betas, 0, 0.999)
    alphas = 1.0 - betas
    acp = torch.cumprod(alphas, dim=0)
    acp_prev = torch.cat([torch.ones(1, device=device), acp[:-1]])
    return betas, alphas, acp, acp_prev

def beta_squash_torch(u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    umag = torch.linalg.norm(u, dim=-1, keepdim=True)
    uhat = u / (umag + 1e-12)
    s = torch.tanh(umag)
    return (1.0 - eps) * s * uhat

def beta_squash_np(u: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    u = np.asarray(u, dtype=np.float32)
    umag = np.linalg.norm(u, axis=1, keepdims=True)
    uhat = u / (umag + 1e-12)
    s = np.tanh(umag)
    return (1.0 - eps) * s * uhat

def make_linear_gamma_schedule(T: int, g0: float, g1: float, device: str):
    return torch.linspace(g0, g1, T, device=device)


def charge_balance_loss(pdg_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Soft charge balance loss.
    Real data has exactly N(e-) == N(e+) per event.
    Penalises squared expected charge imbalance, differentiable through softmax.
    pdg_logits: (B, K, 2)  index 0=e-, index 1=e+
    mask:       (B, K) bool
    """
    probs    = torch.softmax(pdg_logits, dim=-1)
    p_eminus = probs[:, :, 0] * mask
    p_eplus  = probs[:, :, 1] * mask
    imbalance = (p_eminus - p_eplus).sum(dim=1)
    return (imbalance ** 2).mean()

def q_sample_pdg(pdg0: torch.Tensor, t: torch.Tensor, gammas: torch.Tensor,
                 n_classes: int, mask: torch.Tensor):
    B, K = pdg0.shape
    g = gammas[t].view(B, 1)
    u = torch.rand((B, K), device=pdg0.device)
    flip = (u < g) & mask
    pdg_t = pdg0.clone()
    if flip.any():
        pdg_t[flip] = torch.randint(0, n_classes, (int(flip.sum()),), device=pdg0.device)
    return pdg_t

def beta_unsquash_np(beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    beta = np.asarray(beta, dtype=np.float32)
    bmag = np.linalg.norm(beta, axis=1, keepdims=True)
    bmag = np.clip(bmag, 0.0, 1.0 - eps)
    bhat = beta / (bmag + 1e-12)
    s = bmag / (1.0 - eps)
    umag = np.arctanh(np.clip(s, 0.0, 1.0 - 1e-7))
    return umag * bhat

class GPUMonitor(threading.Thread):
    """Background thread to monitor GPU utilization and memory."""
    def __init__(self, interval=0.5):
        super().__init__()
        self.interval = interval
        self.stop_event = threading.Event()
        self.gpu_utils = []
        self.mem_used = []

    def run(self):
        while not self.stop_event.is_set():
            try:
                # Query nvidia-smi for utilization and memory
                result = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                    encoding="utf-8"
                )
                lines = result.strip().split('\n')
                if lines:
                    # If multiple GPUs, process the first one (or modify to average)
                    util, mem = lines[0].split(',')
                    self.gpu_utils.append(float(util.strip()))
                    self.mem_used.append(float(mem.strip()))
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()

    def print_stats(self):
        if not self.gpu_utils:
            print("No GPU stats collected (maybe no NVIDIA GPU or run too fast).")
            return
        avg_util = sum(self.gpu_utils) / len(self.gpu_utils)
        max_util = max(self.gpu_utils)
        max_mem = max(self.mem_used)
        print(f"--- GPU Monitoring Stats ---")
        print(f"Avg GPU Utilization: {avg_util:.1f}%")
        print(f"Max GPU Utilization: {max_util:.1f}%")
        print(f"Max VRAM Used:       {max_mem:.0f} MB")
        if torch.cuda.is_available():
            print(f"PyTorch Max VRAM:    {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
        print(f"----------------------------")


# ============================================================
# CONFIG
# ============================================================
@dataclass
class CFG:
    data_path: str = "/work/submit/haoyun22/FCC-Beam-Background/DM/guineapig_raw_trimmed_new.npy"
    outdir: str = "/work/submit/haoyun22/FCC-Beam-Background/DM/results_nosquash_cosine_chargeloss"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    max_particles: int = 1300
    min_particles: int = 1
    keep_fraction: float = 1.0
    
    T: int = 1000
    cosine_s: float = 0.002273608087915074  # offset for cosine schedule; smaller = more noise at low t
    
    d_model = 512
    nhead = 2
    num_layers = 3
    dropout = 0.08545214831324029
    
    batch_size: int = 2
    lr: float = 0.00015982608583855863
    epochs: int = 50
    num_workers: int = 8
    grad_clip: float = 4.587196486849941
    seed: int = 123

    frac_range = 0.60

    me: float = 0.00051099895069  # GeV

    feat_dim: int = 7
    n_pdg: int = 2
    lambda_pdg: float = 0.3522458811249478
    lambda_charge: float = 0.011241862095793064

    gamma_start: float = 5.232216089948759e-05
    gamma_end: float = 0.17539090890647513

    n_events = 500
    sample_batch_size: int = 16
    num_steps: int = None  # None means use full T, otherwise sample with num_steps timesteps

    pct_start: float = 0.07173423081368346
    div_factor: float = 30.86403908528625


# ============================================================
# DATASET
# ============================================================
class MCPDataset(Dataset):
    """
    Input rows (7 cols):
        [E_signed, betax, betay, betaz, x, y, z]

    PDG inferred from sign(E):
        E > 0  -> electron (11)
        E < 0  -> positron (-11)

    Continuous features:
        [log|E|, u_x, u_y, u_z, x, y, z]
    """

    def __init__(self, path, max_particles=64, min_particles=1, keep_fraction=1.0):
        raw = np.load(path, allow_pickle=True)

        if keep_fraction < 1.0:
            raw = raw[: int(len(raw) * keep_fraction)]

        self.pdg_to_idx = {11: 0, -11: 1}
        self.idx_to_pdg = {0: 11, 1: -11}

        events_cont = []
        events_pdg  = []

        for ev in raw:
            if ev is None:
                continue

            ev = np.asarray(ev)

            if ev.ndim != 2 or ev.shape[1] < 7:
                continue

            if len(ev) < min_particles:
                continue

            ev = ev.astype(np.float32)

            E_signed = ev[:, 0]
            beta = ev[:, 1:4]
            pos  = ev[:, 4:7]

            pdg = np.where(E_signed >= 0.0, 11, -11)
            pdg_idx = np.where(pdg == 11, 0, 1).astype(np.int64)

            Eabs = np.maximum(np.abs(E_signed), 1e-12)
            logE = np.log(Eabs)

            u = beta_unsquash_np(beta)

            cont = np.concatenate(
                [logE[:, None], u, pos],
                axis=1
            ).astype(np.float32)

            events_cont.append(cont)
            events_pdg.append(pdg_idx)

        if len(events_cont) == 0:
            raise RuntimeError("No events left — check that .npy contains (K,7) arrays.")

        self.events_cont = events_cont
        self.events_pdg  = events_pdg
        self.max_particles = max_particles
        self.feat_dim = 7

        all_feats = np.concatenate(events_cont, axis=0)
        self.feat_mean = all_feats.mean(axis=0).astype(np.float32)
        self.feat_std  = np.maximum(all_feats.std(axis=0), 1e-6).astype(np.float32)

        self.multiplicities = np.array([len(ev) for ev in events_cont], dtype=np.int64)

    def __len__(self):
        return len(self.events_cont)

    def __getitem__(self, idx):
        cont = self.events_cont[idx]
        pdg  = self.events_pdg[idx]

        N = len(cont)
        Kmax = self.max_particles

        if N <= Kmax:
            chosen = np.arange(N)
        else:
            chosen = torch.randperm(N)[:Kmax].numpy()

        cont = cont[chosen]
        pdg  = pdg[chosen]
        K = cont.shape[0]

        cont_norm = (cont - self.feat_mean) / self.feat_std

        x0   = np.zeros((Kmax, self.feat_dim), dtype=np.float32)
        pdg0 = np.zeros((Kmax,), dtype=np.int64)
        mask = np.zeros((Kmax,), dtype=np.bool_)

        x0[:K] = cont_norm
        pdg0[:K] = pdg
        mask[:K] = True

        return torch.from_numpy(x0), torch.from_numpy(pdg0), torch.from_numpy(mask)


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
    def __init__(self, d_model=128, nhead=4, num_layers=3, dropout=0.1, n_pdg=2):
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

        self.k_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.pdg_emb = nn.Embedding(n_pdg, d_model)
        self.pdg_head = nn.Linear(d_model, n_pdg)
        self.skip_alpha = 0.2

    def forward(self, x_t, t, pdg_t, mask):
        B, K, _ = x_t.shape

        t_emb = self.time_emb(t)
        t_emb = self.t_mlp(t_emb).unsqueeze(1).expand(B, K, self.d_model)

        mom_emb = self.mom_proj(x_t)

        pdg_emb = self.pdg_emb(pdg_t.clamp(0, self.pdg_emb.num_embeddings - 1))
        
        h = t_emb + mom_emb + pdg_emb

        K_event = mask.sum(dim=1)
        k = torch.log(K_event.float().clamp(min=1)).unsqueeze(-1)
        k_emb = self.k_mlp(k).unsqueeze(1)
        h = h + k_emb

        src_key_padding_mask = ~mask
        h_in = h
        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)
        h = h + self.skip_alpha * h_in
        h = h * mask.unsqueeze(-1)

        eps_hat = self.output(h) * mask.unsqueeze(-1)
        pdg_logits = self.pdg_head(h)  # (B,K,C)

        return eps_hat, pdg_logits


# ============================================================
# DDPM WRAPPER
# ============================================================
class DDPM:
    def __init__(self, model, T, device, cosine_s=0.008):
        self.model = model
        self.T = T
        self.device = device

        betas, alphas, acp, acp_prev = make_cosine_beta_schedule(T, s=cosine_s, device=device)
        
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

    def p_sample(self, x_t, t, pdg_t, mask):
        """
        Single denoising step.
        Returns (x_prev, pdg_logits) — logits come for free from the
        single forward pass so the caller never needs a second one.
        """
        B = x_t.shape[0]

        eps_hat, pdg_logits = self.model(x_t, t, pdg_t, mask)

        beta_t  = self.betas[t].view(B, 1, 1)
        alpha_t = self.alphas[t].view(B, 1, 1)
        acp_t   = self.acp[t].view(B, 1, 1)

        mu = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1.0 - acp_t)) * eps_hat)
        var = self.posterior_variance[t].view(B, 1, 1)

        z = torch.zeros_like(x_t) if t[0].item() == 0 else torch.randn_like(x_t)

        x_prev = (mu + torch.sqrt(var) * z) * mask.unsqueeze(-1)

        # Return logits so the caller does NOT need a second forward pass
        return x_prev, pdg_logits

    @torch.no_grad()
    def sample(self, mask, pdg_init, num_steps=None):
        """
        Full reverse diffusion for a batch of events.
        One forward pass per timestep (logits reused from p_sample).
        
        Args:
            mask: (B, K) bool tensor
            pdg_init: (B, K) initial PDG indices
            num_steps: int or None. If None, use full T steps.
                      Otherwise, sample uniformly num_steps timesteps.
        """
        B, K = mask.shape
        x = torch.randn((B, K, 7), device=self.device) * mask.unsqueeze(-1)
        pdg = pdg_init.clone()

        # Determine which timesteps to use
        if num_steps is None:
            timesteps = list(reversed(range(self.T)))
        else:
            # Uniformly sample num_steps timesteps from [0, T-1]
            num_steps = max(1, min(num_steps, self.T))
            timesteps = np.linspace(0, self.T - 1, num_steps, dtype=int)
            timesteps = list(reversed(timesteps))

        for ti in timesteps:
            t = torch.full((B,), ti, device=self.device, dtype=torch.long)

            # FIX: p_sample now returns logits — no second forward pass needed
            x, logits = self.p_sample(x, t, pdg, mask)

            probs = torch.softmax(logits, dim=-1)
            samp  = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, K)
            pdg   = torch.where(mask, samp, pdg)

        return x, pdg


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

    meta_path = os.path.join(cfg.outdir, "meta.pt")
    if getattr(args, "resume", False) and os.path.exists(meta_path):
        meta_old = torch.load(meta_path, map_location="cpu")
        if "train_idx" in meta_old and "val_idx" in meta_old:
            train_idx = np.asarray(meta_old["train_idx"])
            val_idx   = np.asarray(meta_old["val_idx"])
        else:
            train_idx, val_idx = split_indices(len(ds_full), val_frac, cfg.seed)
    else:
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

    ddpm = DDPM(model, cfg.T, cfg.device, cosine_s=cfg.cosine_s)
    gammas = make_linear_gamma_schedule(cfg.T, cfg.gamma_start, cfg.gamma_end, cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=cfg.lr,
        epochs=cfg.epochs,
        steps_per_epoch=len(dl_train),
        pct_start=cfg.pct_start,
        anneal_strategy='cos',
        div_factor=cfg.div_factor,
        final_div_factor=1e4,
    )

    # ----------------------------
    # Resume (optional)
    # ----------------------------
    start_epoch = 0
    train_losses = []
    val_losses = []

    ckpt_path = os.path.join(cfg.outdir, "ckpt_last.pt")
    if getattr(args, "resume", False) and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=cfg.device)
        model.load_state_dict(ckpt["model"])
        if "opt" in ckpt and ckpt["opt"] is not None:
            opt.load_state_dict(ckpt["opt"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])

        start_epoch = int(ckpt.get("epoch", -1)) + 1

        tl_path = os.path.join(cfg.outdir, "train_losses.npy")
        vl_path = os.path.join(cfg.outdir, "val_losses.npy")
        if os.path.exists(tl_path) and os.path.exists(vl_path):
            train_losses = list(np.load(tl_path).astype(float))
            val_losses   = list(np.load(vl_path).astype(float))

        print(f"Resuming from {ckpt_path} at epoch {start_epoch}")
    else:
        print("Starting training from scratch")

    meta = {
        "multiplicities": ds_full.multiplicities,
        "feat_mean": ds_full.feat_mean,
        "feat_std": ds_full.feat_std,
        "feat_dim": ds_full.feat_dim,
        "me": cfg.me,
        "n_pdg": cfg.n_pdg,
        "idx_to_pdg": ds_full.idx_to_pdg,
        "max_particles": cfg.max_particles,
        "T": cfg.T,
        "cosine_s": cfg.cosine_s,
        "d_model": cfg.d_model,
        "nhead": cfg.nhead,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
        "val_frac": val_frac,
        "seed": cfg.seed,
        "n_events": len(ds_full),
        "n_train_events": len(train_idx),
        "n_val_events": len(val_idx),
        "train_idx": train_idx,
        "val_idx": val_idx,
    }
    if not os.path.exists(meta_path):
        torch.save(meta, meta_path)
        
    for epoch in range(start_epoch, cfg.epochs):
        # TRAIN
        model.train()
        total_train = 0.0
        n_train = 0
        
        pbar = tqdm(dl_train, desc=f"Epoch {epoch+1:03d}/{cfg.epochs} [train]", leave=False)
        for x0, pdg0, mask in pbar:
            x0   = x0.to(cfg.device)
            pdg0 = pdg0.to(cfg.device)
            mask = mask.to(cfg.device)

            B = x0.shape[0]
            t = torch.randint(0, cfg.T, (B,), device=cfg.device)

            noise = torch.randn_like(x0) * mask.unsqueeze(-1)
            x_t   = ddpm.q_sample(x0, t, noise)

            pdg_t = q_sample_pdg(pdg0, t, gammas, cfg.n_pdg, mask)

            eps_hat, pdg_logits = model(x_t, t, pdg_t, mask)

            mse       = (eps_hat - noise).pow(2).sum(dim=-1)
            diff_loss = (mse * mask).sum() / mask.sum().clamp(min=1)
            pdg_loss  = F.cross_entropy(pdg_logits[mask], pdg0[mask])
            c_loss    = charge_balance_loss(pdg_logits, mask)
            loss      = diff_loss + cfg.lambda_pdg * pdg_loss + cfg.lambda_charge * c_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            scheduler.step()
            
            total_train += loss.item()
            n_train += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        train_loss = total_train / max(n_train, 1)
        
        # VALIDATION
        model.eval()
        total_val = 0.0
        n_val = 0
        
        with torch.no_grad():
            pbarv = tqdm(dl_val, desc=f"Epoch {epoch+1:03d}/{cfg.epochs} [val]", leave=False)
            for x0, pdg0, mask in pbarv:
                x0   = x0.to(cfg.device)
                pdg0 = pdg0.to(cfg.device)
                mask = mask.to(cfg.device)

                B = x0.shape[0]
                t = torch.randint(0, cfg.T, (B,), device=cfg.device)

                noise = torch.randn_like(x0) * mask.unsqueeze(-1)
                x_t   = ddpm.q_sample(x0, t, noise)

                pdg_t = q_sample_pdg(pdg0, t, gammas, cfg.n_pdg, mask)

                eps_hat, pdg_logits = model(x_t, t, pdg_t, mask)

                mse       = (eps_hat - noise).pow(2).sum(dim=-1)
                diff_loss = (mse * mask).sum() / mask.sum().clamp(min=1)
                pdg_loss  = F.cross_entropy(pdg_logits[mask], pdg0[mask])
                c_loss    = charge_balance_loss(pdg_logits, mask)
                loss      = diff_loss + cfg.lambda_pdg * pdg_loss + cfg.lambda_charge * c_loss

                total_val += loss.item()
                n_val += 1
                pbarv.set_postfix(loss=f"{loss.item():.4f}")

        val_loss = total_val / max(n_val, 1)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        np.save(os.path.join(cfg.outdir, "train_losses.npy"), np.array(train_losses))
        np.save(os.path.join(cfg.outdir, "val_losses.npy"),   np.array(val_losses))
        
        print(f"Epoch {epoch+1:03d}/{cfg.epochs} | train={train_loss:.6f} | val={val_loss:.6f} | lr={scheduler.get_last_lr()[0]:.2e}")

        torch.save(
            {
                "model":      model.state_dict(),
                "opt":        opt.state_dict(),
                "scheduler":  scheduler.state_dict(),
                "epoch":      epoch,
                "train_loss": train_loss,
                "val_loss":   val_loss,
            },
            os.path.join(cfg.outdir, "ckpt_last.pt"),
        )
    
    np.save(os.path.join(cfg.outdir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(cfg.outdir, "val_losses.npy"),   np.array(val_losses))
    
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
        device=device,
        cosine_s=float(meta.get("cosine_s", 0.008)),
    )

    return meta, model, ddpm


def _decode_batch(x_norm_batch, pdg_idx_batch, Ks, meta):
    """
    Convert a batch of normalised model outputs back to physical units.
    Returns a list of (K_i, 8) float32 arrays: [pdg, E, betax, betay, betaz, x, y, z]
    """
    mean = np.asarray(meta["feat_mean"], dtype=np.float32)
    std  = np.asarray(meta["feat_std"],  dtype=np.float32)
    idx_to_pdg = meta["idx_to_pdg"]

    events = []
    for i, K in enumerate(Ks):
        x_i   = x_norm_batch[i, :K]   # (K, 7)
        pdg_i = pdg_idx_batch[i, :K]  # (K,)

        cont = x_i * std + mean
        logE = cont[:, 0]
        u    = cont[:, 1:4]
        pos  = cont[:, 4:7]

        E    = np.exp(logE)
        beta = beta_squash_np(u)
        pdg  = np.array([idx_to_pdg[int(j)] for j in pdg_i], dtype=np.int64)

        out = np.concatenate([pdg[:, None], E[:, None], beta, pos], axis=1).astype(np.float32)
        events.append(out)

    return events


def sample_batch(meta: dict, ddpm: DDPM, device: str, batch_size: int, num_steps=None):
    """
    Generate `batch_size` events in a single ddpm.sample call.
    All events in the batch are denoised simultaneously — one forward
    pass per timestep regardless of batch size.
    
    Args:
        meta: metadata dict
        ddpm: DDPM instance
        device: torch device
        batch_size: number of events to generate
        num_steps: int or None. If None, use full T steps.
                  Otherwise, sample uniformly num_steps timesteps.
    """
    multiplicities = np.asarray(meta["multiplicities"], dtype=np.int64)
    Kmax  = int(meta["max_particles"])
    n_pdg = int(meta["n_pdg"])

    # Draw multiplicities for the whole batch at once
    Ks = np.random.choice(multiplicities, size=batch_size, replace=True)
    Ks = np.clip(Ks, 1, Kmax).astype(np.int64)

    # Build mask: (B, Kmax)
    mask_np = np.zeros((batch_size, Kmax), dtype=np.bool_)
    for i, K in enumerate(Ks):
        mask_np[i, :K] = True
    mask_t = torch.from_numpy(mask_np).to(device)

    # Initialise PDG uniformly (zero outside mask)
    pdg_init = torch.randint(0, n_pdg, (batch_size, Kmax), device=device)
    pdg_init = pdg_init * mask_t.long()

    with torch.no_grad():
        x_norm, pdg_idx = ddpm.sample(mask_t, pdg_init, num_steps=num_steps)

    x_norm_np  = x_norm.cpu().numpy()   # (B, Kmax, 7)
    pdg_idx_np = pdg_idx.cpu().numpy()  # (B, Kmax)

    return _decode_batch(x_norm_np, pdg_idx_np, Ks, meta)


def sample_single(meta: dict, ddpm: DDPM, device: str):
    """
    Generate exactly one event (original behaviour, kept for reference /
    debugging).  Calls ddpm.sample with batch_size=1.
    """
    return sample_batch(meta, ddpm, device, batch_size=1)[0]


def sample(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir      = args.outdir
    n_events    = args.n_events
    batch_size  = args.sample_batch_size
    num_steps   = getattr(args, "num_steps", None)
    print("Using device:", device)
    print("n_events:", n_events)
    print("sample_batch_size:", batch_size)
    print("num_steps:", num_steps)

    meta, model, ddpm = load_meta_and_model(outdir, device)

    if num_steps is not None:
        print(f"Sampling with {num_steps} steps (out of {meta['T']} total)")
    else:
        print(f"Sampling with full {meta['T']} steps")


    monitor = GPUMonitor(interval=0.2)
    monitor.start()

    start_time = time.time()

    events   = []
    n_done   = 0
    n_batches = math.ceil(n_events / batch_size)

    for _ in tqdm(range(n_batches), desc="Generating batches"):
        remaining = n_events - n_done
        bs = min(batch_size, remaining)
        events.extend(sample_batch(meta, ddpm, device, bs, num_steps=num_steps))
        n_done += bs

    end_time = time.time()
    monitor.stop()
    monitor.join()

    elapsed = end_time - start_time
    print(f"\n" + "="*60)
    print(f"Generated {n_events} events in {elapsed:.2f} seconds")
    avg_time = elapsed / n_events
    print(f"Average time per event: {avg_time*1000:.4f} ms")
    print(f"Throughput: {n_events/elapsed:.1f} events/sec")
    print("="*60)
    monitor.print_stats()
    print("="*60)

    out_path = os.path.join(outdir, f"generated_events_{num_steps}steps.npy")
    np.save(out_path, np.array(events, dtype=object))
    print("Saved:", out_path)



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
  Sample:   python particle_diffusion.py sample --outdir results --n_events 1000 --num_steps 200
  Sample (batched): python particle_diffusion.py sample --outdir results --n_events 1000 --sample_batch_size 32 --num_steps 200
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Mode of operation')
    
    # Train
    train_parser = subparsers.add_parser('train', help='Train the diffusion model')
    train_parser.add_argument('--data_path',     type=str, help='Path to training data (.npy)')
    train_parser.add_argument('--outdir',        type=str, help='Output directory')
    train_parser.add_argument('--max_particles', type=int, help='Max particles per event')
    train_parser.add_argument('--epochs',        type=int, help='Number of epochs')
    train_parser.add_argument('--batch_size',    type=int, help='Batch size')
    train_parser.add_argument('--T',             type=int, help='Diffusion steps')
    train_parser.add_argument('--seed',          type=int, help='Random seed')
    train_parser.add_argument('--resume', action='store_true',
                              help='Resume from outdir/ckpt_last.pt if it exists')
    
    # Sample
    sample_parser = subparsers.add_parser('sample', help='Generate synthetic events')
    sample_parser.add_argument('--outdir',             type=str, default=cfg.outdir,
                               help='Model directory')
    sample_parser.add_argument('--n_events',           type=int, default=cfg.n_events,
                               help='Number of events to generate')
    sample_parser.add_argument('--sample_batch_size',  type=int, default=cfg.sample_batch_size,
                               help='Events per GPU batch during sampling (default: 16). '
                                    'Increase until you hit memory limits for best throughput.')
    sample_parser.add_argument('--num_steps',          type=int, default=None,
                               help='Number of denoising steps (default: None means use full T). '
                                    'Smaller values = faster but lower quality. E.g., --num_steps 250 '
                                    'for 1/4 speed with ~1000 total steps.')

    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'sample':
        sample(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()