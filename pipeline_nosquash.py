#!/usr/bin/env python3
"""
Unified particle generation pipeline with DDPM.

Modes:
    train    - Train the diffusion model
    sample   - Generate synthetic events from trained model
    evaluate - Compare real vs generated distributions

Usage:
    python DM/pipeline_nosquash.py train --data_path mc_gen1.npy --outdir results
    python DM/pipeline_nosquash.py train --outdir results --epochs 400 --resume
    python DM/pipeline_nosquash.py sample --outdir /ceph/submit/data/user/h/haoyun22/dm_generated_results/results_nosquash_cosine --n_events 128 --sample_steps 200
    python DM/pipeline_nosquash.py evaluate --real_path mc_gen1.npy --gen_path results/generated_events.npy
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

import threading
import subprocess

from unet import build_unet_denoiser




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
    """
    Cosine noise schedule from Nichol & Dhariwal (2021).
    ᾱ_t = cos²(π/2 · (t/T + s)/(1 + s))
    """
    steps = T + 1
    t = torch.linspace(0, T, steps, device=device)
    
    # Compute cumulative alphas
    f_t = torch.cos(((t / T + s) / (1 + s)) * math.pi / 2) ** 2
    acp = f_t / f_t[0]                              # normalize so acp[0] = 1
    
    # Derive betas from acp
    betas = 1.0 - (acp[1:] / acp[:-1])
    betas = torch.clamp(betas, min=1e-5, max=0.999) # prevent instability
    
    alphas = 1.0 - betas
    acp = acp[1:]                                    # drop t=0, shape (T,)
    acp_prev = torch.cat([torch.ones(1, device=device), acp[:-1]])
    
    return betas, alphas, acp, acp_prev

def beta_squash_torch(u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # u: (B,K,3) or (...,3)
    umag = torch.linalg.norm(u, dim=-1, keepdim=True)
    uhat = u / (umag + 1e-12)
    s = torch.tanh(umag)  # [0,1)
    return (1.0 - eps) * s * uhat

def beta_squash_np(u: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    u = np.asarray(u, dtype=np.float32)
    umag = np.linalg.norm(u, axis=1, keepdims=True)
    uhat = u / (umag + 1e-12)
    s = np.tanh(umag)
    return (1.0 - eps) * s * uhat

def beta_squash_np_batch(u: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Vectorized beta squashing for batch of events.
    
    Args:
        u: shape (..., 3) - can be (K, 3), (B, K, 3), etc.
        eps: small epsilon for clipping
    
    Returns:
        beta: shape (..., 3) - same shape as input
    """
    u = np.asarray(u, dtype=np.float32)
    umag = np.linalg.norm(u, axis=-1, keepdims=True)
    uhat = u / (umag + 1e-12)
    s = np.tanh(umag)
    return (1.0 - eps) * s * uhat

def make_linear_gamma_schedule(T: int, g0: float, g1: float, device: str):
    return torch.linspace(g0, g1, T, device=device)

def q_sample_pdg(pdg0: torch.Tensor, t: torch.Tensor, gammas: torch.Tensor,
                 n_classes: int, mask: torch.Tensor):
    # pdg0: (B,K) long, mask: (B,K) bool
    B, K = pdg0.shape
    g = gammas[t].view(B, 1)  # (B,1)
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

    # inverse of s = tanh(umag)  => umag = atanh(s)
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
    data_path: str = "DM/guineapig_raw_trimmed_new.npy"
    outdir: str = "/ceph/submit/data/user/h/haoyun22/dm_generated_results/results_nosquash_cosine_bigdata"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    max_particles: int = 1300
    min_particles: int = 1
    keep_fraction: float = 1.0
    
    T: int = 200
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    
    d_model = 256 # 128 
    nhead = 8
    num_layers = 8 # 4
    dropout = 0.1
    
    batch_size: int = 1
    lr: float = 2e-4
    epochs: int = 30
    num_workers: int = 8
    grad_clip: float = 1.0
    seed: int = 123

    frac_range = 0.60

    me: float = 0.00051099895069 # GeV


    feat_dim: int = 7            # stays 7 (logE + u(3) + xyz)
    n_pdg: int = 2               # start with e-, e+
    lambda_pdg: float = 1.0
    lambda_pt: float = 0.3

    gamma_start: float = 1e-4
    gamma_end: float = 0.2


    n_events = 1000


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

            # --- Columns ---
            E_signed = ev[:, 0]
            beta = ev[:, 1:4]
            pos  = ev[:, 4:7]

            # --- Infer PDG from sign(E) ---
            pdg = np.where(E_signed >= 0.0, 11, -11)
            pdg_idx = np.where(pdg == 11, 0, 1).astype(np.int64)

            # --- Continuous features ---
            Eabs = np.maximum(np.abs(E_signed), 1e-12)
            logE = np.log(Eabs)

            u = beta_unsquash_np(beta)  # NOW u is truly unconstrained


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

        # NEW: multiplicity conditioning MLP
        self.k_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )


        self.pdg_emb = nn.Embedding(n_pdg, d_model)
        self.pdg_head = nn.Linear(d_model, n_pdg)
        self.skip_alpha = 0.2  # make it constant if you prefer

    def forward(self, x_t, t, pdg_t, mask):
        B, K, _ = x_t.shape

        t_emb = self.time_emb(t)
        t_emb = self.t_mlp(t_emb).unsqueeze(1).expand(B, K, self.d_model)

        mom_emb = self.mom_proj(x_t)

        pdg_emb = self.pdg_emb(pdg_t.clamp(0, self.pdg_emb.num_embeddings - 1))
        
        h = t_emb + mom_emb + pdg_emb

        # multiplicity conditioning (keep)
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
    def __init__(self, model, T, beta_start, beta_end, device, schedule="cosine"):
        self.model = model
        self.T = T
        self.device = device
        self.schedule = schedule

        if schedule == "linear":
            betas, alphas, acp, acp_prev = make_linear_beta_schedule(T, beta_start, beta_end, device)
            print(f"Using linear beta schedule: beta_start={beta_start}, beta_end={beta_end}")
        elif schedule == "cosine":
            betas, alphas, acp, acp_prev = make_cosine_beta_schedule(T, s=0.008, device=device)
            print(f"Using cosine beta schedule with s=0.008")
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        self.betas = betas
        self.alphas = alphas
        self.acp = acp
        self.acp_prev = acp_prev
        
        self.sqrt_acp = torch.sqrt(acp)
        self.sqrt_1m_acp = torch.sqrt(1.0 - acp)
        self.posterior_variance = (betas * (1.0 - acp_prev) / (1.0 - acp).clamp(min=1e-12))
    
    def q_sample(self, x0, t, noise):
        B = x0.shape[0]
        a = self.sqrt_acp[t].view(B, 1, 1)
        b = self.sqrt_1m_acp[t].view(B, 1, 1)
        return a * x0 + b * noise
    
    def p_sample(self, x_t, t, pdg_t, mask):
        B = x_t.shape[0]

        eps_hat, logits = self.model(x_t, t, pdg_t, mask)

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
        return x_prev * mask.unsqueeze(-1), logits

    
    @torch.no_grad()
    def sample(self, mask, pdg_init, sample_steps=None):
        """
        Sample from the diffusion model.
        
        Args:
            mask: particle mask
            pdg_init: initial PDG codes
            sample_steps: number of sampling steps (if None, use self.T)
                         if < self.T, will use strided sampling
        """
        B, K = mask.shape
        x = torch.randn((B, K, 7), device=self.device) * mask.unsqueeze(-1)
        pdg = pdg_init.clone()

        # Determine timesteps to use
        if sample_steps is None or sample_steps >= self.T:
            timesteps = list(reversed(range(self.T)))
        else:
            if self.schedule == "linear":
                # Strided sampling: select evenly spaced timesteps
                stride = self.T // sample_steps
                timesteps = list(reversed(range(0, self.T, stride)))
                if timesteps[0] != self.T - 1:
                    timesteps.insert(0, self.T - 1)
            else: 
                indices = np.linspace(0, self.T - 1, sample_steps, dtype=int)
                timesteps = list(reversed(sorted(set(indices.tolist()))))

        for ti in timesteps:
            t = torch.full((B,), ti, device=self.device, dtype=torch.long)
            x, logits = self.p_sample(x, t, pdg, mask)

            # denoise PDG at each step (simple)
            probs = torch.softmax(logits, dim=-1)
            samp = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, K)
            pdg = torch.where(mask, samp, pdg)

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

    # model = build_unet_denoiser(
    #     d_model = cfg.d_model,
    #     n_pdg = cfg.n_pdg,
    #     base_ch = 32,
    #     channel_mults = (1, 2),
    #     attn_levels = (0, 1),
    # ).to(cfg.device)

    
    ddpm = DDPM(model, cfg.T, cfg.beta_start, cfg.beta_end, cfg.device)
    gammas = make_linear_gamma_schedule(cfg.T, cfg.gamma_start, cfg.gamma_end, cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    ckpt_last_path = os.path.join(cfg.outdir, "ckpt_last.pt")
    train_losses = []
    val_losses = []
    start_epoch = 0

    if args.resume is not None:
        resume_path = ckpt_last_path if args.resume == "auto" else args.resume

        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        ckpt = torch.load(resume_path, map_location=cfg.device, weights_only=False)
        model.load_state_dict(ckpt["model"])

        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])

        start_epoch = int(ckpt.get("epoch", -1)) + 1
        train_losses = list(np.asarray(ckpt.get("train_losses", []), dtype=np.float32))
        val_losses = list(np.asarray(ckpt.get("val_losses", []), dtype=np.float32))

        # backward compatibility with older checkpoints that did not store full histories
        if len(train_losses) == 0:
            train_losses_path = os.path.join(cfg.outdir, "train_losses.npy")
            if os.path.isfile(train_losses_path):
                train_losses = np.load(train_losses_path).astype(np.float32).tolist()
        if len(val_losses) == 0:
            val_losses_path = os.path.join(cfg.outdir, "val_losses.npy")
            if os.path.isfile(val_losses_path):
                val_losses = np.load(val_losses_path).astype(np.float32).tolist()

        # keep history aligned with resumed epoch
        if len(train_losses) > start_epoch:
            train_losses = train_losses[:start_epoch]
        if len(val_losses) > start_epoch:
            val_losses = val_losses[:start_epoch]

        print(f"Resumed from: {resume_path}")
        print(f"Resume epoch: {start_epoch}/{cfg.epochs}")


    feat_mean_t = torch.as_tensor(ds_full.feat_mean, device=cfg.device).view(1, 1, 7)
    feat_std_t  = torch.as_tensor(ds_full.feat_std,  device=cfg.device).view(1, 1, 7)

    
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
        "beta_start": cfg.beta_start,
        "beta_end": cfg.beta_end,
        # ── model identity ──────────────────────────────────
        "model_type": "transformer",          # "transformer" for ParticleDenoiser
        # transformer kwargs (kept for backward compat)
        "d_model": cfg.d_model,
        "nhead": cfg.nhead,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
        # unet kwargs
        "base_ch": 32,
        "channel_mults": (1, 2),
        "attn_levels": (0, 1),
        "pdg_emb_dim": 16,
        "groups": 8,
        "attn_heads": 4,
        # ────────────────────────────────────────────────────
        "val_frac": val_frac,
        "seed": cfg.seed,
        "n_events": len(ds_full),
        "n_train_events": len(train_idx),
        "n_val_events": len(val_idx),
    }
    torch.save(meta, os.path.join(cfg.outdir, "meta.pt"))

    if start_epoch >= cfg.epochs:
        print(f"Checkpoint already reached target epochs ({start_epoch} >= {cfg.epochs}).")
        print("No additional training needed.")
        return

    feat_mean_gpu = torch.tensor(ds_full.feat_mean, device=cfg.device)
    feat_std_gpu = torch.tensor(ds_full.feat_std, device=cfg.device)
    
    for epoch in range(start_epoch, cfg.epochs):
        # TRAIN
        model.train()
        total_train = 0.0
        n_train = 0
        
        pbar = tqdm(dl_train, desc=f"Epoch {epoch+1:03d}/{cfg.epochs} [train]", leave=False)
        for x0, pdg0, mask in pbar:


            x0 = x0.to(cfg.device)
            pdg0 = pdg0.to(cfg.device)
            mask = mask.to(cfg.device)

            B = x0.shape[0]
            t = torch.randint(0, cfg.T, (B,), device=cfg.device)

            noise = torch.randn_like(x0) * mask.unsqueeze(-1)
            x_t = ddpm.q_sample(x0, t, noise)

            pdg_t = q_sample_pdg(pdg0, t, gammas, cfg.n_pdg, mask)

            eps_hat, pdg_logits = model(x_t, t, pdg_t, mask)

            # diffusion loss (token-normalised)
            mse = (eps_hat - noise).pow(2).sum(dim=-1)
            diff_loss = (mse * mask).sum() / mask.sum().clamp(min=1)

            # pdg CE loss
            pdg_loss = F.cross_entropy(pdg_logits[mask], pdg0[mask])

            # Physics loss
            # sqrt_acp_t = ddpm.sqrt_acp[t].view(-1, 1, 1)
            # sqrt_1m_acp_t = ddpm.sqrt_1m_acp[t].view(-1, 1, 1)

            # pred_x0 = (x_t - sqrt_1m_acp_t * eps_hat) / sqrt_acp_t.clamp(min=1e-3)
            # pred_cont = pred_x0 * feat_std_gpu + feat_mean_gpu
            
            # p_logE = pred_cont[..., 0].clamp(min=-10, max=10)
            # p_u = pred_cont[..., 1:4]
            # p_u_mag = torch.norm(p_u, dim=-1, keepdim=True)
            # p_beta = torch.tanh(p_u_mag) * (p_u / (p_u_mag + 1e-12))
            # p_E = torch.exp(p_logE).unsqueeze(-1)
            # p_P = p_E * p_beta
            
            # p_P_x = (p_P[..., 0] * mask).sum(dim=1)
            # p_P_y = (p_P[..., 1] * mask).sum(dim=1)
            
            # p_loss = (p_P_x.pow(2) + p_P_y.pow(2)).mean()

            loss = diff_loss + cfg.lambda_pdg * pdg_loss # + cfg.lambda_pt * p_loss

            # loss = diff_loss + cfg.lambda_pdg * pdg_loss
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
            for x0, pdg0, mask in pbarv:


                x0 = x0.to(cfg.device)
                pdg0 = pdg0.to(cfg.device)
                mask = mask.to(cfg.device)

                B = x0.shape[0]
                t = torch.randint(0, cfg.T, (B,), device=cfg.device)

                noise = torch.randn_like(x0) * mask.unsqueeze(-1)
                x_t = ddpm.q_sample(x0, t, noise)

                pdg_t = q_sample_pdg(pdg0, t, gammas, cfg.n_pdg, mask)

                eps_hat, pdg_logits = model(x_t, t, pdg_t, mask)

                # diffusion loss (token-normalised)
                mse = (eps_hat - noise).pow(2).sum(dim=-1)
                diff_loss = (mse * mask).sum() / mask.sum().clamp(min=1)

                # pdg CE loss
                pdg_loss = F.cross_entropy(pdg_logits[mask], pdg0[mask])

                # Physics loss
                # sqrt_acp_t = ddpm.sqrt_acp[t].view(-1, 1, 1)
                # sqrt_1m_acp_t = ddpm.sqrt_1m_acp[t].view(-1, 1, 1)

                # pred_x0 = (x_t - sqrt_1m_acp_t * eps_hat) / sqrt_acp_t.clamp(min=1e-3)
                # pred_cont = pred_x0 * feat_std_gpu + feat_mean_gpu

                # p_logE = pred_cont[..., 0].clamp(min=-10, max=10)
                # p_u = pred_cont[..., 1:4]
                # p_u_mag = torch.norm(p_u, dim=-1, keepdim=True)
                # p_beta = torch.tanh(p_u_mag) * (p_u / (p_u_mag + 1e-12))
                # p_E = torch.exp(p_logE).unsqueeze(-1)
                # p_P = p_E * p_beta

                # p_P_x = (p_P[..., 0] * mask).sum(dim=1)
                # p_P_y = (p_P[..., 1] * mask).sum(dim=1)
                                                 
                # p_loss = (p_P_x.pow(2) + p_P_y.pow(2)).mean()

                loss = diff_loss + cfg.lambda_pdg * pdg_loss # + cfg.lambda_pt * p_loss

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
                "train_losses": train_losses,
                "val_losses": val_losses,
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

    #model_type = meta.get("model_type", "transformer")
    model_type = 'transformer'  # force to transformer for now since we only have that checkpoint
    if model_type == "unet":
        model = build_unet_denoiser(
            d_model       = meta["d_model"],
            n_pdg         = meta["n_pdg"],
            base_ch       = meta.get("base_ch", 32),
            channel_mults = tuple(meta.get("channel_mults", (1, 2))),
            attn_levels   = tuple(meta.get("attn_levels", (1,))),
            pdg_emb_dim   = meta.get("pdg_emb_dim", 16),
            groups        = meta.get("groups", 8),
            attn_heads    = meta.get("attn_heads", 4),
        ).to(device)
    else:
        model = ParticleDenoiser(
            d_model    = meta["d_model"],
            nhead      = meta["nhead"],
            num_layers = meta["num_layers"],
            dropout    = meta["dropout"],
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


def sample_one_event(meta: dict, ddpm: DDPM, device: str, sample_steps=None):
    multiplicities = np.asarray(meta["multiplicities"], dtype=np.int64)
    K = int(np.random.choice(multiplicities))

    Kmax = int(meta["max_particles"])
    K = max(1, min(K, Kmax))

    mask = np.ones((K,), dtype=np.bool_)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)

    # init PDG uniformly
    n_pdg = int(meta["n_pdg"])
    pdg_init = torch.randint(0, n_pdg, (1, K), device=device)
    pdg_init = pdg_init * mask_t.long()

    with torch.no_grad():
        x_norm, pdg_idx = ddpm.sample(mask_t, pdg_init, sample_steps=sample_steps)

    x_norm = x_norm[0].cpu().numpy()[:K]        # (K,7)
    pdg_idx = pdg_idx[0].cpu().numpy()[:K]      # (K,)

    mean = np.asarray(meta["feat_mean"], dtype=np.float32)
    std  = np.asarray(meta["feat_std"],  dtype=np.float32)
    cont = x_norm * std + mean                   # (K,7)

    logE = cont[:, 0]
    u = cont[:, 1:4]
    pos = cont[:, 4:7]

    logE_lo = mean[0] - 8 * std[0]
    logE_hi = mean[0] + 8 * std[0]
    E = np.exp(np.clip(logE, logE_lo, logE_hi))
    beta = beta_squash_np(u)

    # convert pdg_idx -> actual pdg
    idx_to_pdg = meta["idx_to_pdg"]
    pdg = np.array([idx_to_pdg[int(i)] for i in pdg_idx], dtype=np.int64)

    # output format (K,8): [pdg, E, betax, betay, betaz, x, y, z]
    out = np.concatenate([pdg[:,None], E[:,None], beta, pos], axis=1).astype(np.float32)
    return out


def sample_event_batch(meta: dict, ddpm: DDPM, device: str, batch_size: int = 32, sample_steps=None):
    """
    Batch sampling method with vectorized post-processing.
    """
    multiplicities = np.asarray(meta["multiplicities"], dtype=np.int64)
    Kmax_limit = int(meta["max_particles"])

    Ks = np.random.choice(multiplicities, size=batch_size).astype(np.int64)
    Ks = np.clip(Ks, 1, Kmax_limit)
    
    # DYNAMIC PADDING: Only pad to the max K in the current batch!
    cur_Kmax = int(Ks.max())

    mask = np.zeros((batch_size, cur_Kmax), dtype=np.bool_)
    for i, k in enumerate(Ks):
        mask[i, : int(k)] = True
    mask_t = torch.from_numpy(mask).to(device)

    n_pdg = int(meta["n_pdg"])
    pdg_init = torch.randint(0, n_pdg, (batch_size, cur_Kmax), device=device)
    pdg_init = pdg_init * mask_t.long()

    with torch.no_grad():
        x_norm, pdg_idx = ddpm.sample(mask_t, pdg_init, sample_steps=sample_steps)

    x_norm = x_norm.cpu().numpy()     # (B,cur_Kmax,7)
    pdg_idx = pdg_idx.cpu().numpy()   # (B,cur_Kmax)

    # Prepare vectorized post-processing (no Python loop)
    mean = np.asarray(meta["feat_mean"], dtype=np.float32)            # (7,)
    std = np.asarray(meta["feat_std"], dtype=np.float32)              # (7,)
    idx_to_pdg_arr = np.array(list(meta["idx_to_pdg"].values()), dtype=np.int64)  # (n_pdg,)

    # Vectorized denormalization: (B, cur_Kmax, 7)
    cont = x_norm * std[np.newaxis, np.newaxis, :] + mean[np.newaxis, np.newaxis, :]
    
    logE = cont[..., 0]      # (B, cur_Kmax)
    u = cont[..., 1:4]       # (B, cur_Kmax, 3)
    pos = cont[..., 4:7]     # (B, cur_Kmax, 3)

    # Vectorized exponential and beta squashing
    logE_lo = mean[0] - 8 * std[0]
    logE_hi = mean[0] + 8 * std[0]
    E = np.exp(np.clip(logE, logE_lo, logE_hi))  # (B, cur_Kmax)
    beta = beta_squash_np_batch(u)  # (B, cur_Kmax, 3)

    # Vectorized PDG mapping
    pdg = idx_to_pdg_arr[pdg_idx]   # (B, cur_Kmax)

    # Stack all features: (B, cur_Kmax, 8)
    all_particles = np.stack([pdg, E, beta[..., 0], beta[..., 1], 
                              beta[..., 2], pos[..., 0], pos[..., 1], 
                              pos[..., 2]], axis=-1).astype(np.float32)

    # Trim to variable lengths and return as list
    events = [all_particles[i, :Ks[i]] for i in range(batch_size)]
    return events


def sample_event_batch_given_Ks(meta: dict, ddpm: DDPM, device: str, Ks: np.ndarray, sample_steps=None):
    """
    Batch sampling method using pre-sampled K values (allows sorting to minimize padding waste).
    """
    batch_size = len(Ks)
    cur_Kmax = int(Ks.max())

    mask = np.zeros((batch_size, cur_Kmax), dtype=np.bool_)
    for i, k in enumerate(Ks):
        mask[i, : int(k)] = True
    mask_t = torch.from_numpy(mask).to(device)

    n_pdg = int(meta["n_pdg"])
    pdg_init = torch.randint(0, n_pdg, (batch_size, cur_Kmax), device=device)
    pdg_init = pdg_init * mask_t.long()

    with torch.no_grad():
        x_norm, pdg_idx = ddpm.sample(mask_t, pdg_init, sample_steps=sample_steps)

    x_norm = x_norm.cpu().numpy()     # (B,cur_Kmax,7)
    pdg_idx = pdg_idx.cpu().numpy()   # (B,cur_Kmax)

    # Prepare vectorized post-processing
    mean = np.asarray(meta["feat_mean"], dtype=np.float32)            
    std = np.asarray(meta["feat_std"], dtype=np.float32)              
    idx_to_pdg_arr = np.array(list(meta["idx_to_pdg"].values()), dtype=np.int64)  

    # Vectorized denormalization: (B, cur_Kmax, 7)
    cont = x_norm * std[np.newaxis, np.newaxis, :] + mean[np.newaxis, np.newaxis, :]
    
    logE = cont[..., 0]      # (B, cur_Kmax)
    u = cont[..., 1:4]       # (B, cur_Kmax, 3)
    pos = cont[..., 4:7]     # (B, cur_Kmax, 3)

    # Vectorized exponential and beta squashing
    logE_lo = mean[0] - 8 * std[0]
    logE_hi = mean[0] + 8 * std[0]
    E = np.exp(np.clip(logE, logE_lo, logE_hi))  # (B, cur_Kmax)
    beta = beta_squash_np_batch(u)  # (B, cur_Kmax, 3)

    # Vectorized PDG mapping
    pdg = idx_to_pdg_arr[pdg_idx]   # (B, cur_Kmax)

    # Stack all features: (B, cur_Kmax, 8)
    all_particles = np.stack([pdg, E, beta[..., 0], beta[..., 1], 
                              beta[..., 2], pos[..., 0], pos[..., 1], 
                              pos[..., 2]], axis=-1).astype(np.float32)

    # Trim to variable lengths and return as list
    events = [all_particles[i, :Ks[i]] for i in range(batch_size)]
    return events






def sample(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    outdir = args.outdir
    n_events = args.n_events
    sample_steps = args.sample_steps if hasattr(args, 'sample_steps') else None
    
    meta, model, ddpm = load_meta_and_model(outdir, device)
    
    if sample_steps is not None:
        print(f"Using {sample_steps} sampling steps (trained with {meta['T']} steps)")
    else:
        print(f"Using {meta['T']} sampling steps")

    # Optimized batch sampling with vectorized post-processing
    batch_size = 1  # Increased from 32 for better GPU utilization
    print(f"Batch sampling enabled with batch size = {batch_size}")


    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Start GPU monitor
    monitor = GPUMonitor(interval=0.2)
    monitor.start()
    
    start_time = time.time()
    from tqdm import trange
    events = []

    # Single sampling loop
    for _ in trange(n_events, desc="Generating events"):
        events.append(sample_one_event(meta, ddpm, device, sample_steps=sample_steps))

    # # Pre-sample all lengths and sort them for Bucketing
    # # This prevents the padding waste in O(N^2) Attention computation
    # multiplicities = np.asarray(meta["multiplicities"], dtype=np.int64)
    # Kmax_limit = int(meta["max_particles"])
    # all_Ks = np.random.choice(multiplicities, size=n_events).astype(np.int64)
    # all_Ks = np.clip(all_Ks, 1, Kmax_limit)
    # all_Ks.sort()  # Sort to group similar lengths into the same batch

    # # Optimized: Batch length-bucketing loop
    # for i in trange(0, n_events, batch_size, desc="Generating events"):
    #     Ks_batch = all_Ks[i:i+batch_size]
    #     events.extend(
    #         sample_event_batch_given_Ks(
    #             meta,
    #             ddpm,
    #             device,
    #             Ks_batch,
    #             sample_steps=sample_steps,
    #         )
    #     )
    
    # # Shuffle back the events because we sorted them by length
    # import random
    # random.shuffle(events)
    
    end_time = time.time()
    monitor.stop()
    monitor.join()

    total_time = end_time - start_time
    print(f"\n" + "="*60)
    print(f"Generated {n_events} events in {total_time:.2f} seconds")
    avg_time = total_time / n_events
    print(f"Average time per event: {avg_time*1000:.3f} ms")
    print(f"Throughput: {n_events/total_time:.1f} events/sec")
    print("="*60)

    monitor.print_stats()
    print("="*60)
    
    out_path = os.path.join(outdir, f"generated_events_{sample_steps}steps_20k.npy")
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
    train_parser.add_argument(
        '--resume',
        nargs='?',
        const='auto',
        default=None,
        type=str,
        help='Resume training from checkpoint. Use --resume to load <outdir>/ckpt_last.pt or pass a checkpoint path.'
    )
    
    # Sample
    sample_parser = subparsers.add_parser('sample', help='Generate synthetic events')
    sample_parser.add_argument('--outdir', type=str, default=cfg.outdir, help='Model directory')
    sample_parser.add_argument('--n_events', type=int, default=cfg.n_events, help='Number of events')
    sample_parser.add_argument('--sample_steps', type=int, default=None, help='Number of sampling steps (default: use training steps)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'sample':
        sample(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()