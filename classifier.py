'''
This script is used for generation quality evaluation using a classifier. It prepares data for training a classifier to distinguish between real and generated samples, trains the classifier, and evaluates its performance using AUC scores.
'''




import numpy as np
import random
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument('--data', action='store_true', help='Making 2D tabular data for classifier training and evaluation')
parser.add_argument('--run', action='store_true', help='Run the 2D tabular classifier training and evaluation')
parser.add_argument('--data_energy_betas', action='store_true', help='Making 2D tabular data with only energy and betas (4 features)')
parser.add_argument('--run_energy_betas', action='store_true', help='Run classifier training on energy and betas data (4 features)')
parser.add_argument('--data_3d', action='store_true', help='Making 3D data [events, hits, features] for GNN')
parser.add_argument('--run_gnn', action='store_true', help='Run the GNN/3D classifier training and evaluation')
parser.add_argument('--data_3d_beta', action='store_true', help='Making 3D data with only energy and betas (4 features) for GNN')
parser.add_argument('--run_gnn_beta', action='store_true', help='Run the GNN/3D classifier training with energy and betas only')
parser.add_argument('--gnn_ablation', action='store_true', help='Run GNN feature ablation study (15 epochs, avg last 3)')
parser.add_argument('--steps', type=int, default=1000, help='Number of steps to load generated data from (default: 1000)')
args = parser.parse_args()

# ============================================================================
# SHARED CLASSES AND FUNCTIONS FOR GNN/NEURAL NETWORKS
# ============================================================================
# These are defined here to avoid duplication across different execution paths

# Import PyTorch modules if needed for GNN
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class VariableLengthEventDataset(Dataset):
    """
    Dataset for variable-length event data.
    Handles cases where events (point clouds) have different numbers of particles.
    """
    def __init__(self, X_data, y_data, n_features=7):
        self.X_data = X_data
        self.y_data = y_data
        self.n_features = n_features
        
    def __len__(self):
        return len(self.X_data)
        
    def __getitem__(self, idx):
        x = torch.tensor(self.X_data[idx], dtype=torch.float32)
        # If an event has 0 hits, create dummy zero padding to prevent model crashes
        if x.shape[0] == 0:
            x = torch.zeros((1, self.n_features), dtype=torch.float32)
        y = torch.tensor(self.y_data[idx], dtype=torch.float32)
        return x, y


def create_collate_fn(n_features=7):
    """
    Factory function to create a collate function for variable-length batches.
    
    Args:
        n_features: Number of features per particle (e.g., 7 for full or 4 for energy+betas)
    
    Returns:
        Collate function for DataLoader
    """
    def collate_fn(batch):
        xs, ys = zip(*batch)
        max_len = max(x.size(0) for x in xs)
        
        padded_xs = torch.zeros((len(xs), max_len, n_features), dtype=torch.float32)
        masks = torch.zeros((len(xs), max_len, 1), dtype=torch.float32)
        
        for i, x in enumerate(xs):
            length = x.size(0)
            padded_xs[i, :length, :] = x
            masks[i, :length, :] = 1.0
            
        ys = torch.stack(ys).unsqueeze(1)
        return padded_xs, ys, masks
    
    return collate_fn


class MaskedPointCloudNet(nn.Module):
    """
    A Deep Sets model for point cloud classification with masking support.
    Handles variable-length sequences via mean-pooling with masks.
    """
    def __init__(self, in_features=7, hidden_dim=64):
        super().__init__()
        self.particle_mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x, mask):
        """
        Args:
            x: [Batch, MaxHits, Features] - padded point cloud data
            mask: [Batch, MaxHits, 1] - binary mask for valid particles
        
        Returns:
            out: [Batch, 1] - classification logits
        """
        particle_features = self.particle_mlp(x)  # [Batch, MaxHits, hidden_dim]
        
        # Apply mask to zero out padded positions
        masked_features = particle_features * mask
        
        # Compute mean pooling: only sum over valid particles
        sum_features = masked_features.sum(dim=1)  # [Batch, hidden_dim]
        valid_lengths = mask.sum(dim=1).clamp(min=1.0)  # [Batch, 1]
        
        global_feature = sum_features / valid_lengths  # Properly masked mean
        
        out = self.global_mlp(global_feature)
        return out


# ============================================================================
# Utility functions
def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)

def train_classifier(X_train, y_train, X_test, y_test, seed):
    """Train Random Forest classifier and return AUC score"""
    # Train classifier
    clf = RandomForestClassifier(
        n_estimators=50, 
        max_depth=5, 
        random_state=seed, 
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    
    return auc, y_pred_proba, clf

def beta_squash_np(u: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Map 3 continuous values u to a beta vector within the unit sphere.
    """
    u = np.asarray(u, dtype=np.float64)
    norm = np.linalg.norm(u, axis=-1, keepdims=True)
    beta = np.tanh(norm + eps) * (u / (norm + eps))
    return beta

def load_events(path: str):
    """
    Load events from a .npy file, handling different possible formats.
    """
    arr = np.load(path, allow_pickle=True)
    
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return list(arr)
    
    if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[-1] >= 4:
        return [arr[i] for i in range(arr.shape[0])]
    
    raise ValueError(f"Unrecognized format in {path}")

def sanitize_event(ev, me=0.000511):
    """
    Standardize event data into a consistent format.
    
    Return: (pdg, px, py, pz, Eabs, E_signed, beta_mag, x, y, z, betax, betay, betaz)
    """
    ev = np.asarray(ev)

    # Case A: generated / explicit PDG format: [pdg, E, betax, betay, betaz, x, y, z]
    if ev.ndim == 2 and ev.shape[1] >= 8:
        pdg = ev[:, 0].astype(np.int64, copy=False)
        Eabs = np.abs(ev[:, 1].astype(np.float64, copy=False))
        betax = ev[:, 2].astype(np.float64, copy=False)
        betay = ev[:, 3].astype(np.float64, copy=False)
        betaz = ev[:, 4].astype(np.float64, copy=False)
        x = ev[:, 5].astype(np.float64, copy=False)
        y = ev[:, 6].astype(np.float64, copy=False)
        z = ev[:, 7].astype(np.float64, copy=False)

        beta = np.stack([betax, betay, betaz], axis=1)
        beta_mag = np.linalg.norm(beta, axis=1)
        pvec = Eabs[:, None] * beta
        px, py, pz = pvec[:, 0], pvec[:, 1], pvec[:, 2]
        E_signed = np.where(pdg == -11, -Eabs, Eabs)

        return (pdg, px, py, pz, Eabs, E_signed, beta_mag,
                x, y, z, betax, betay, betaz)

    # Case B: real guineapig format [E_signed, betax, betay, betaz, x, y, z]
    if ev.ndim == 2 and ev.shape[1] >= 7:
        E_signed = ev[:, 0].astype(np.float64, copy=False)
        betax = ev[:, 1].astype(np.float64, copy=False)
        betay = ev[:, 2].astype(np.float64, copy=False)
        betaz = ev[:, 3].astype(np.float64, copy=False)
        x = ev[:, 4].astype(np.float64, copy=False)
        y = ev[:, 5].astype(np.float64, copy=False)
        z = ev[:, 6].astype(np.float64, copy=False)

        pdg = np.where(E_signed >= 0.0, 11, -11).astype(np.int64)
        Eabs = np.abs(E_signed)
        beta = np.stack([betax, betay, betaz], axis=1)
        beta_mag = np.linalg.norm(beta, axis=1)
        pvec = Eabs[:, None] * beta
        px, py, pz = pvec[:, 0], pvec[:, 1], pvec[:, 2]

        return (pdg, px, py, pz, Eabs, E_signed, beta_mag,
                x, y, z, betax, betay, betaz)

    # fallback: empty
    empty = np.array([], dtype=np.float64)
    return (
        empty.astype(np.int64),
        empty, empty, empty,
        empty, empty, empty,
        empty, empty, empty,
        empty, empty, empty
    )

def extract_species(events, pdgs=None, me=0.000511):
    """
    Extract physical quantities for specified particle species from a list of events.
    """
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
        "E": cat_or_empty(E_list),
        "E_abs": cat_or_empty(E_list),
        "E_signed": E_signed_all,
        "beta_mag": cat_or_empty(bmag_list),
        "x": cat_or_empty(x_list), "y": cat_or_empty(y_list), "z": cat_or_empty(z_list),
        "betax": cat_or_empty(bx_list),
        "betay": cat_or_empty(by_list),
        "betaz": cat_or_empty(bz_list),
    }

def prepare_classifier_data(real_sp, gen_sp, output_path, max_samples=500000):
    """
    Prepare data for classifier training to evaluate sampling quality.
    
    7 features: E, betax, betay, betaz, x, y, z
    
    Args:
        real_sp: Dictionary with real data
        gen_sp: Dictionary with generated data
        output_path: Path to save the data file
        max_samples: Maximum samples per class (None = use all)
    
    Returns:
        Dictionary with X (features), y (labels), and feature names
    """
    # Extract 7 features for real data
    real_features = np.column_stack([
        real_sp['E'],
        real_sp['betax'],
        real_sp['betay'],
        real_sp['betaz'],
        real_sp['x'],
        real_sp['y'],
        real_sp['z']
    ])
    
    # Extract 7 features for generated data
    gen_features = np.column_stack([
        gen_sp['E'],
        gen_sp['betax'],
        gen_sp['betay'],
        gen_sp['betaz'],
        gen_sp['x'],
        gen_sp['y'],
        gen_sp['z']
    ])
    
    # Remove any rows with NaN or Inf
    real_mask = np.all(np.isfinite(real_features), axis=1)
    gen_mask = np.all(np.isfinite(gen_features), axis=1)
    
    real_features = real_features[real_mask]
    gen_features = gen_features[gen_mask]
    
    print(f"Real data: {len(real_features)} samples")
    print(f"Generated data: {len(gen_features)} samples")
    
    # Downsample if requested
    if max_samples is not None:
        if len(real_features) > max_samples:
            indices = np.random.choice(len(real_features), max_samples, replace=False)
            real_features = real_features[indices]
            print(f"  Downsampled real to {max_samples}")
        
        if len(gen_features) > max_samples:
            indices = np.random.choice(len(gen_features), max_samples, replace=False)
            gen_features = gen_features[indices]
            print(f"  Downsampled generated to {max_samples}")
    
    # Create labels: 0 = real, 1 = generated
    real_labels = np.zeros(len(real_features), dtype=np.int32)
    gen_labels = np.ones(len(gen_features), dtype=np.int32)
    
    # Combine data
    X = np.vstack([real_features, gen_features])
    y = np.concatenate([real_labels, gen_labels])
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    # Feature names
    feature_names = ['E', 'betax', 'betay', 'betaz', 'x', 'y', 'z']
    
    # Save data
    data_dict = {
        'X': X,
        'y': y,
        'feature_names': feature_names,
        'n_real': len(real_features),
        'n_generated': len(gen_features),
        'description': 'Classifier data: 0=real, 1=generated. Lower AUC = better sampling quality.'
    }
    
    np.save(output_path, data_dict)
    print(f"\n✓ Saved classifier data to: {output_path}")
    print(f"  Total samples: {len(X)}")
    print(f"  Features: {feature_names}")
    print(f"  Real (0): {np.sum(y==0)}, Generated (1): {np.sum(y==1)}")
    print(f"\nNote: Lower classifier AUC indicates better sampling quality")
    
    return data_dict

def prepare_classifier_data_energy_betas(real_sp, gen_sp, output_path, max_samples=500000):
    """
    Prepare data for classifier training with only energy and betas (4 features).
    
    4 features: E, betax, betay, betaz (no position information)
    
    Args:
        real_sp: Dictionary with real data
        gen_sp: Dictionary with generated data
        output_path: Path to save the data file
        max_samples: Maximum samples per class (None = use all)
    
    Returns:
        Dictionary with X (features), y (labels), and feature names
    """
    # Extract 4 features for real data (energy and betas only)
    real_features = np.column_stack([
        real_sp['E'],
        real_sp['betax'],
        real_sp['betay'],
        real_sp['betaz']
    ])
    
    # Extract 4 features for generated data (energy and betas only)
    gen_features = np.column_stack([
        gen_sp['E'],
        gen_sp['betax'],
        gen_sp['betay'],
        gen_sp['betaz']
    ])
    
    # Remove any rows with NaN or Inf
    real_mask = np.all(np.isfinite(real_features), axis=1)
    gen_mask = np.all(np.isfinite(gen_features), axis=1)
    
    real_features = real_features[real_mask]
    gen_features = gen_features[gen_mask]
    
    print(f"Real data: {len(real_features)} samples")
    print(f"Generated data: {len(gen_features)} samples")
    
    # Downsample if requested
    if max_samples is not None:
        if len(real_features) > max_samples:
            indices = np.random.choice(len(real_features), max_samples, replace=False)
            real_features = real_features[indices]
            print(f"  Downsampled real to {max_samples}")
        
        if len(gen_features) > max_samples:
            indices = np.random.choice(len(gen_features), max_samples, replace=False)
            gen_features = gen_features[indices]
            print(f"  Downsampled generated to {max_samples}")
    
    # Create labels: 0 = real, 1 = generated
    real_labels = np.zeros(len(real_features), dtype=np.int32)
    gen_labels = np.ones(len(gen_features), dtype=np.int32)
    
    # Combine data
    X = np.vstack([real_features, gen_features])
    y = np.concatenate([real_labels, gen_labels])
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    # Feature names
    feature_names = ['E', 'betax', 'betay', 'betaz']
    
    # Save data
    data_dict = {
        'X': X,
        'y': y,
        'feature_names': feature_names,
        'n_real': len(real_features),
        'n_generated': len(gen_features),
        'description': 'Classifier data (energy & betas only): 0=real, 1=generated. Lower AUC = better sampling quality.'
    }
    
    np.save(output_path, data_dict)
    print(f"\n✓ Saved classifier data (energy & betas) to: {output_path}")
    print(f"  Total samples: {len(X)}")
    print(f"  Features: {feature_names}")
    print(f"  Real (0): {np.sum(y==0)}, Generated (1): {np.sum(y==1)}")
    print(f"\nNote: Lower classifier AUC indicates better sampling quality")
    
    return data_dict

def prepare_data_per_event(events, pdgs=None, me=0.000511, max_hits=None):
    """
    Prepare data in [events, hits in events, features] format.
    features: [E, betax, betay, betaz, x, y, z]
    
    Args:
        events: List of event arrays
        pdgs: List of particle PDG codes to include (None = all)
        me: Electron mass
        max_hits: Maximum hits per event to pad/truncate to (None = return list of variable length arrays)
        
    Returns:
        event_data: Array or list of arrays with shape [events, hits, 7]
    """
    event_data = []
    
    for ev in events:
        pdg, px, py, pz, Eabs, E_signed, bmag, x, y, z, betax, betay, betaz = sanitize_event(ev, me=me)
        
        if pdgs is None:
            sel = np.ones(len(px), dtype=bool)
        else:
            sel = np.zeros(len(px), dtype=bool)
            for code in pdgs:
                sel |= (pdg == code)
        
        if not np.any(sel):
            if max_hits is not None:
                event_data.append(np.zeros((max_hits, 7)))
            else:
                event_data.append(np.zeros((0, 7)))
            continue
            
        features = np.column_stack([
            Eabs[sel],
            betax[sel],
            betay[sel],
            betaz[sel],
            x[sel],
            y[sel],
            z[sel]
        ])
        
        # Remove any rows with NaN or Inf
        mask = np.all(np.isfinite(features), axis=1)
        features = features[mask]
        
        if max_hits is not None:
            if len(features) > max_hits:
                # Randomly sample or just take the first max_hits
                # Here we take the first max_hits
                features = features[:max_hits]
            elif len(features) < max_hits:
                # Zero padding
                padding = np.zeros((max_hits - len(features), 7))
                features = np.vstack([features, padding])
                
        event_data.append(features)
        
    if max_hits is not None:
        return np.array(event_data)
    return event_data

def prepare_data_per_event_energy_betas(events, pdgs=None, me=0.000511, max_hits=None):
    """
    Prepare data in [events, hits in events, features] format with ONLY energy and betas.
    features: [E, betax, betay, betaz]
    
    Args:
        events: List of event arrays
        pdgs: List of particle PDG codes to include (None = all)
        me: Electron mass
        max_hits: Maximum hits per event to pad/truncate to (None = return list of variable length arrays)
        
    Returns:
        event_data: Array or list of arrays with shape [events, hits, 4]
    """
    event_data = []
    
    for ev in events:
        pdg, px, py, pz, Eabs, E_signed, bmag, x, y, z, betax, betay, betaz = sanitize_event(ev, me=me)
        
        if pdgs is None:
            sel = np.ones(len(px), dtype=bool)
        else:
            sel = np.zeros(len(px), dtype=bool)
            for code in pdgs:
                sel |= (pdg == code)
        
        if not np.any(sel):
            if max_hits is not None:
                event_data.append(np.zeros((max_hits, 4)))
            else:
                event_data.append(np.zeros((0, 4)))
            continue
            
        features = np.column_stack([
            Eabs[sel],
            betax[sel],
            betay[sel],
            betaz[sel]
        ])
        
        # Remove any rows with NaN or Inf
        mask = np.all(np.isfinite(features), axis=1)
        features = features[mask]
        
        if max_hits is not None:
            if len(features) > max_hits:
                features = features[:max_hits]
            elif len(features) < max_hits:
                padding = np.zeros((max_hits - len(features), 4))
                features = np.vstack([features, padding])
                
        event_data.append(features)
        
    if max_hits is not None:
        return np.array(event_data)
    return event_data

# Data preparation
if args.data:

    # Making data into [events, particles, features], Grpah neural network
    # Seperatte events instead of particle


    steps = args.steps
    REAL_DATA_PATH = "/work/submit/haoyun22/FCC-Beam-Background/DM/guineapig_raw_trimmed.npy"
    GEN_DATA_PATH = f"/work/submit/haoyun22/FCC-Beam-Background/DM/results_nosquash_cosine_chargeloss/generated_events_{steps}steps.npy"

    print(f"Real data path: {REAL_DATA_PATH}")
    print(f"Generated data path: {GEN_DATA_PATH}")

    print("Loading real data...")
    real_events = load_events(REAL_DATA_PATH)
    n_real = len(real_events)
    print(f"Loaded {n_real} real events")

    print("\nLoading generated data...")
    gen_events = load_events(GEN_DATA_PATH)
    n_gen = len(gen_events)
    print(f"Loaded {n_gen} generated events")

    print(f"\nStat: {n_real} real events, {n_gen} generated events")

    species_list = [
    {"name": "e−",  "pdgs": [11],   "tag": "eminus"},
    {"name": "e+",  "pdgs": [-11],  "tag": "eplus"},
    {"name": "all", "pdgs": None,   "tag": "all"},
    ]

    real_data = {}
    gen_data = {}

    for sp in species_list:
        print(f"\nExtracting {sp['name']}...")
        
        real_data[sp['tag']] = extract_species(real_events, sp['pdgs'])
        gen_data[sp['tag']] = extract_species(gen_events, sp['pdgs'])
        
        print(f"  Real: {len(real_data[sp['tag']]['E'])} particles")
        print(f"  Gen:  {len(gen_data[sp['tag']]['E'])} particles")

    print("\n✓ All species extracted!")

    # Save the data
    print("\nPreparing classifier data...")
    tag = 'all'
    real_sp = real_data[tag]
    gen_sp = gen_data[tag]

    output_path = f'/work/submit/haoyun22/FCC-Beam-Background/DM/results_nosquash_cosine_chargeloss/classifier_data_{steps}steps.npy'

    classifier_data = prepare_classifier_data(
        real_sp=real_sp,
        gen_sp=gen_sp,
        output_path=output_path
    )

    print(f"\n✓ Data preparation complete!, saved to: {output_path}")


# Data preparation - energy and betas only
if args.data_energy_betas:

    steps = args.steps
    REAL_DATA_PATH = "/work/submit/haoyun22/FCC-Beam-Background/DM/guineapig_raw_trimmed.npy"
    GEN_DATA_PATH = f"/work/submit/haoyun22/FCC-Beam-Background/DM/results_nosquash_cosine_chargeloss/generated_events_{steps}steps.npy"

    print(f"Real data path: {REAL_DATA_PATH}")
    print(f"Generated data path: {GEN_DATA_PATH}")

    print("Loading real data...")
    real_events = load_events(REAL_DATA_PATH)
    n_real = len(real_events)
    print(f"Loaded {n_real} real events")

    print("\nLoading generated data...")
    gen_events = load_events(GEN_DATA_PATH)
    n_gen = len(gen_events)
    print(f"Loaded {n_gen} generated events")

    print(f"\nStat: {n_real} real events, {n_gen} generated events")

    species_list = [
    {"name": "e−",  "pdgs": [11],   "tag": "eminus"},
    {"name": "e+",  "pdgs": [-11],  "tag": "eplus"},
    {"name": "all", "pdgs": None,   "tag": "all"},
    ]

    real_data = {}
    gen_data = {}

    for sp in species_list:
        print(f"\nExtracting {sp['name']}...")
        
        real_data[sp['tag']] = extract_species(real_events, sp['pdgs'])
        gen_data[sp['tag']] = extract_species(gen_events, sp['pdgs'])
        
        print(f"  Real: {len(real_data[sp['tag']]['E'])} particles")
        print(f"  Gen:  {len(gen_data[sp['tag']]['E'])} particles")

    print("\n✓ All species extracted!")

    # Save the data
    print("\nPreparing classifier data (energy & betas only)...")
    tag = 'all'
    real_sp = real_data[tag]
    gen_sp = gen_data[tag]

    output_path = f'/work/submit/haoyun22/FCC-Beam-Background/DM/results_nosquash_cosine_chargeloss/classifier_data_energy_betas_{steps}steps.npy'

    classifier_data = prepare_classifier_data_energy_betas(
        real_sp=real_sp,
        gen_sp=gen_sp,
        output_path=output_path
    )

    print(f"\n✓ Data preparation complete!, saved to: {output_path}")


# Train and evaluate classifier
if args.run:
    steps = args.steps
    path = f'/work/submit/haoyun22/FCC-Beam-Background/DM/results_nosquash_cosine_chargeloss/classifier_data_{steps}steps.npy'
    data = np.load(path, allow_pickle=True).item()
    X = data['X']
    y = data['y']
    feature_names = data['feature_names']

    print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {feature_names}")

    # Train over 10 different random seeds
    n_runs = 50
    auc_scores = []

    for run in range(n_runs):
        seed = random.randint(0, 10000000)
        set_seed(seed)
        
        print(f"Run {run+1}/{n_runs} | Seed: {seed}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train classifier
        auc, y_pred_proba, clf = train_classifier(
            X_train_scaled, y_train, X_test_scaled, y_test, seed
        )
        
        auc_scores.append(auc)
        print(f"  AUC: {auc:.4f}\n")

    # Compute statistics
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)

    print(f"FINAL RESULTS OVER {n_runs} RUNS")
    print(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Min AUC:  {np.min(auc_scores):.4f}")
    print(f"Max AUC:  {np.max(auc_scores):.4f}")
    print(f"\nAll AUC scores: {[f'{auc:.4f}' for auc in auc_scores]}")

    # Feature importance from the last trained model
    importances = clf.feature_importances_
    feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE (Last Run)")
    print(f"{'='*60}")
    for feat, imp in feature_importance:
        print(f"  {feat:10s}: {imp:.4f}")


# Train and evaluate classifier - energy and betas only
if args.run_energy_betas:
    steps = args.steps
    path = f'/work/submit/haoyun22/FCC-Beam-Background/DM/results_nosquash_cosine_chargeloss/classifier_data_energy_betas_{steps}steps.npy'
    data = np.load(path, allow_pickle=True).item()
    X = data['X']
    y = data['y']
    feature_names = data['feature_names']

    print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {feature_names}")

    # Train over 50 different random seeds
    n_runs = 500
    auc_scores = []

    for run in range(n_runs):
        seed = random.randint(0, 10000000)
        set_seed(seed)
        
        print(f"Run {run+1}/{n_runs} | Seed: {seed}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train classifier
        auc, y_pred_proba, clf = train_classifier(
            X_train_scaled, y_train, X_test_scaled, y_test, seed
        )
        
        auc_scores.append(auc)
        print(f"  AUC: {auc:.4f}\n")

    # Compute statistics
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)

    print(f"FINAL RESULTS OVER {n_runs} RUNS (Energy & Betas Only)")
    print(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Min AUC:  {np.min(auc_scores):.4f}")
    print(f"Max AUC:  {np.max(auc_scores):.4f}")
    print(f"\nAll AUC scores: {[f'{auc:.4f}' for auc in auc_scores]}")

    # Feature importance from the last trained model
    importances = clf.feature_importances_
    feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE (Last Run)")
    print(f"{'='*60}")
    for feat, imp in feature_importance:
        print(f"  {feat:10s}: {imp:.4f}")

# Extract 3D numpy array for GNN
if args.data_3d:
    steps = args.steps
    REAL_DATA_PATH = "/work/submit/haoyun22/FCC-Beam-Background/DM/guineapig_raw_trimmed.npy"
    GEN_DATA_PATH = f"/work/submit/haoyun22/FCC-Beam-Background/DM/results_nosquash_cosine_chargeloss/generated_events_{steps}steps.npy"
    
    print("Loading real and generated data for 3D preparation...")
    real_events = load_events(REAL_DATA_PATH)
    gen_events = load_events(GEN_DATA_PATH)
    
    # max_hits=None will keep variable lengths. Result is a list of arrays.
    print("\nPreparing variable length 3D sequential data [events, hits, features]...")
    real_data_3d = prepare_data_per_event(real_events, pdgs=None, max_hits=None)
    gen_data_3d = prepare_data_per_event(gen_events, pdgs=None, max_hits=None)
    
    # Labels: 0 for real, 1 for generated
    real_labels = np.zeros(len(real_data_3d), dtype=np.int32)
    gen_labels = np.ones(len(gen_data_3d), dtype=np.int32)
    
    n_real = len(real_data_3d)
    n_gen = len(gen_data_3d)
    
    print(f"\nClass imbalance before balancing:")
    print(f"  Real (0): {n_real}, Generated (1): {n_gen}")
    
    # Balance classes: downsample the larger class to match the smaller one
    min_samples = min(n_real, n_gen)
    
    if n_real > n_gen:
        # Downsample real data
        real_indices = np.random.choice(n_real, min_samples, replace=False)
        real_data_3d = [real_data_3d[i] for i in real_indices]
        real_labels = real_labels[real_indices]
        print(f"  Downsampled Real (0): {n_real} -> {min_samples}")
    elif n_gen > n_real:
        # Downsample generated data
        gen_indices = np.random.choice(n_gen, min_samples, replace=False)
        gen_data_3d = [gen_data_3d[i] for i in gen_indices]
        gen_labels = gen_labels[gen_indices]
        print(f"  Downsampled Generated (1): {n_gen} -> {min_samples}")
    
    X_3d = np.array(real_data_3d + gen_data_3d, dtype=object) # List of variable length arrays
    y_3d = np.concatenate([real_labels, gen_labels])
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X_3d))
    X_3d = X_3d[shuffle_idx]
    y_3d = y_3d[shuffle_idx]
    
    output_path_3d = f'/work/submit/haoyun22/FCC-Beam-Background/DM/results_nosquash_cosine_chargeloss/classifier_data_3d_var_{steps}steps.npy'
    
    data_dict_3d = {
        'X': X_3d,   # Object array storing variable length 2D arrays
        'y': y_3d,
        'feature_names': ['E', 'betax', 'betay', 'betaz', 'x', 'y', 'z'],
        'n_real': np.sum(y_3d==0),
        'n_generated': np.sum(y_3d==1),
        'description': 'Variable length 3D Classifier data [events, variable_hits, features=7]: 0=real, 1=generated. Balanced 1:1 class ratio.'
    }
    
    np.save(output_path_3d, data_dict_3d)
    print(f"\n✓ Saved variable length 3D classifier data to: {output_path_3d}")
    print(f"  Total events: {len(X_3d)}")
    print(f"  Real (0): {np.sum(y_3d==0)}, Generated (1): {np.sum(y_3d==1)}")
    print(f"  Class balance: 1:1 ✓")


# Train and evaluate GNN
if args.run_gnn:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    
    steps = args.steps
    path_3d = f'/work/submit/haoyun22/FCC-Beam-Background/DM/results_nosquash_cosine_chargeloss/classifier_data_3d_var_{steps}steps.npy'
    print(f"Loading variable length 3D data from {path_3d}...")
    
    data_3d = np.load(path_3d, allow_pickle=True).item()
    X = data_3d['X']  # Array of variable-length numpy arrays
    y = data_3d['y']
    
    seed = 42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # Use shared dataset and collate function
    train_dataset = VariableLengthEventDataset(X_train, y_train, n_features=7)
    test_dataset = VariableLengthEventDataset(X_test, y_test, n_features=7)
    
    collate_fn = create_collate_fn(n_features=7)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MaskedPointCloudNet(in_features=7).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"Training GNN/PointCloud model on {device}...")
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y, batch_mask in train_loader:
            batch_X, batch_y, batch_mask = batch_X.to(device), batch_y.to(device), batch_mask.to(device)
            optimizer.zero_grad()
            out = model(batch_X, batch_mask)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Inference / Validation
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_X, batch_y, batch_mask in test_loader:
                batch_X, batch_mask = batch_X.to(device), batch_mask.to(device)
                out = model(batch_X, batch_mask)
                preds = torch.sigmoid(out).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(batch_y.numpy())
                
        auc = roc_auc_score(all_targets, all_preds)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f} - Test AUC: {auc:.4f}")

    print("\n✓ Completed GNN training!")


if args.gnn_ablation:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from sklearn.metrics import roc_auc_score
    import glob

    feature_names = ['E', 'betax', 'betay', 'betaz']
    num_features = len(feature_names)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Ablation Study (Energy & Betas) on {device}...")

    # Load data
    STEPS = args.steps
    data_path = f'/work/submit/haoyun22/FCC-Beam-Background/DM/results_nosquash_cosine_chargeloss/classifier_data_3d_beta_var_{STEPS}steps.npy'
    print(f"Loading data from {data_path}")
    data_3d = np.load(data_path, allow_pickle=True).item()
    X_3d_all = data_3d['X']
    y_3d_all = data_3d['y']
    
    # Dataset & Collate 
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_3d_all, y_3d_all, test_size=0.2, random_state=42)
    
    # Use shared collate function with 4 features
    collate_fn_ablation = create_collate_fn(n_features=4)

    train_loader = DataLoader(VariableLengthEventDataset(X_train, y_train, n_features=4), batch_size=64, shuffle=True, collate_fn=collate_fn_ablation)
    test_loader = DataLoader(VariableLengthEventDataset(X_test, y_test, n_features=4), batch_size=64, shuffle=False, collate_fn=collate_fn_ablation)

    def run_training_experiment(ablate_feat_idx=None, epochs=15):
        model = MaskedPointCloudNet(in_features=4, hidden_dim=64).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        
        test_aucs = []
        
        for epoch in range(epochs):
            model.train()
            for X_b, y_b, m_b in train_loader:
                X_b, y_b, m_b = X_b.to(device), y_b.to(device), m_b.to(device)
                
                if ablate_feat_idx is not None:
                    # Ablate the specified feature by zeroing it out in the input
                    X_b[:, :, ablate_feat_idx] = 0.0
                    
                optimizer.zero_grad()
                out = model(X_b, m_b)
                loss = criterion(out, y_b)
                loss.backward()
                optimizer.step()
                
            # Eval
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for X_b, y_b, m_b in test_loader:
                    X_b, y_b, m_b = X_b.to(device), y_b.to(device), m_b.to(device)
                    if ablate_feat_idx is not None:
                        X_b[:, :, ablate_feat_idx] = 0.0
                        
                    out = model(X_b, m_b)
                    probs = torch.sigmoid(out).cpu().numpy()
                    all_preds.extend(probs.flatten())
                    all_labels.extend(y_b.cpu().numpy().flatten())
                    
            try:
                auc = roc_auc_score(all_labels, all_preds)
            except ValueError:
                auc = 0.5
            test_aucs.append(auc)
            
        # Mean AUC over last 3 epochs to smooth out fluctuations
        avg_last_3 = np.mean(test_aucs[-3:])
        return avg_last_3

    # 1. Run Baseline
    print("\n--- Running Baseline Model ---")
    baseline_auc = run_training_experiment(ablate_feat_idx=None, epochs=15)
    print(f"Baseline AUC (avg last 3 epochs): {baseline_auc:.4f}")
    
    # 2. Run Ablation for each feature
    results = []
    for idx, feature in enumerate(feature_names):
        print(f"\n--- Ablating Feature: {feature} (idx {idx}) ---")
        ablated_auc = run_training_experiment(ablate_feat_idx=idx, epochs=15)
        drop = baseline_auc - ablated_auc
        results.append((feature, ablated_auc, drop))
        print(f"Ablated {feature:5s} -> AUC: {ablated_auc:.4f} | Drop: {drop:.4f}")
        
    # 3. Print Summary
    results.sort(key=lambda x: x[2], reverse=True)
    print(f"\n{'='*60}")
    print("FEATURE ABLATION RESULTS (Energy & Betas Only) (Sorted by Impact)")
    print(f"{'='*60}")
    print(f"Baseline AUC: {baseline_auc:.4f}")
    for feature, ab_auc, drop in results:
        print(f"Feature: {feature:10s} | AUC: {ab_auc:.4f} | Drop: {drop:.4f}")


# Extract 3D numpy array for GNN with energy and betas only
if args.data_3d_beta:
    steps = args.steps
    REAL_DATA_PATH = "/work/submit/haoyun22/FCC-Beam-Background/DM/guineapig_raw_trimmed.npy"
    GEN_DATA_PATH = f"/work/submit/haoyun22/FCC-Beam-Background/DM/results_nosquash_cosine_chargeloss/generated_events_{steps}steps.npy"
    
    print("Loading real and generated data for 3D energy-betas preparation...")
    real_events = load_events(REAL_DATA_PATH)
    gen_events = load_events(GEN_DATA_PATH)
    
    print("\nPreparing variable length 3D sequential data [events, hits, features] (Energy & Betas only)...")
    real_data_3d = prepare_data_per_event_energy_betas(real_events, pdgs=None, max_hits=None)
    gen_data_3d = prepare_data_per_event_energy_betas(gen_events, pdgs=None, max_hits=None)
    
    # Labels: 0 for real, 1 for generated
    real_labels = np.zeros(len(real_data_3d), dtype=np.int32)
    gen_labels = np.ones(len(gen_data_3d), dtype=np.int32)
    
    n_real = len(real_data_3d)
    n_gen = len(gen_data_3d)
    
    print(f"\nClass imbalance before balancing:")
    print(f"  Real (0): {n_real}, Generated (1): {n_gen}")
    
    # Balance classes: downsample the larger class to match the smaller one
    min_samples = min(n_real, n_gen)
    
    if n_real > n_gen:
        # Downsample real data
        real_indices = np.random.choice(n_real, min_samples, replace=False)
        real_data_3d = [real_data_3d[i] for i in real_indices]
        real_labels = real_labels[real_indices]
        print(f"  Downsampled Real (0): {n_real} -> {min_samples}")
    elif n_gen > n_real:
        # Downsample generated data
        gen_indices = np.random.choice(n_gen, min_samples, replace=False)
        gen_data_3d = [gen_data_3d[i] for i in gen_indices]
        gen_labels = gen_labels[gen_indices]
        print(f"  Downsampled Generated (1): {n_gen} -> {min_samples}")
    
    X_3d = np.array(real_data_3d + gen_data_3d, dtype=object)
    y_3d = np.concatenate([real_labels, gen_labels])
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X_3d))
    X_3d = X_3d[shuffle_idx]
    y_3d = y_3d[shuffle_idx]
    
    output_path_3d = f'/work/submit/haoyun22/FCC-Beam-Background/DM/results_nosquash_cosine_chargeloss/classifier_data_3d_beta_var_{steps}steps.npy'
    
    data_dict_3d = {
        'X': X_3d,
        'y': y_3d,
        'feature_names': ['E', 'betax', 'betay', 'betaz'],
        'n_real': np.sum(y_3d==0),
        'n_generated': np.sum(y_3d==1),
        'description': 'Variable length 3D Classifier data [events, variable_hits, features=4 (E, betas)]: 0=real, 1=generated. Balanced 1:1 class ratio.'
    }
    
    np.save(output_path_3d, data_dict_3d)
    print(f"\n✓ Saved variable length 3D classifier data (energy & betas) to: {output_path_3d}")
    print(f"  Total events: {len(X_3d)}")
    print(f"  Real (0): {np.sum(y_3d==0)}, Generated (1): {np.sum(y_3d==1)}")
    print(f"  Class balance: 1:1 ✓")


# Train and evaluate GNN with energy and betas only
if args.run_gnn_beta:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    
    steps = args.steps
    path_3d = f'/work/submit/haoyun22/FCC-Beam-Background/DM/results_nosquash_cosine_chargeloss/classifier_data_3d_beta_var_{steps}steps.npy'
    print(f"Loading variable length 3D energy-betas data from {path_3d}...")
    
    data_3d = np.load(path_3d, allow_pickle=True).item()
    X = data_3d['X']
    y = data_3d['y']
    
    seed = 42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # Use shared dataset and collate function with 4 features for energy+betas
    train_dataset = VariableLengthEventDataset(X_train, y_train, n_features=4)
    test_dataset = VariableLengthEventDataset(X_test, y_test, n_features=4)
    
    collate_fn = create_collate_fn(n_features=4)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MaskedPointCloudNet(in_features=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"Training GNN/PointCloud model (Energy & Betas) on {device}...")
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y, batch_mask in train_loader:
            batch_X, batch_y, batch_mask = batch_X.to(device), batch_y.to(device), batch_mask.to(device)
            optimizer.zero_grad()
            out = model(batch_X, batch_mask)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_X, batch_y, batch_mask in test_loader:
                batch_X, batch_mask = batch_X.to(device), batch_mask.to(device)
                out = model(batch_X, batch_mask)
                preds = torch.sigmoid(out).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(batch_y.numpy())
                
        auc = roc_auc_score(all_targets, all_preds)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f} - Test AUC: {auc:.4f}")

    print("\n✓ Completed GNN training (Energy & Betas)!")