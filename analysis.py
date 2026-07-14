'''
This script is used for analyzing and comparing the distributions of physical quantities between real and generated events in a particle physics context. It includes functions for loading event data, sanitizing it into a consistent format, extracting relevant physical quantities for specified particle species, and comparing the distributions of these quantities across different sampling steps with fractional difference plots.
'''


import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy.stats import entropy, wasserstein_distance

print("Libraries imported successfully!")

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

def compare_distributions_across_steps(real_data, gen_data_all, physics, tag='all', 
                                       bins=50, range_quantile=0.95, 
                                       figsize=(12, 10), save=False):
    """
    Compare a specific physics variable across all sampling steps with fractional difference.
    
    Args:
        real_data: Dictionary containing real data for different particle species
        gen_data_all: Dictionary containing generated data for different steps
        physics: Physics variable to compare (e.g., 'E', 'px', 'py', 'pz', 'p', 'pt', 'betax', etc.)
        tag: Particle species tag ('all', 'eminus', 'eplus')
        bins: Number of bins for histogram (default: 50)
        range_quantile: Quantile for determining the range (default: 0.95)
        figsize: Figure size tuple (default: (12, 10))
        outdir: Output directory for saving plots (default: None)
        save: Whether to save the plot as PDF (default: False)
    """
    # Tag mapping for display names
    tag_map = {
        'all': 'All particles',
        'eminus': r'$e^-$',
        'eplus': r'$e^+$'
    }
    # Set global plot style for consistency
    plt.rcParams.update({
        'font.size': 14,           # Base font size
        'axes.labelsize': 20,      # Axis label font size
        'axes.titlesize': 22,      # Title font size
        'xtick.labelsize': 16,     # X-axis tick label size
        'ytick.labelsize': 16,     # Y-axis tick label size
        'legend.fontsize': 18,     # Legend font size
        'axes.linewidth': 1.5,     # Axis line width
        'grid.linewidth': 0.8,     # Grid line width
        'lines.linewidth': 2.0,    # Line width
        'xtick.major.width': 1.2,  # X tick width
        'ytick.major.width': 1.2,  # Y tick width
        'xtick.major.pad': 7,      # Padding between ticks and labels
        'ytick.major.pad': 7,      # Padding between ticks and labels
    })

    
    # Title mapping for physics variables
    title_map = {
        'mult': 'Multiplicity',
        'px': '$p_x$ [GeV/C]',
        'py': '$p_y$ [GeV/C]',
        'pz': '$p_z$ [GeV/C]',
        'p': '|p| [GeV/C]',
        'pt': '$p_T$ [GeV/C]',
        'E': '|E| [GeV]',
        'E_abs': '|E| [GeV]',
        'E_signed': '$E_{signed}$ [GeV]',
        'beta_mag': '|β|',
        'betax': '$\\beta_x$',
        'betay': '$\\beta_y$',
        'betaz': '$\\beta_z$',
        'x': 'x [nm]',
        'y': 'y [nm]',
        'z': 'z [nm]'
    }
    
    # Define colors for different steps
    colors = {
        "Simulated": "#bdbdbd",   # 灰填充
        "Step 50":  "#0000ff",    # 蓝
        "Step 100":  "#008000",    # 绿
        "Step 200": "#800080",    # 紫
        "Step 250": "#ff0000",    # 红
        "Step 500": "#00bcd4",    # 青
        "Step 1000": "#ff9800",   # 橙
    }
    
    # Extract real data
    real_sp = real_data[tag]
    real_vals = real_sp[physics]
    
    # Compute range from real data
    range_lim = (0, np.quantile(real_vals, range_quantile)) if np.min(real_vals) >= 0 else \
                (np.quantile(real_vals, 1-range_quantile), np.quantile(real_vals, range_quantile))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'{title_map[physics]} Distribution {tag_map[tag]}', fontsize=22, fontweight='bold')
    
    # Compute real histogram once
    real_hist, bin_edges = np.histogram(real_vals, bins=bins, range=range_lim, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Upper subplot: Distribution comparison
    ax1.hist(real_vals, bins=bin_edges, density=True, 
            alpha=0.6, label='Simulated', color=colors["Simulated"], zorder=10)
    
    # Plot each step and compute fractional differences
    for step in sorted(gen_data_all.keys()):
        gen_sp = gen_data_all[step][tag]
        gen_vals = gen_sp[physics]
        
        color = colors.get(f"Step {step}", 'gray')
        
        # Plot distribution
        ax1.hist(gen_vals, bins=bin_edges, density=True, 
                histtype='step', linewidth=2.5, label=f'Step {step}', 
                color=color, alpha=1, zorder=15)
        
        # Compute fractional difference for lower subplot
        gen_hist, _ = np.histogram(gen_vals, bins=bin_edges, density=True)
        frac_diff = np.where(real_hist > 0, (gen_hist - real_hist) / real_hist, 0)
        
        # Plot fractional difference
        ax2.plot(bin_centers, frac_diff, linewidth=2, 
                 color=color, alpha=0.8)
    
    ax1.set_ylabel('Density')
    ax1.set_yscale('log')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(bin_edges[0], bin_edges[-1])
    
    # Lower subplot: Fractional difference
    ax2.axhline(0, color='black', linestyle='-', linewidth=1.5)
    ax2.axhline(0.1, color='grey', linestyle='--', linewidth=1, alpha=0.8, label='±10%')
    ax2.axhline(-0.1, color='grey', linestyle='--', linewidth=1, alpha=0.8)
    
    ax2.set_xlabel(f'{title_map[physics]}')
    ax2.set_ylabel('Frac Diff')
    ax2.legend(loc='best', ncol=2, fontsize = 14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1, 1)
    ax2.set_xlim(bin_edges[0], bin_edges[-1])
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f'/work/submit/haoyun22/FCC-Beam-Background/output_figures/cosine_charge_step_comp/distribution_comparison_{physics}_{tag}_all_steps.pdf', dpi=300)
    
    plt.show()
    
    print(f"Compared {physics} distributions across steps: {list(gen_data_all.keys())}")
    print(f'Saved plot: distribution_comparison_{physics}_{tag}_all_steps.pdf' if save else "Plot not saved.")



print("Helper functions defined!")

REAL_DATA_PATH = "/work/submit/haoyun22/FCC-Beam-Background/DM/guineapig_raw_trimmed.npy"
GEN_DATA_DIR = "/work/submit/haoyun22/FCC-Beam-Background/DM/results_nosquash_cosine_chargeloss"
# Define steps to load
STEPS = [50, 100, 200, 250, 500, 1000]

print(f"Real data path: {REAL_DATA_PATH}")
print(f"Generated data directory: {GEN_DATA_DIR}")
print(f"Steps to load: {STEPS}")

print("Loading real data...")
real_events = load_events(REAL_DATA_PATH)
n_real = len(real_events)
print(f"✓ Loaded {n_real} real events")

print(f"\nLoading generated data for multiple steps...")
gen_events_all = {}

for step in STEPS:
    gen_path = f"{GEN_DATA_DIR}/generated_events_{step}steps.npy"
    try:
        gen_events_all[step] = load_events(gen_path)
        n_gen = len(gen_events_all[step])
        print(f"  ✓ Step {step:3d}: {n_gen} events loaded")
    except FileNotFoundError:
        print(f"  ✗ Step {step:3d}: File not found - {gen_path}")
    except Exception as e:
        print(f"  ✗ Step {step:3d}: Error - {e}")

print(f"\n✓ Data loading complete!")
print(f"  Real events: {n_real}")
print(f"  Generated data loaded for steps: {list(gen_events_all.keys())}")


species_list = [
    {"name": "e−",  "pdgs": [11],   "tag": "eminus"},
    {"name": "e+",  "pdgs": [-11],  "tag": "eplus"},
    {"name": "all", "pdgs": None,   "tag": "all"},
]

# Extract real data (only once)
print("="*60)
print("Extracting real data...")
print("="*60)
real_data = {}

for sp in species_list:
    print(f"\nExtracting {sp['name']}...")
    real_data[sp['tag']] = extract_species(real_events, sp['pdgs'])
    print(f"  Real: {len(real_data[sp['tag']]['E'])} particles")

print("\n" + "="*60)
print("Extracting generated data for each step...")
print("="*60)
gen_data_all = {}

for step in gen_events_all.keys():
    print(f"\n--- Step {step} ---")
    gen_data_all[step] = {}
    
    for sp in species_list:
        print(f"  Extracting {sp['name']}...")
        gen_data_all[step][sp['tag']] = extract_species(gen_events_all[step], sp['pdgs'])
        print(f"    Gen: {len(gen_data_all[step][sp['tag']]['E'])} particles")

print("\n" + "="*60)
print("✓ All species extracted for all steps!")
print("="*60)
print(f"Real data tags: {list(real_data.keys())}")
print(f"Generated data steps: {list(gen_data_all.keys())}")


# Example: Compare distributions across all steps for all physics variables

# Define physics variables and particle types
physics_list = ['px', 'py', 'pz', 'p', 'pt', 'E', 'beta_mag', 'betax', 'betay', 'betaz', 'x', 'y', 'z']
tag_list = ['all', 'eminus', 'eplus']  # Particle species to compare

# Select which particle type to compare
tag = 'eplus'  # Change to 'eminus' or 'eplus' to compare specific species

print(f"Generating comparison plots for {tag}...")
print(f"Total plots to generate: {len(physics_list)}")
print("="*80)

# Generate comparison plots for all physics variables
for tag in tag_list:
    print(f'Generating comparison plots for {tag}...')
    print(f'Total physics variables: {len(physics_list)}')
    print("="*80)

    for idx, physics in enumerate(physics_list, 1):
        print(f"\n[{idx}/{len(physics_list)}] Processing {physics}...")
        
        compare_distributions_across_steps(
            real_data=real_data,
            gen_data_all=gen_data_all,
            physics=physics,
            tag=tag,
            bins=50,
            range_quantile=0.99,
            figsize=(12, 10),
            save=True  # Set to True to save all plots
        )
        
        print(f"✓ Completed {physics}")

print("\n" + "="*80)
print(f"✓ All {len(physics_list)} comparison plots generated successfully!")
print("="*80)