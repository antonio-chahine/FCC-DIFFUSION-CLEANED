import glob
import numpy as np
from tqdm import tqdm

FILE_PATTERNS = [
    "/ceph/submit/data/group/fcc/ee/beam_backgrounds/guineapig/FCCee_Z_GHC_V25p3_4_FCCee_Z256_2T_grids8/*.pairs",
    "/ceph/submit/data/user/b/bmaier/BiB/*.pairs",
]

matched_counts = {}
FILES = []
for pattern in FILE_PATTERNS:
    matched = glob.glob(pattern)
    matched_counts[pattern] = len(matched)
    FILES.extend(matched)

FILES = [f for f in FILES if "/output_" in f and "/output0_" not in f]
FILES = sorted(FILES)

print("Matched .pairs files by source:")
for pattern in FILE_PATTERNS:
    print(f"  {pattern}: {matched_counts[pattern]}")
print(f"Total files after output filter (output0 excluded): {len(FILES)}")

OUTFILE = "/ceph/submit/data/user/h/haoyun22/diffusion_data/guineapig_raw_trimmed_new_2.npy"

all_events = []
skipped = []

for f in tqdm(FILES, desc="Reading .pairs", unit="event"):
    try:
        d = np.loadtxt(f, dtype=np.float32)

        # np.loadtxt returns 1D for single-line files
        if d.ndim == 1:
            if d.size == 0:
                skipped.append(f)
                continue
            d = d.reshape(1, -1)

        # skip malformed events with too few columns
        if d.ndim != 2 or d.shape[1] < 7:
            skipped.append(f)
            continue

        # keep only raw kinematics + vertex
        d = d[:, :7]

        all_events.append(d)

    except PermissionError:
        skipped.append(f)
        continue

    except OSError as e:
        # catches other filesystem weirdness
        skipped.append(f)
        continue

    except ValueError:
        # catches parse errors / irregular text formats
        skipped.append(f)
        continue

events = np.array(all_events, dtype=object)
np.save(OUTFILE, events, allow_pickle=True)

print(f"Saved {len(events)} events")
print(f"Skipped {len(skipped)} files")
if len(events) > 0:
    print("Example event shape:", events[0].shape)
else:
    print("No valid events were loaded.")