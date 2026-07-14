#!/bin/bash
#
# ====== Slurm Configuration ======
#SBATCH --job-name=dm_pipeline                # Job name
#SBATCH --partition=submit-gpu                # GPU partition
#SBATCH --constraint=nvidia_a30
#SBATCH --gres=gpu:1                          # Request 1 GPU
#SBATCH --cpus-per-gpu=8                      # CPU cores per GPU
#SBATCH --mem=32G                             # Memory
#SBATCH --time=48:00:00                       # Max runtime
#SBATCH --output=logs/%x-%j.out               # Stdout log
#SBATCH --error=logs/%x-%j.err                # Stderr log

# ====== Environment Setup ======

mkdir -p logs

cd /work/submit/haoyun22/FCC-Beam-Background

# Clean environment variables that may interfere with virtual env
unset PYTHONPATH
unset PYTHONHOME

# Activate virtual environment
source /work/submit/haoyun22/FCC-Beam-Background/FCC310/bin/activate

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "Running on host: $(hostname)"

# Check CUDA availability
python - << 'EOF'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU device:", torch.cuda.get_device_name(0))
    print("Number of GPUs:", torch.cuda.device_count())
EOF

# ====== Script Path and Parameters ======

SCRIPT_DIR="/work/submit/haoyun22/FCC-Beam-Background/DM"
SCRIPT="$SCRIPT_DIR/pipeline_cosine_charge.py"
DATA_PATH="$SCRIPT_DIR/guineapig_raw_trimmed_new.npy"
OUTDIR="$SCRIPT_DIR/results_nosquash_cosine_chargeloss"

echo ""
echo "=========================================="
echo "Diffusion Model Pipeline"
echo "=========================================="
echo "GPU Device: $CUDA_VISIBLE_DEVICES"
echo "Script: $SCRIPT"
echo "Data Path: $DATA_PATH"
echo "Output Directory: $OUTDIR"
echo "=========================================="
echo ""

# ====== Data Path Check ======

if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Error: Data file not found!"
    echo "   Path: $DATA_PATH"
    echo ""
    echo "Data not prepared, run DM/track_data.py make data"
    echo ""
    exit 1
fi

echo "✓ Found data file: $DATA_PATH"
echo ""

# ====== Training ======

echo "=========================================="
echo "[1/2] Running DM Training..."
echo "=========================================="
echo ""

mkdir -p "$OUTDIR"

python "$SCRIPT" train \
    --data_path "$DATA_PATH" \
    --outdir "$OUTDIR" \
    --epochs 50 \
    --batch_size 2 \
    --T 1000

if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi

echo ""
echo "✓ Training completed"
echo ""

# ====== Sampling ======

echo "=========================================="
echo "[2/2] Running DM Sampling..."
echo "=========================================="
echo ""

# Define sampling steps to generate
STEPS=(50 100 200 500 1000)

for STEP in "${STEPS[@]}"; do
    echo ""
    echo "Generating synthetic events with $STEP denoising steps..."
    
    OUTPUT_FILE="$OUTDIR/generated_events_${STEP}steps.npy"
    
    if [ -f "$OUTPUT_FILE" ]; then
        echo "✓ Output already exists, skipping: $OUTPUT_FILE"
        continue
    fi
    
    python "$SCRIPT" sample \
        --outdir "$OUTDIR" \
        --n_events 500 \
        --sample_batch_size 16 \
        --num_steps "$STEP"
    
    if [ $? -ne 0 ]; then
        echo "❌ Sampling with $STEP steps failed!"
        exit 1
    fi
    
    echo "✓ Sampling with $STEP steps completed"
done

echo ""
echo "=========================================="
echo "✓ Pipeline execution completed!"
echo "=========================================="
echo ""
echo "Generated file locations:"
echo "  - Generated events: $OUTDIR/generated_events_*steps.npy"
echo "  - Model checkpoint: $OUTDIR/ckpt_last.pt"
echo "  - Training losses: $OUTDIR/train_losses.npy"
echo "  - Validation losses: $OUTDIR/val_losses.npy"
echo "=========================================="
