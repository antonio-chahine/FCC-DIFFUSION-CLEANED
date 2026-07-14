#!/bin/bash
#
# ====== Slurm Configuration ======
#SBATCH --job-name=classifier_all_steps      # Job name
#SBATCH --partition=submit-gpu               # GPU partition
#SBATCH --constraint=nvidia_a30
#SBATCH --gres=gpu:1                         # Request 1 GPU
#SBATCH --cpus-per-gpu=8                     # CPU cores per GPU
#SBATCH --mem=24G                            # Memory
#SBATCH --time=24:00:00                      # Max runtime
#SBATCH --output=logs/%x-%j.out              # Stdout log
#SBATCH --error=logs/%x-%j.err               # Stderr log

# ====== Environment Setup ======"

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

# ====== Batch Run Classifier ======

STEPS=(50 100 200 500 1000)
SCRIPT="/work/submit/haoyun22/FCC-Beam-Background/DM/classifier.py"

echo ""
echo "=========================================="
echo "Batch Run Classifier - All Steps (GPU Mode)"
echo "=========================================="

for step in "${STEPS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing STEP = $step"
    echo "=========================================="
    
    # Check if generated data exists
    GEN_DATA_PATH="/work/submit/haoyun22/FCC-Beam-Background/DM/results_nosquash_cosine_chargeloss/generated_events_${step}steps.npy"
    if [ ! -f "$GEN_DATA_PATH" ]; then
        echo "❌ Error: Generated data not found"
        echo "   Path: $GEN_DATA_PATH"
        echo "   Skipping step $step"
        echo ""
        continue
    fi
    echo "✓ Found generated data for step $step"
    
    # 1. Prepare energy and beta data (no position)
    echo ""
    CLASSIFIER_DATA_PATH="/work/submit/haoyun22/FCC-Beam-Background/DM/results_nosquash_cosine_chargeloss/classifier_data_energy_betas_${step}steps.npy"
    
    if [ -f "$CLASSIFIER_DATA_PATH" ]; then
        echo "[1/2] Energy and beta classifier data already exists, skipping data prep"
        echo "   Path: $CLASSIFIER_DATA_PATH"
    else
        echo "[1/2] Preparing energy and beta data..."
        python "$SCRIPT" --data_energy_betas --steps "$step"
        
        if [ $? -ne 0 ]; then
            echo "❌ Step 1 failed: Could not prepare energy and beta data (steps=$step)"
            continue
        fi
        echo "✓ Data preparation completed"
    fi
    
    # 2. Train energy and beta classifier
    echo ""
    echo "[2/2] Training energy and beta feature classifier..."
    python "$SCRIPT" --run_energy_betas --steps "$step"
    
    if [ $? -ne 0 ]; then
        echo "❌ Step 2 failed: Could not train energy and beta classifier (steps=$step)"
        continue
    fi
    echo "✓ Classifier training completed"
    
    echo "✓ Completed steps=$step"
done

echo ""
echo "=========================================="
echo "✓ All steps processed successfully!"
echo "=========================================="
echo ""
echo "Generated file locations:"
echo "  - Energy and beta classifier data: DM/results_nosquash_cosine_chargeloss/classifier_data_energy_betas_*steps.npy"
echo "=========================================="

