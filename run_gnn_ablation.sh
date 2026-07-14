#!/bin/bash
#
# ====== Slurm Configuration ======
#SBATCH --job-name=gnn_ablation   # Job name
#SBATCH --partition=submit-gpu                 # GPU partition
#SBATCH --constraint=nvidia_a30
#SBATCH --gres=gpu:1                           # Request 1 GPU
#SBATCH --cpus-per-gpu=8                       # CPU cores per GPU
#SBATCH --mem=32G                              # Memory
#SBATCH --time=5:00:00                        # Max runtime
#SBATCH --output=logs/%x-%j.out                # Stdout log
#SBATCH --error=logs/%x-%j.err                 # Stderr log

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

# ====== GNN Ablation Configuration ======

SCRIPT_DIR="/work/submit/haoyun22/FCC-Beam-Background/DM"
SCRIPT="$SCRIPT_DIR/classifier.py"
STEPS=(500 1000)

echo ""
echo "=========================================="
echo "GNN Feature Ablation Study (Energy & Betas)"
echo "=========================================="
echo "GPU Device: $CUDA_VISIBLE_DEVICES"
echo "Processing Steps: ${STEPS[@]}"
echo "Script Location: $SCRIPT"
echo "=========================================="
echo ""

# Loop over each step
for STEP in "${STEPS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running Ablation Study for STEP = $STEP"
    echo "=========================================="
    
    # Check if 3D beta classifier data exists
    CLASSIFIER_DATA_PATH="$SCRIPT_DIR/results_nosquash_cosine_chargeloss/classifier_data_3d_beta_var_${STEP}steps.npy"
    
    if [ ! -f "$CLASSIFIER_DATA_PATH" ]; then
        echo "❌ Error: 3D beta classifier data not found"
        echo "   Path: $CLASSIFIER_DATA_PATH"
        echo "   Please run --data_3d_beta first to prepare the data"
        echo "   Skipping step $STEP"
        echo ""
        continue
    fi
    echo "✓ Found 3D beta classifier data for step $STEP"
    
    # Run ablation study
    echo ""
    echo "Running GNN feature ablation study..."
    python "$SCRIPT" --gnn_ablation --steps "$STEP"
    
    if [ $? -ne 0 ]; then
        echo "❌ Error: Ablation study failed for step $STEP"
        echo ""
        continue
    fi
    echo "✓ Ablation study completed for step $STEP"
    echo ""
    
done

echo ""
echo "=========================================="
echo "✓ All ablation studies completed!"
echo "=========================================="
echo ""
echo "Generated file locations:"
echo "  - 3D energy beta data (input): DM/results_nosquash_cosine_chargeloss/classifier_data_3d_beta_var_*steps.npy"
echo "  - Ablation results printed to stdout/logs"
echo "=========================================="
