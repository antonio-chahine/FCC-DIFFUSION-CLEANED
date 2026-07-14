#!/bin/bash
#
# ====== Slurm Configuration ======
#SBATCH --job-name=gnn_beta_pipeline         # Job name
#SBATCH --partition=submit-gpu               # GPU partition
#SBATCH --constraint=nvidia_a30
#SBATCH --gres=gpu:1                         # Request 1 GPU
#SBATCH --cpus-per-gpu=8                     # CPU cores per GPU
#SBATCH --mem=32G                            # Memory
#SBATCH --time=48:00:00                      # Max runtime
#SBATCH --output=logs/%x-%j.out              # Stdout log
#SBATCH --error=logs/%x-%j.err               # Stderr log

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
SCRIPT="$SCRIPT_DIR/classifier.py"
STEPS=(50 100 200 500 1000)

echo ""
echo "=========================================="
echo "GNN Energy and Beta Features Pipeline"
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
    echo "Processing STEP = $STEP"
    echo "=========================================="
    
    # Check if generated data exists
    GEN_DATA_PATH="$SCRIPT_DIR/results_nosquash_cosine_chargeloss/generated_events_${STEP}steps.npy"
    if [ ! -f "$GEN_DATA_PATH" ]; then
        echo "❌ Error: Generated data not found"
        echo "   Path: $GEN_DATA_PATH"
        echo "   Skipping step $STEP"
        echo ""
        continue
    fi
    echo "✓ Found generated data for step $STEP"
    
    # Prepare 3D energy and beta data
    echo ""
    CLASSIFIER_DATA_PATH="$SCRIPT_DIR/results_nosquash_cosine_chargeloss/classifier_data_3d_beta_var_${STEP}steps.npy"
    
    if [ -f "$CLASSIFIER_DATA_PATH" ]; then
        echo "[1/2] 3D energy and beta data already exists, skipping data prep"
        echo "   Path: $CLASSIFIER_DATA_PATH"
    else
        echo "[1/2] Preparing 3D energy and beta data..."
        python "$SCRIPT" --data_3d_beta --steps "$STEP"
        
        if [ $? -ne 0 ]; then
            echo "❌ Step 1 failed: Could not prepare 3D data (steps=$STEP)"
            echo ""
            continue
        fi
        echo "✓ Data preparation completed"
    fi
    
    # GNN Training
    echo ""
    echo "[2/2] Running GNN training..."
    python "$SCRIPT" --run_gnn_beta --steps "$STEP"
    
    if [ $? -ne 0 ]; then
        echo "❌ Step 2 failed: Could not run GNN training (steps=$STEP)"
        echo ""
        continue
    fi
    echo "✓ GNN training completed"
    
    echo "✓ Completed steps=$STEP"
    echo ""
    
done

echo ""
echo "=========================================="
echo "✓ Pipeline execution completed!"
echo "=========================================="
echo ""
echo "Generated file locations:"
echo "  - 3D energy beta data: DM/results_nosquash_cosine_chargeloss/classifier_data_3d_beta_var_*steps.npy"
echo "=========================================="

