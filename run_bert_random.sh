#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=bert_random
#SBATCH --mem=16GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/bert_random_%j.out
#SBATCH --error=logs/bert_random_%j.err

echo "=========================================="
echo " BERT RANDOM SEARCH (8 HOURS)"
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Submit dir: $SLURM_SUBMIT_DIR"
echo "=========================================="
echo ""

# Create logs directory
mkdir -p logs

# -------------------------------------------
# Load Anaconda module
# -------------------------------------------
module load anaconda3/2024.06
module load cuda/12.1.1

# -------------------------------------------
# Enable conda and activate env
# -------------------------------------------
# This line is CRUCIAL in batch jobs
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh

conda activate cs4120-bert

echo "Environment activated."
echo "Python executable: $(which python)"
python -c "import sys; print('sys.executable:', sys.executable)"
echo ""

# -------------------------------------------
# GPU check
# -------------------------------------------
echo "=========================================="
echo " GPU CHECK "
echo "=========================================="
python << EOF
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("Total GPU Memory: {:.2f} GB".format(props.total_memory / 1e9))
try:
    import accelerate
    print("accelerate:", accelerate.__version__)
except Exception as e:
    print("accelerate import FAILED:", e)
print("")
EOF

# -------------------------------------------
# Run BERT random search
# -------------------------------------------
echo "=========================================="
echo " RUNNING RANDOM SEARCH "
echo "=========================================="

START_TIME=$(date +%s)

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"
echo ""

python -m bert.bert_random_search

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo " JOB COMPLETED "
echo "=========================================="
echo "Total time: $ELAPSED seconds ($((ELAPSED/60)) minutes)"
echo "Finished at: $(date)"
echo "=========================================="
