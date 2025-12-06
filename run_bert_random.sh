#!/bin/bash
#SBATCH --partition=gpu          # NEU GPU partition
#SBATCH --nodes=1
#SBATCH --gres=gpu:1             # request 1 GPU
#SBATCH --time=08:00:00
#SBATCH --job-name=bert_random_weighted
#SBATCH --mem=16GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/bert_random_weighted_%j.out
#SBATCH --error=logs/bert_random_weighted_%j.err

echo "=========================================="
echo " BERT RANDOM SEARCH (CLASS-WEIGHTED, NEU GPU) "
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Submit dir: $SLURM_SUBMIT_DIR"
echo "=========================================="
echo ""

mkdir -p logs

# -------------------------------
# Modules (Explorer)
# -------------------------------
module load anaconda3/2024.06
module load cuda/12.1.1

# -------------------------------
# Conda env
# -------------------------------
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate cs4120-bert

echo "Environment activated."
echo "Python executable: $(which python)"
python -c "import sys; print('sys.executable:', sys.executable)"
echo ""

# -------------------------------
# Quick GPU sanity check
# -------------------------------
echo "=========================================="
echo " GPU CHECK "
echo "=========================================="

python << 'EOF'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("Total GPU Memory: {:.2f} GB".format(props.total_memory / 1e9))
print()
EOF

# -------------------------------
# Run BERT random search
# -------------------------------
echo "=========================================="
echo " RUNNING RANDOM SEARCH (CLASS-WEIGHTED) "
echo "=========================================="

START_TIME=$(date +%s)

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"
echo ""

# Use the weighted version you just created
python -m bert.bert_random_search_weighted

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo " JOB COMPLETED "
echo "=========================================="
echo "Total time: $ELAPSED seconds ($((ELAPSED/60)) minutes)"
echo "Finished at: $(date)"
echo "=========================================="
