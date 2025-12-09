#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=bert_cascade_plm
#SBATCH --mem=16GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/bert_cascade_plm_%j.out
#SBATCH --error=logs/bert_cascade_plm_%j.err

echo "=========================================="
echo " BERT-BASE CASCADED BINARY (HC vs Non-HC, MCI vs Dementia) "
echo " Module: plm_cascade (MODEL_NAME=bert-base-uncased) "
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
# Load modules (Explorer)
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

# -------------------------------------------
# Run PLM cascaded script with BERT-base
# -------------------------------------------
echo "=========================================="
echo " RUNNING plm_cascade.py (PLM_MODEL_NAME=bert-base-uncased) "
echo "=========================================="

START_TIME=$(date +%s)

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"
echo ""

# Set the model to BERT-base-uncased for plm_cascade
export PLM_MODEL_NAME=bert-base-uncased

# If plm_cascade.py is in the project root:
python -m pretrained_lm.plm_cascade


END_TIME=$(date +%s )
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo " JOB COMPLETED "
echo "=========================================="
echo "Total time: $ELAPSED seconds ($((ELAPSED/60)) minutes)"
echo "Finished at: $(date)"
echo "=========================================="
