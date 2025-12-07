#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=bert_mnli_zs
#SBATCH --mem=16GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/bert_mnli_zs_%j.out
#SBATCH --error=logs/bert_mnli_zs_%j.err

echo "=========================================="
echo " BERT-MNLI ZERO-SHOT (NEU GPU) "
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Submit dir: $SLURM_SUBMIT_DIR"
echo "=========================================="
echo ""

mkdir -p logs

# -------------------------------------------
# Load modules (Explorer)
# -------------------------------------------
module load anaconda3/2024.06
module load cuda/12.1.1

# -------------------------------------------
# Enable conda and activate env
# -------------------------------------------
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate cs4120-bert

echo "Environment activated."
echo "Python executable: $(which python)"
python -c "import sys; print('sys.executable:', sys.executable)"
echo ""

# -------------------------------------------
# Quick GPU sanity check
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
# Run zero-shot script
# -------------------------------------------
echo "=========================================="
echo " RUNNING BERT-MNLI ZERO-SHOT "
echo "=========================================="

START_TIME=$(date +%s)

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"
echo ""

# OPTION 1: bert_mnli_zero_shot.py is in a package directory, e.g. mnli_tuned_model/
# Make sure the directory is named with underscores, NOT dashes.
python -m mnli_tuned_model.bert_mnli_zero_shot

# OPTION 2: bert_mnli_zero_shot.py is in the current directory
# Comment out the line above and use:
# python bert_mnli_zero_shot.py

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo " JOB COMPLETED "
echo "=========================================="
echo "Total time: $ELAPSED seconds ($((ELAPSED/60)) minutes)"
echo "Finished at: $(date)"
echo "=========================================="
