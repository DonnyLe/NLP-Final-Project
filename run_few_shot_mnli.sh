#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=fewshot_mnli
#SBATCH --mem=16GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/fewshot_mnli_%j.out
#SBATCH --error=logs/fewshot_mnli_%j.err

echo "=========================================="
echo " FEW-SHOT MNLI-INITIALIZED BERT "
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Submit dir: $SLURM_SUBMIT_DIR"
echo "=========================================="
echo ""

# Make sure logs directory exists
mkdir -p logs

# -------------------------------------------
# Load modules (Explorer)
# -------------------------------------------
module load anaconda3/2024.06
module load cuda/12.1.1

# -------------------------------------------
# Enable conda and activate env
# -------------------------------------------
# This line is crucial in batch jobs
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate cs4120-bert

echo "Environment activated."
echo "Python executable: $(which python)"
python -c "import sys; print('sys.executable:', sys.executable)"
echo ""

# -------------------------------------------
# GPU sanity check
# -------------------------------------------
echo "=========================================="
echo " GPU CHECK "
echo "=========================================="

python << 'EOF'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("Total GPU Memory: {:.2f} GB".format(props.total_memory / 1e9))
print()
EOF

# -------------------------------------------
# Run few-shot MNLI script
# -------------------------------------------
echo "=========================================="
echo " RUNNING FEW-SHOT MNLI BERT "
echo "=========================================="

START_TIME=$(date +%s)

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"
echo ""

# If bert_few_shot_mnli.py is in the current directory:
python -m mnli_tuned_model.bert_few_shot_mnli

# If instead it's inside a package, e.g. nlp_experiments/bert_few_shot_mnli.py,
# comment the line above and use:
# python -m nlp_experiments.bert_few_shot_mnli

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo " JOB COMPLETED "
echo "=========================================="
echo "Total time: $ELAPSED seconds ($((ELAPSED/60)) minutes)"
echo "Finished at: $(date)"
echo "=========================================="
