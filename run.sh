#!/usr/bin/env bash
#SBATCH --qos=normal
#SBATCH --partition=cpu
#SBATCH -c 8
#SBATCH --account=vector
#SBATCH --output=outputs/quotient-%A_%a.out
#SBATCH --error=errors/quotient-%A_%a.err
#SBATCH --mem=8000M        # memory per node
#SBATCH --time=04:00:00
#SBATCH --job-name=QQG_relaunched
#SBATCH --array=[0-100]

cd
export PATH=/pkgs/anaconda3/bin:$PATH
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate ./condaenvs/pennylane

cd ./quotient_quantum_gates

python FIG2_compare_gates.py --seed_idx=$SLURM_ARRAY_TASK_ID --N=12 --L=12
