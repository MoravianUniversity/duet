#!/bin/bash
#SBATCH --job-name=duet_opt
#SBATCH --partition=cpu
#SBATCH --account=bebh-delta-cpu

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256g
#SBATCH --time=02-00:00:00
## Not for now: SBATCH --constraint=projects
#SBATCH -o duet_opt-%j.out
#SBATCH -e duet_opt-%j.err

module reset
module load python
source /u/ryum/.venv/bin/activate
echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
python3 duet_opt.py
python3 duet_opt.py --grid grid.jsonc --out results-$SLURM_JOB_ID.csv --steps-per-save 10000