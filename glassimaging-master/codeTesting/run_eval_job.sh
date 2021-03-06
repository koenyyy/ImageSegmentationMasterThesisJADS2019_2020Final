#!/bin/bash
#SBATCH --ntasks=6
#SBATCH --mem=14G
#SBATCH --gres=gpu:1
#SBATCH -p GPU_short
#SBATCH -t 08:00:00
#SBATCH -o /media/data/kderaad/glassimaging/out_%j.log
#SBATCH -e /media/data/kderaad/glassimaging/error_%j.log

# This is the temporary dir for your job on the SSD
# It will be deleted once your job finishes so don't forget to copy your files!
MY_TMP_DIR=/slurmtmp/${SLURM_JOB_USER}.${SLURM_JOB_ID}


# Load the modules
module purge
module load python/3.6.7
module load tensorflow/1.12.0

python -m glassimaging.execution.jobs.jobeval eval_nifti_2 config/eval_unet.json experiment_results/20200529161504_BTD_zscore_nohOtsu_noBC_Res1