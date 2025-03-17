#!/bin/bash -l 
#SBATCH -J lzqhzo_gpt
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=15-00:00:00
#SBATCH --mail-user=zqliang@ucdavis.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -o output/bench-%j.output
#SBATCH -e output/bench-%j.output
#SBATCH --partition=gpu-homayoun
hostname
srun python -u index.py