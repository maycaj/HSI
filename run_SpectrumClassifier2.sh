#!/bin/bash
#SBATCH --job-name=run_SpectrumClassifier2
#SBATCH --output=/data/maycaj/printAndErrors/my_python_job_%j.out
#SBATCH --error=/data/maycaj/printAndErrors/my_python_job_%j.err
#SBATCH --time=13:00:00  # HH:MM:SS, adjust as needed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # Or more if your script is parallelized
#SBATCH --mem=200G  # Adjust memory request as needed

# Load any necessary modules (e.g., Python, virtual environment)
# module load python/3.x

# If you are using a virtual environment:
# source /Volumes/data/.venv/bin/activate

# Run your Python script
python SpectrumClassifier2.py

# If you used a virtual environme nt, deactivate it:
# deactivate