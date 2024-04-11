#!/bin/bash
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --time=48:00:00
#SBATCH --output=ai-%j.out
for i in $(seq 1 5000); do ./perm_SP_IFP ${i}.pam; done

