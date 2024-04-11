#!/bin/bash
#SBATCH --partition=serc
#SBATCH --time=5:00:00
#SBATCH --mem=32G

module load matlab
matlab out.out < generate.m
