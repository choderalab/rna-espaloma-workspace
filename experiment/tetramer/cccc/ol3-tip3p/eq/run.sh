#!/bin/bash
#BSUB -P "benchmark"
#BSUB -J "c4"
#BSUB -n 1
#BSUB -R rusage[mem=8]
#BSUB -R span[hosts=1]
#BSUB -q gpuqueue
#BSUB -sp 1 # low priority. default is 12, max is 25
#BSUB -gpu num=1:j_exclusive=yes:mode=shared
#BSUB -W 3:00
##BSUB -m "lu-gpu ld-gpu lx-gpu ly-gpu lj-gpu ll-gpu ln-gpu lv-gpu"
#BSUB -m "ld-gpu lj-gpu ll-gpu ln-gpu lv-gpu"

## Asking for A100s
##BSUB -R A100 || A40
#BSUB -o out_%J_%I.stdout
#BSUB -eo out_%J_%I.stderr
#BSUB -L /bin/bash

source ~/.bashrc
OPENMM_CPU_THREADS=1
#export OE_LICENSE=~/.openeye/oe_license.txt   # Open eye license activation/env


# chnage dir
echo "changing directory to ${LS_SUBCWD}"
cd $LS_SUBCWD


# Report node in use
echo "======================"
hostname
env | sort | grep 'CUDA'
nvidia-smi
echo "======================"


# run job
conda activate espaloma
python openmm_eq.py -i ../../crd/rna_noh.pdb --output_prefix .
