#!/bin/bash
#BSUB -P "repx"
#BSUB -J "caau"
#BSUB -n 1
#BSUB -R rusage[mem=8]
#BSUB -R span[hosts=1]
#BSUB -q gpuqueue
#BSUB -sp 1 # low priority. default is 12, max is 25
#BSUB -gpu num=1:j_exclusive=yes:mode=shared
#BSUB -W 35:00
#####BSUB -m "lu-gpu ld-gpu lx-gpu ly-gpu lj-gpu ll-gpu ln-gpu lv-gpu"
#####BSUB -m "lu-gpu ld-gpu ly-gpu lj-gpu ll-gpu lv-gpu"
#####BSUB -R A100 || A40
#BSUB -o out_%J_%I.stdout
#BSUB -eo out_%J_%I.stderr
#BSUB -L /bin/bash

source ~/.bashrc
OPENMM_CPU_THREADS=1
#export OE_LICENSE=~/.openeye/oe_license.txt   # Open eye license activation/env


# change dir
echo "changing directory to ${LS_SUBCWD}"
cd ${LS_SUBCWD}


# Report node in use
echo "======================"
hostname
env | sort | grep 'CUDA'
nvidia-smi
echo "======================"


# backup
date=$(date '+%Y-%m-%d')
cp enhanced.nc enhanced.nc.${date}
cp enhanced_checkpoint.nc enhanced_checkpoint.nc.${date}

# define
INPUT_NCFILE="enhanced.nc"
N_ITERATIONS=10000   # 10 ns/replica

# run
conda activate openmm
python repx_extend.py --input_ncfile ${INPUT_NCFILE} --n_iterations ${N_ITERATIONS}