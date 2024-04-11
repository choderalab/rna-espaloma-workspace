#!/bin/bash
#BSUB -P "repx"
#BSUB -J "cccc"
#BSUB -n 1
#BSUB -R rusage[mem=8]
#BSUB -R span[hosts=1]
#BSUB -q gpuqueue
#BSUB -sp 1 # low priority. default is 12, max is 25
#BSUB -gpu num=1:j_exclusive=yes:mode=shared
#BSUB -W 35:00
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
#date=$(date '+%Y-%m-%d')
#cp enhanced.nc enhanced.nc.${date}
#cp enhanced_checkpoint.nc enhanced_checkpoint.nc.${date}

# define
INPUT_NCFILE="enhanced.nc"
N_ITERATIONS=10000   # 10 ns/replica

# run
conda activate openmm
script_path="/home/takabak/data/exploring-rna/rna-espaloma/experiment/tetramer/script"
python ${script_path}/repx_extend.py --input_ncfile ${INPUT_NCFILE} --n_iterations ${N_ITERATIONS}