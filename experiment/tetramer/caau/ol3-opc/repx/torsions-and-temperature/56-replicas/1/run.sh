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
#BSUB -m "lu-gpu ld-gpu lx-gpu ly-gpu lj-gpu ll-gpu ln-gpu lv-gpu"
#BSUB -o out_%J_%I.stdout
#BSUB -eo out_%J_%I.stderr
#BSUB -L /bin/bash

source ~/.bashrc
OPENMM_CPU_THREADS=1
#export OE_LICENSE=~/.openeye/oe_license.txt   # Open eye license activation/env


# chnage dir
echo "changing directory to ${LS_SUBCWD}"
cd ${LS_SUBCWD}


# Report node in use
echo "======================"
hostname
env | sort | grep 'CUDA'
nvidia-smi
echo "======================"


# run job
conda activate openmm
script_path="/home/takabak/data/exploring-rna/rna-espaloma/experiment/tetramer/script"

INPUT_PREFIX="../../../../eq"
TEMP1=275
TEMP2=396

python ${script_path}/repx.py --input_prefix ${INPUT_PREFIX} --temperature1 ${TEMP1} --temperature2 ${TEMP2}