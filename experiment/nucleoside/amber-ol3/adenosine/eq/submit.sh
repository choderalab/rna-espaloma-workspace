#!/bin/bash
#BSUB -P "eq"
#BSUB -J "a"
#BSUB -n 1
#BSUB -R rusage[mem=8]
#BSUB -R span[hosts=1]
#BSUB -q gpuqueue
#BSUB -sp 1 # low priority. default is 12, max is 25
#BSUB -gpu num=1:j_exclusive=yes:mode=shared
#BSUB -W 3:00
#BSUB -m "ld-gpu lj-gpu ll-gpu ln-gpu lv-gpu"
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


# conda
conda activate openmm

# setting
name="adenosine"

# run
script_path="/home/takabak/data/exploring-rna/rna-espaloma/experiment/nucleoside/script"
benchmark_path="/home/takabak/data/exploring-rna/rna-benchmark/data/nucleoside"
pdbfile="${benchmark_path}/${name}/01_crd/rna_noh.pdb"

DIR=${PWD}
water_models=('tip3p' 'opc')
for water_model in ${water_models[*]};
do
    echo "process ${water_model}"
    mkdir ${water_model}
    cd ${water_model}
    python ${script_path}/openmm_eq_amber.py --pdbfile ${pdbfile} --water_model ${water_model} --output_prefix .
    cd ${DIR}
done
