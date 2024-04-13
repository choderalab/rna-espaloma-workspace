#!/bin/bash
#BSUB -P "nucleoside"
#BSUB -J "prep"
#BSUB -n 1
#BSUB -R rusage[mem=8]
#BSUB -R span[hosts=1]
#BSUB -q cpuqueue
#BSUB -sp 1 # low priority. default is 12, max is 25
#BSUB -W 0:30
#BSUB -o out_%J_%I.stdout
#BSUB -eo out_%J_%I.stderr
#BSUB -L /bin/bash

source ~/.bashrc
OPENMM_CPU_THREADS=1
#export OE_LICENSE=~/.openeye/oe_license.txt   # Open eye license activation/env


# change dir
echo "changing directory to ${LS_SUBCWD}"
cd $LS_SUBCWD


# Report node in use
echo "======================"
hostname
env | sort | grep 'CUDA'
nvidia-smi
echo "======================"


# conda
conda activate openmmforcefields-dev

# settting
name="cytidine"
net_model="/home/takabak/.espaloma/espaloma-0.3.1rc1.pt"

# run
script_path="/home/takabak/data/exploring-rna/rna-espaloma/experiment/nucleoside/script"
benchmark_path="/home/takabak/data/exploring-rna/rna-benchmark/data/nucleoside"
pdbfile="${benchmark_path}/${name}/01_crd/rna_noh.pdb"

DIR=${PWD}
water_models=('tip3p' 'tip3pfb' 'spce' 'tip4pew' 'tip4pfb' 'opc')
for water_model in ${water_models[*]};
do
    echo "process ${water_model}"
    mkdir ${water_model}
    cd ${water_model}
    python ${script_path}/create_system_espaloma.py --pdbfile ${pdbfile} --water_model ${water_model} --net_model ${net_model}
    cd ${DIR}
done