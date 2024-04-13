#!/bin/bash
#BSUB -P "eq"
#BSUB -J "u"
#BSUB -n 1
#BSUB -R rusage[mem=8]
#BSUB -R span[hosts=1]
#BSUB -q gpuqueue
#BSUB -sp 1 # low priority. default is 12, max is 25
#BSUB -gpu num=1:j_exclusive=yes:mode=shared
#BSUB -W 3:00
##BSUB -m "lu-gpu ld-gpu lx-gpu ly-gpu lj-gpu ll-gpu ln-gpu lv-gpu"
#BSUB -m "ld-gpu lj-gpu ll-gpu ln-gpu lv-gpu"
#BSUB -o out_%J_%I.stdout
#BSUB -eo out_%J_%I.stderr

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
conda activate openmm


# settting
name="uridine"


# run
DIR=${PWD}
script_path="/home/takabak/data/exploring-rna/rna-espaloma/experiment/nucleoside/script"
water_models=('tip3p' 'tip3pfb' 'spce' 'tip4pew' 'tip4pfb' 'opc')
for water_model in ${water_models[*]};
do
    echo "process ${water_model}"
    mkdir ${water_model}
    cd ${water_model}
    restart_prefix="../../prep/${water_model}"
    python ${script_path}/openmm_eq.py -i state.pdb --restart_prefix ${restart_prefix} --output_prefix .
    cd ${DIR}
done