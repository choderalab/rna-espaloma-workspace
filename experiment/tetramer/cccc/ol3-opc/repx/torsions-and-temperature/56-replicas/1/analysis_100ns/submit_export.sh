#!/bin/bash
#BSUB -P "repx"
#BSUB -J "export"
#BSUB -n 1
#BSUB -R rusage[mem=64]
#BSUB -R span[hosts=1]
#BSUB -q cpuqueue
#BSUB -sp 1 # low priority. default is 12, max is 25
#BSUB -W 3:00
#BSUB -L /bin/bash
#BSUB -o export_%J_%I.stdout
#BSUB -eo export_%J_%I.stderr

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


conda activate openmm
script_path="/home/takabak/data/exploring-rna/rna-espaloma/experiment/tetramer/script"

frame=-1
python ${script_path}/export_snapshot.py --input_prefix "../" --ref_prefix "../../../../../eq" --keep_solvent True --frame ${frame}
