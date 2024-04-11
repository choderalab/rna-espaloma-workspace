#!/bin/bash
#BSUB -P "repx"
#BSUB -J "population"
#BSUB -n 1
#BSUB -R rusage[mem=16]
#BSUB -R span[hosts=1]
#BSUB -q cpuqueue
#BSUB -sp 1 # low priority. default is 12, max is 25
#BSUB -W 3:00
#BSUB -L /bin/bash
#BSUB -o pop_%J_%I.stdout
#BSUB -eo pop_%J_%I.stderr

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

# setitting
npzfile="mydata.npz"
ncfile="../enhanced.nc"
plot_title="GACC RNA.OL3"
#output_prefix="./test"
script_path="/home/takabak/data/exploring-rna/rna-espaloma/experiment/tetramer/script"

# run
python ${script_path}/reweight_mbar_population.py --npzfile ${npzfile} --ncfile ${ncfile} --plot_title "${plot_title}"
