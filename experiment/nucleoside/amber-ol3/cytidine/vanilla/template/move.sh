#!/bin/bash


cyc1=1
cyc2=1
DIR=${PWD}

#------------------
for (( i=${cyc1}; i<=${cyc2}; i++ ))
do
    echo "md${i}"
    mv md${i} _obsolete
done