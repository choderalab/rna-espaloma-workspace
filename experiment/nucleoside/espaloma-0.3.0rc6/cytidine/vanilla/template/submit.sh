#!/bin/bash


cyc1=1
cyc2=1
basename="@@@BASENAME@@@"
DIR=${PWD}

#------------------
for (( i=${cyc1}; i<=${cyc2}; i++ ))
do
    j=$(($i - 1))
    # soft link
    if [ ${i} == 1 ]
    then
        ln -s ../../eq/@@@WATER_MODEL@@@ md0
    fi
    # make input and submit job
    mkdir md${i}
    sed -e 's/@@@JOBNAME@@@/'${basename}''${i}'/g' \
        -e 's/@@@RESTART_PREFIX@@@/..\/md'${j}'/g' \
        -e 's/@@@INITIALIZE_VELOCITY@@@/True/g' \
        ../template/_run.sh > ./md${i}/run.sh
    chmod u+x ./md${i}/run.sh

    echo "submit job ${i}"
    cd md${i}
    bsub < run.sh
    cd ${DIR}
    
    sleep 1
done