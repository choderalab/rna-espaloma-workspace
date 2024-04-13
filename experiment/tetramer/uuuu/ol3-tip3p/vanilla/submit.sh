#!/bin/bash


cyc1=1
cyc2=5
basename="uuuu-ol3"
DIR=${PWD}

#------------------
for (( i=${cyc1}; i<=${cyc2}; i++ ))
do
    j=$(($i - 1))
    
    # soft link
    if [ ${i} == 1 ]
    then
        ln -s ../eq md0
    fi

    # make input and submit job
    if [ ${i} == ${cyc1} ]
    then
        mkdir md${i}
        sed -e 's/@@@JOBNAME@@@/'${basename}''${i}'/g' \
            -e 's/@@@RESTART_PREFIX@@@/..\/md'${j}'/g' \
            -e 's/@@@INITIALIZE_VELOCITY@@@/False/g' \
            run_template.sh > ./md${i}/run.sh
        chmod u+x ./md${i}/run.sh

        echo "submit job ${i}"
        cd md${i}
        bsub < run.sh
        cd ${DIR}
    else
        mkdir md${i}
        sed -e 's/@@@JOBNAME@@@/'${basename}''${i}'/g' \
            -e 's/@@@RESTART_PREFIX@@@/..\/md'${j}'/g' \
            -e 's/@@@INITIALIZE_VELOCITY@@@/False/g' \
            run_template.sh > ./md${i}/run.sh
        chmod u+x ./md${i}/run.sh

        echo "submit job ${i}"

        cd md${i}
        jobname=${basename}${j}
        bsub -w 'done('${jobname}')' < run.sh
        cd ${DIR}
    fi

    sleep 1
done