#!/bin/bash

name='g'

water_models=('tip3p' 'tip3pfb' 'spce' 'tip4pew' 'tip4pfb' 'opc')
for water_model in ${water_models[*]};
do
    mkdir -p ${water_model}
    sed -e 's/@@@BASENAME@@@/'${name}'-'${water_model}'/g' \
        -e 's/@@@WATER_MODEL@@@/'${water_model}'/g' \
        template/submit.sh > ${water_model}/submit.sh
    chmod u+x ${water_model}/submit.sh
done