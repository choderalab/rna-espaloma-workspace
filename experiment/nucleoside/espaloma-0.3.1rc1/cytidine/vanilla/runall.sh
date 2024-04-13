#!/bin/bash

water_models=('tip3p' 'tip3pfb' 'spce' 'tip4pew' 'tip4pfb' 'opc')
for water_model in ${water_models[*]};
do
    cd ${water_model}
    bsub < submit.sh
    cd ..
done