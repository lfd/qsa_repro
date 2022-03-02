#!/bin/bash

# in case the script is not started from within qsa-repro directory
if [ ! "${PWD}" = "/home/repro/qsa-repro" ]; then
    cd /home/repro/qsa-repro/
fi

cd scripts

# run all RL trainings
./run_rl.sh

# run all MQO experiments
./run_mqo.sh

# generate paper 
./generate_paper.sh

cd ..