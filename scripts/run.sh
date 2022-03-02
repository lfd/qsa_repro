#!/bin/bash

if [ $# -eq 0 ]; then
	echo "Usage: ./scripts/run.sh [all|experiments_only|rl_only|mqo_only|paper_only|bash]"
	exit 1
fi

# in case the script is not started from within qsa-repro directory
if [ ! "${PWD}" = "/home/repro/qsa-repro" ]; then
    cd /home/repro/qsa-repro/
fi

cd scripts/

if [ "$1" = "all" ]; then
	./run_all.sh
elif [ "$1" = "experiments_only" ]; then
	./run_rl.sh
    ./run_mqo.sh
elif [ "$1" = "rl_only" ]; then
	./run_rl.sh
elif [ "$1" = "mqo_only" ]; then
	./run_mqo.sh
elif [ "$1" = "paper_only" ]; then
	./generate_paper.sh
elif [ "$1" = "bash" ]; then
	# launch shell
	cd ..
	/bin/bash
	exit 0
else
    echo "Usage: ./scripts/run.sh [all|experiments_only|rl_only|mqo_only|paper_only|bash]"
fi

cd ..

# launch shell
/bin/bash