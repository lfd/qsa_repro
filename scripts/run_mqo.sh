#!/bin/bash

# in case the script is not started from within qsa-repro directory
if [ ! "${PWD}" = "/home/repro/qsa-repro" ]; then
    cd /home/repro/qsa-repro/
fi

cd expAnalysis/MQO/

echo "started running MQO experiments..."

# Run experiments with all frameworks
echo "started running MQO Qiskit experiments..."
python3 qiskit/mqo.py
echo "Qiskit experiments done."

echo "started running MQO Pennylane experiments..."
python3 pennylane/mqo.py
echo "Pennylane experiments done."

echo "started running MQO TFQ experiments..."
python3 tensorflowquantum/mqo.py
echo "TFQ experiments done."

echo "started running MQO D-Wave experiments..."
python3 dwave/mqo.py
echo "D-Wave experiments done."

echo "all MQO experiments done."

cd /home/repro/