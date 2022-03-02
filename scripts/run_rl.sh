#!/bin/bash

# in case the script is not started from within qsa-repro directory
if [ ! "${PWD}" = "/home/repro/qsa-repro" ]; then
    cd /home/repro/qsa-repro/
fi

cd expAnalysis/RL/

echo "started running RL trainings..."

# Run training with all frameworks
echo "started running training with TFQ..."
python3 train_TFQ.py
echo "TFQ training done"

echo "started running training with Pennylane..."
python3 train_PL.py
echo "Pennylane training done"

echo "started running training with Qiskit..."
python3 train_Qiskit.py
echo "Qiskit training done"

echo "all RL trainings done."

cd /home/repro/
