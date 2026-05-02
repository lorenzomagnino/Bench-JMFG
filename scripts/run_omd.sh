#!/bin/bash
# Online Mirror Descent experiments

set -e

ENVIRONMENT="LasryLionsChain"
# ENVIRONMENT="NoInteractionGame"
# ENVIRONMENT="FourRoomsObstacles"
# ENVIRONMENT="RockPaperScissors"
# ENVIRONMENT="SISEpidemic"
# ENVIRONMENT="KineticCongestion"
# ENVIRONMENT="MultipleEquilibriaGame"
# ENVIRONMENT="ContractionGame"

echo "Running OMD sweep..."
python main.py -m \
  experiment.name="omd_sweep" \
  experiment.random_seed=42,10,111,1032 \
  algorithm.omd.learning_rate=0.5,0.05,0.005 \
  algorithm.omd.temperature=0.2,0.5,0.8

echo "Generating OMD sweep plots..."
PYTHONPATH=src python -m utility.plot_sweep "$ENVIRONMENT" OMD

echo "All OMD experiments completed!"
