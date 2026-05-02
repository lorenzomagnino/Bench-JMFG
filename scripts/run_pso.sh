#!/bin/bash
# Particle Swarm Optimization experiments

set -e

ENVIRONMENT="LasryLionsChain"
# ENVIRONMENT="NoInteractionGame"
# ENVIRONMENT="FourRoomsObstacles"
# ENVIRONMENT="RockPaperScissors"
# ENVIRONMENT="SISEpidemic"
# ENVIRONMENT="KineticCongestion"
# ENVIRONMENT="MultipleEquilibriaGame"
# ENVIRONMENT="ContractionGame"

echo "Running PSO sweep..."
python main.py -m \
  experiment.name="pso_sweep" \
  experiment.random_seed=42,10,111,1032 \
  algorithm.pso.w=0.3,0.7 \
  algorithm.pso.c1=0.3,0.7,1.2 \
  algorithm.pso.c2=0.3,0.6,1.2 \
  algorithm.pso.temperature=0.2,0.7

echo "Generating PSO sweep plots..."
PYTHONPATH=src python -m utility.plot_sweep "$ENVIRONMENT" PSO

echo "All PSO experiments completed!"
