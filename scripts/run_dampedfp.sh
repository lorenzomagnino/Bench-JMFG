#!/bin/bash
# Damped Fixed Point experiments (damped, pure, fictitious play)

set -e

ENVIRONMENT="LasryLionsChain"
# ENVIRONMENT="NoInteractionGame"
# ENVIRONMENT="FourRoomsObstacles"
# ENVIRONMENT="RockPaperScissors"
# ENVIRONMENT="SISEpidemic"
# ENVIRONMENT="KineticCongestion"
# ENVIRONMENT="MultipleEquilibriaGame"
# ENVIRONMENT="ContractionGame"

echo "Running Damped FP sweep..."
python main.py -m \
  experiment.name="damped_sweep" \
  experiment.random_seed=42,10,111,1032 \
  algorithm.dampedfp.lambda_schedule=damped \
  algorithm.dampedfp.damped_constant=0.1,0.5,0.8
echo "Generating Damped FP sweep plots..."
PYTHONPATH=src python -m utility.plot_sweep "$ENVIRONMENT" DampedFP_damped

echo "Running Pure FP sweep..."
python main.py -m \
  experiment.name="pure_fp_sweep" \
  experiment.random_seed=42,10,111,1032 \
  algorithm.dampedfp.lambda_schedule=pure
echo "Generating Pure FP sweep plots..."
PYTHONPATH=src python -m utility.plot_sweep "$ENVIRONMENT" DampedFP_pure

echo "Running Fictitious Play sweep..."
python main.py -m \
  experiment.name="fplay_sweep" \
  experiment.random_seed=42,10,111,1032 \
  algorithm.dampedfp.lambda_schedule=fictitious_play
echo "Generating Fictitious Play sweep plots..."
PYTHONPATH=src python -m utility.plot_sweep "$ENVIRONMENT" DampedFP_fictitious_play

echo "All Damped FP experiments completed!"
