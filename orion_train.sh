#!/bin/bash
EXP_MAX_TRIAL=25

EXP_NAME="roteq-6d3-truepdir"
EXP_PATH="experiments/$EXP_NAME"
orion -v --debug hunt -c "orion_config.yaml" -n "$EXP_NAME" --exp-max-trials $EXP_MAX_TRIAL python3 main_train.py --config "$EXP_PATH/exp_config.yaml"

EXP_NAME="roteq-8-truepdir"
EXP_PATH="experiments/$EXP_NAME"
orion -v --debug hunt -c "orion_config.yaml" -n "$EXP_NAME" --exp-max-trials $EXP_MAX_TRIAL python3 main_train.py --config "$EXP_PATH/exp_config.yaml"

