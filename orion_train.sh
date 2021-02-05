#!/bin/bash
EXP_MAX_TRIAL=25

EXP_NAME="roteq-4"
EXP_PATH="experiments/$EXP_NAME"
orion -v --debug hunt -c "orion_config.yaml" -n "$EXP_NAME" --exp-max-trials $EXP_MAX_TRIAL python3 main_train.py --config "$EXP_PATH/exp_config.yaml"

