#!/usr/bin/env bash

########################################
# Train estimator for immediate reward #
########################################
python train_estimators.py voted_data_db_1509563020.37.pkl voted_data_round1_1509562999.05.pkl \
    --gpu 2 \
    -bs 128 \
    -h1 500 300 100 50 \
    -a1 swish \
    -d1 0.1

