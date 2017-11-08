#!/usr/bin/env bash

########################################
# Train estimator for immediate reward #
########################################
python train.py ./data/voted_data_db_1510012489.57.json ./data/voted_data_round1_1510012503.6.json \
    --gpu 0 \
    --batch_size 128 \
    --patience 20 \
    --hidden_sizes 900 500 300 100 50 \
    --activation swish \
    --dropout_rate 0.1 \
    --optimizer adam \
    --learning_rate 0.001

