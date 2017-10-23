########################################
# Train estimator for immediate reward #
########################################
python train_estimators.py voted_data_db_1508722816.49.pkl voted_data_round1_1508722837.51.pkl \
    --gpu 2 \
    -bs 128 \
    -ts 10000 \
    -h1 500 300 100 50 \
    -a1 swish \
    -d1 0.1

