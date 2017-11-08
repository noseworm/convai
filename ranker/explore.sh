###
# RUN SHORT TERM RANKER IN EXPLORATION MODE
###
hyperdash run -n "explore short term ranker" python train.py \
    data/voted_data_db_1510012489.57.json \
    data/voted_data_round1_1510012503.6.json \
    --gpu 0 \
    --explore 1000 \
    --threshold 0.63

