#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model twolayernn \
    --hidden-dim 30 \
    --epochs 2 \
    --weight-decay 0.0001 \
    --momentum 0.4 \
    --batch-size 500 \
    --lr 0.01 | tee twolayernn.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
