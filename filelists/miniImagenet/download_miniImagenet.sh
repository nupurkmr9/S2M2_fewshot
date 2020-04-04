#!/usr/bin/env bash
wget https://github.com/twitter/meta-learning-lstm/blob/master/data/miniImagenet/test.csv
wget https://github.com/twitter/meta-learning-lstm/blob/master/data/miniImagenet/train.csv
wget https://github.com/twitter/meta-learning-lstm/blob/master/data/miniImagenet/val.csv
wget http://image-net.org/image/ILSVRC2015/ILSVRC2015_CLS-LOC.tar.gz -P ../../Datasets/
tar -zxvf ILSVRC2015_CLS-LOC.tar.gz -C ../../Datasets/
python make_json.py