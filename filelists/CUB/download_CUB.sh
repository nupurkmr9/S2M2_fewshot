#!/usr/bin/env bash
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz -P ../../Datasets/
tar -zxvf ../../Datasets/CUB_200_2011.tgz -C ../../Datasets/
mv ../../Datasets/CUB_200_2011 ../../Datasets/CUB
rm -rf ../../Datasets/CUB_200_2011.tgz
python make_json.py
