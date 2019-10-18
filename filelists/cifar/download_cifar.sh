pip3 install gdown

gdown --id 1pTsCCMDj45kzFYgrnO67BWVbKs48Q3NI --output ../../Datasets/cifar100.zip
unzip ../../Datasets/cifar100.zip -d ../../Datasets/
mv ../../Datasets/cifar100 ../../Datasets/cifar-FS
rm -rf ../../Datasets/cifar100.zip
python make_json.py
