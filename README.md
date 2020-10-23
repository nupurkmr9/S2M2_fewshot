S2M2
Charting the Right Manifold: Manifold Mixup for Few-shot Learning
=======

A few-shot classification algorithm: [Charting the Right Manifold: Manifold Mixup for Few-shot Learning](https://arxiv.org/pdf/1907.12087.pdf)

Our code is built upon the code base of [A Closer Look at Few-shot Classification](https://openreview.net/pdf?id=HkxLXnAcFQ) and [Manifold Mixup: Better Representations by Interpolating Hidden States](http://proceedings.mlr.press/v97/verma19a.html)

Running the code
------------

**Training**

DATASETNAME: miniImagenet/cifar/CUB

METHODNAME: S2M2_R/rotation/manifold_mixup


For CIFAR-10

	python train_cifar.py --method [METHODNAME] --model WideResNet28_10 --batch_size <batch_size> --stop_epoch <stop_epoch>
	
For miniImagenet/CUB/tiered-ImageNet

	python train.py --dataset [DATASETNAME] --method [METHODNAME] --model WideResNet28_10 --batch_size <batch_size> --stop_epoch <stop_epoch>
	
* For tiered-ImageNet Novel/Base split taken from https://github.com/renmengye/few-shot-ssl-public
	
------------
**Example Training script to replicate our result on CUB Dataset**: 

* Change directory to filelists/CUB/
* run 'source ./download_CUB.sh' 
* python train.py --dataset CUB --method rotation --model WideResNet28_10 --stop_epoch 200 --batch_size 64 --test_batch_size 16
* python train.py --dataset CUB --method S2M2_R --model WideResNet28_10 --stop_epoch 100 --batch_size 64 --test_batch_size 16


		
**Fetching pretrained WideResNet_28_10 model checkpoints for evaluation**

Directory path to save models should be: checkpoints/[DATASETNAME]/WideResNet28_10_[METHODNAME]/

Pre-trained mdoels can be downloadeded from [https://drive.google.com/open?id=1S-t56H8YWzMn3sjemBcwMtGuuUxZnvb_](https://drive.google.com/open?id=1S-t56H8YWzMn3sjemBcwMtGuuUxZnvb_). Move the tar files for each dataset into 'checkpoints' folder and untar it if required.


**Few-shot evaluation**

Create an empty 'features' directory inside 'S2M2'

	python save_features.py --dataset [DATASETNAME] --method [METHODNAME] --model WideResNet28_10
	python test.py --dataset [DATASETNAME] --method [METHODNAME] --model WideResNet28_10 --n_shot [1/5]


Features of pre-trained network can also be be directly downloaded at this link 'https://drive.google.com/open?id=1JtA7p3sDPksvBmOsJuR4EHw9zRHnKurj' for easy evaluation without the need to download datasets and models. Move the tar files for each dataset into 'features' folder and untar it. 


	
**Comparison with prior/current state-of-the-art methods on mini-ImageNet, CUB and CIFAR-FS dataset.**

Note: We implemented LEO on CUB dataset. Other numbers are reported directly from the paper. 


|      Method    | mini-ImageNet |               |      CUB      |               |   CIFAR-FS     |               |
|:--------------:|:-------------:|:-------------:|:-------------:|:-------------:|:--------------:|:-------------:|
|                |     1-shot    |     5-shot    |     1-shot    |     5-shot    |    1-shot      |     5-shot    |
|   Baseline++   | 57.33 +- 0.10 | 72.99 +- 0.43 |  70.4 +- 0.81 |  82.92 +-0.78 | 67.5 +- 0.64   | 80.08 +- 0.32 |
|       LEO      | 61.76 +- 0.08 | 77.59 +- 0.12 |  68.22+- 0.22 | 78.27 +- 0.16 |       -        |       -       |
|       DCO      | 62.64 +- 0.61 | 78.63 +- 0.46 |       -       |       -       | 72.0 +- 0.7    | 84.2 +- 0.5   |
| Manifold Mixup | 57.6 +- 0.17  | 75.89 +- 0.13 | 73.47 +- 0.89 | 85.42 +- 0.53 | 69.20 +- 0.2   | 83.42 +- 0.15 |               
|    Rotation    | 63.9 +- 0.18  | 81.03 +- 0.11 | 77.61 +- 0.86 | 89.32 +- 0.46 | 70.66 +- 0.2   | 84.15 +- 0.14 |
|     S2M2_R     | 64.93 +- 0.18 | 83.18 +- 0.11 | 80.68 +- 0.81 | 90.85 +- 0.44 | 74.81 +- 0.19  | 87.47 +- 0.13 |


If you use this code for your research, Please cite using

```
@inproceedings{mangla2020charting,
  title={Charting the right manifold: Manifold mixup for few-shot learning},
  author={Mangla, Puneet and Kumari, Nupur and Sinha, Abhishek and Singh, Mayank and Krishnamurthy, Balaji and Balasubramanian, Vineeth N},
  booktitle={The IEEE Winter Conference on Applications of Computer Vision},
  pages={2218--2227},
  year={2020}
}

```



References
------------
[A Closer Look at Few-shot Classification](https://openreview.net/pdf?id=HkxLXnAcFQ)

[Meta-Learning with Latent Embedding Optimization](https://arxiv.org/pdf/1807.05960.pdf)

[Meta Learning with Differentiable Convex Optimization](https://arxiv.org/pdf/1904.03758.pdf)

[Manifold Mixup: Better Representations by Interpolating Hidden States](http://proceedings.mlr.press/v97/verma19a.html)
