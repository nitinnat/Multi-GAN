# Multi-GAN
An attempt at extending the Dual-GAN to more than two generator-discriminator pairs for unsupervised image-image translation.
This work was done as part of the Advanced Machine Learning (CSE 674) course taught at UB, by Prof. Sargur Srihari.


######################

To run the multi-gan implementation, run: 

* Train:
```
python main_multigan.py --phase train --dataset_name materials --image_size 256 --epoch 45 --lambda 20.0 --channels 3 --batch_size 2
```
* Test:
```
python main_multigan.py --phase test --dataset_name materials --image_size 256 --epoch 45 --lambda 20.0 --channels 3 --batch_size 2
```
To run the vanilla implementation, run:

* Train:
```
python main.py --phase train --dataset_name sketch-photo --image_size 256 --epoch 45 --lambda_A 20.0 --lambda_B 20.0 --A_channels 1 --B_channels 1
```
* Test:
```
python main.py --phase test --dataset_name sketch-photo --image_size 256 --epoch 45 --lambda_A 20.0 --lambda_B 20.0 --A_channels 1 --B_channels 1
```
########################
 # ICCV paper of DualGAN
<a href="https://arxiv.org/abs/1704.02510">DualGAN: unsupervised dual learning for image-to-image translation</a>

please cite the paper, if the codes has been used for your research.

# architecture of DualGAN

![architecture](https://github.com/duxingren14/DualGAN/blob/master/0.png)

# How to setup

## Prerequisites

* Python (2.7 or later)

* numpy

* scipy

* NVIDIA GPU + CUDA 8.0 + CuDNN v5.1

* TensorFlow 1.0 or later

* unzip


# Getting Started
## steps

* clone this repo:

```
git clone https://github.com/duxingren14/DualGAN.git

cd DualGAN
```

* download datasets (e.g., sketch-photo), run:

```
bash ./datasets/download_dataset.sh sketch-photo
```

* train the model:

```
python main.py --phase train --dataset_name sketch-photo --image_size 256 --epoch 45 --lambda_A 20.0 --lambda_B 20.0 --A_channels 1 --B_channels 1
```

* test the model:

```
python main.py --phase test --dataset_name sketch-photo --image_size 256 --epoch 45 --lambda_A 20.0 --lambda_B 20.0 --A_channels 1 --B_channels 1
```

## optional

Similarly, run experiments on facades dataset with the following commands:

```
bash ./datasets/download_dataset.sh facades

python main.py --phase train --dataset_name facades --image_size 256 --epoch 45 --lambda_A 20.0 --lambda_B 20.0 --A_channels 3 --B_channels 3

python main.py --phase test --dataset_name facades --image_size 256 --epoch 45 --lambda_A 20.0 --lambda_B 20.0 --A_channels 3 --B_channels 3
```

# datasets

some datasets can also be downloaded manually from the website. Please cite their papers if you use the data. 

facades: http://cmp.felk.cvut.cz/~tylecr1/facade/

sketch: http://mmlab.ie.cuhk.edu.hk/archive/cufsf/

maps: https://mega.nz/#!r8xwCBCD!lNBrY_2QO6pyUJziGj7ikPheUL_yXA8xGXFlM3GPL3c

oil-chinese:  http://www.cs.mun.ca/~yz7241/, jump to http://www.cs.mun.ca/~yz7241/dataset/

day-night: http://www.cs.mun.ca/~yz7241/dataset/


# Experimental results:

![day2night](https://github.com/duxingren14/DualGAN/blob/master/6.PNG)
![da2ni](https://github.com/duxingren14/DualGAN/blob/master/da2ni.png)
![la2ph](https://github.com/duxingren14/DualGAN/blob/master/la2ph.png)
![ph2la](https://github.com/duxingren14/DualGAN/blob/master/ph2la.png)
![sk2ph](https://github.com/duxingren14/DualGAN/blob/master/sk2ph.png)
![ph2sk](https://github.com/duxingren14/DualGAN/blob/master/ph2sk.png)
![ch2oi](https://github.com/duxingren14/DualGAN/blob/master/ch2oi.png)
![oi2ch](https://github.com/duxingren14/DualGAN/blob/master/oi2ch.png)

# Acknowledgments

Codes are built on the top of pix2pix-tensorflow and DCGAN-tensorflow. Thanks for their precedent contributions!
