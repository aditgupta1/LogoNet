# logoGAN

## Overview 
In this project, we investigate how to provide artistic insipiration by generating a vast variety of logo designs. While there exists a small amount of literature on this highly unstable and large problem relating to logo generation, prior work hasn't been able to demonstrate stable generative models without the use intricate unsupervised clustering of the images. We show the creation of multiple stable generative adversarial networks (GANs) that are able to produce logos by optimizing the loss function used - this provides better gradients to the generator to learn during the early epochs. Specifically, we experiment with vanilla DCGAN's, WGAN, WGAN-GP, and LSGAN (see paper for full description). 

The general architecture of the models is as follows:

### Discrminator:

<img src="https://github.com/aditgupta1/logoGAN/blob/main/models/architecture_imgs/disc_img.png" alt="drawing" width="900"/>

### Generator
<img src="https://github.com/aditgupta1/logoGAN/blob/main/models/architecture_imgs/gen_img.png" alt="drawing" width="900"/>

<!-- 
![alt text](?raw=true)

![alt text](https://github.com/aditgupta1/logoGAN/blob/main/models/architecture_imgs/gen_img.png?raw=true) -->


## Usage 

Clone the repository

Run  ```python models/train.py ``` to run the model.

Customize model by changing parameters. This includes the following:

Parameter | Bash Script | Options
--- | --- | ---
Epochs | '--epochs' | Any
Dataset| '--dataset' | 'logo', 'cifar10', 'fashion_mnist', 'mnist', 'celeba'
Learning Rate | '--lr' | Any
Loss Mode (i.e. Wasserstein, Least Squares, minimax, etc) | '--adversarial_loss_mode' | 'gan', 'lsgan', 'wgan'
Gradient Penalty Mode | '--gradient_penalty_mode' | 'none', '1-gp', '0-gp', 'lp'
Gradient Penalty Weight | '--gradient_penalty_weight' | Any
Experiment Name | '--experiment_name' | Any


To train DCGAN for 50 epochs with learning rate 0.0001, run 

```python models/train.py --epoch=50 --experiment_name=dcgan --lr=0.0001```

To train LSGAN for 50 epochs with learning rate 0.0001, run 

```python models/train.py --epoch=50 --adversarial_loss_mode=lsgan --experiment_name=lsgan --lr=0.0001```

To train WGAN-GP with gp-mode 1 for 50 epochs with learning rate 0.0001, run 

```python models/train.py --epoch=50 --adversarial_loss_mode=wgan --gradient_penalty_mode=1-gp --experiment_name=wgan --lr=0.0001```

Create an "outputs" folder inside of your models folder. Within the output folder, a "summaries" folder will be created that stores summaries of the models that are automatically saved (using TensorboardX) during training. Also the images generated at each time step will be stored in "samples_training" (also within outputs) and the folder is created by train.py .

## Requirements 
See requirements.txt 

## Team Members
- Adit Gupta
- Max Yuan
- Chae Young Lee
- Darren Liang

This project was completed as part of the CPSC452 (Deep Learning Theory) final project with Prof. Krishnaswamy. 
