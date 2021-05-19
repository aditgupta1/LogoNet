Pytorch implementations of [DCGAN](https://arxiv.org/abs/1511.06434), [LSGAN](https://arxiv.org/abs/1611.04076), [WGAN-GP](http://arxiv.org/abs/1704.00028)([LP](https://arxiv.org/abs/1709.08894)) and [DRAGAN](https://arxiv.org/abs/1705.07215v5).

Our Code based off of DCGAN, WGAN, WGAN-GP, and LSGAN implementations in Tensorflow from https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Tensorflow-2. Our code is completely in Pytorch

# Requirements
- PyTorch 1.1
- tensorboardX
- scikit-image, oyaml, tqdm
- Python 3.6

## File Structure of /models folder
- train.py: source code to train generator and discriminatorr and store samples of the generated images
- module.py: generator and discriminator classes
- data.py: preprocessing datasets for train.py
- imlib/pylib/torchlib/torchprob: various util files/folders



