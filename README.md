## NoisyTwins: Class-consistent and Diverse Image Generation through StyleGANs

Harsh Rangwani*, Lavish Bansal*, Kartik Sharma, Tejan Karmali, Varun Jampani, R. Venkatesh Babu

Vision and AI Lab, IISc Bangalore

[Project Page]()  [Paper (Pdf)]()

### CVPR 2023


**TLDR**: NoisyTwins is a self-supervised regularization scheme for StyleGANs, which helps in alleviating mode collapse and leads to consistent conditional image generation.


## Datasets
CIFAR-10 dataset will be downloaded automatically in ```./data``` folder in the project directory. 

ImageNet-LT is a subset of ImageNet dataset which can be downloaded from this [link](http://image-net.org/index).

The **long-tailed** version of the CIFAR-10  and ImageNet-LT datasets will be created automatically by code.

For iNaturalist2019 dataset please download the files from the following link: [here](https://www.kaggle.com/competitions/inaturalist-2019-fgvc6/data). After download untar the images, following which use the script available [here](https://github.com/facebookresearch/classifier-balancing/blob/main/data/iNaturalist18/gen_lists.py) to create image file names.

## Requirements
For installing all the requirements use the following command:

``
conda env create -f environment.yml -n noisy_twins
``


## Code
The NoisyTwins code is present in `noisy_twins.py` file.

We will provide the code for reproducing experiments from the paper soon.


## Acknowledgements 

PyTorch-StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN


