## NoisyTwins: Class-consistent and Diverse Image Generation through StyleGANs

Harsh Rangwani*, Lavish Bansal*, Kartik Sharma, Tejan Karmali, Varun Jampani, R. Venkatesh Babu

Vision and AI Lab, IISc Bangalore

[[Project Page](https://rangwani-harsh.github.io/NoisyTwins/)] [[Paper (Pdf)](https://arxiv.org/abs/2304.05866)]

**CVPR 2023**

<p align="center">
  <img width="50%" src="https://github.com/val-iisc/NoisyTwins/blob/main/assets/Teaser.jpg?raw=true" />
</p>

**TLDR**: NoisyTwins is a self-supervised regularization scheme for StyleGANs, which helps in alleviating mode collapse and leads to consistent conditional image generation.


## Datasets
CIFAR-10 dataset will be downloaded automatically in ``./data`` folder in the project directory. 

ImageNet-LT is a subset of ImageNet dataset which can be downloaded from this [link](http://image-net.org/index).

The **long-tailed** version of the CIFAR-10 and ImageNet-LT datasets will be created automatically by code.

For iNaturalist2019 dataset please download the files from the following link: [here](https://www.kaggle.com/competitions/inaturalist-2019-fgvc6/data). After download untar the images, following which use the script available [here](https://github.com/facebookresearch/classifier-balancing/blob/main/data/iNaturalist18/gen_lists.py) to create image file names.

**NOTE:** For ***iNaturalist19*** and ***ImageNet-LT***, the ``-l`` and ``-hdf5`` flags are passed to use the HDF5 files. As these datasets are very large in size, hdf5 files are required for faster processing which provide the data in chunks. Once the HDF5 is created for a specific dataset with a given image resolution in the first run, make sure the ``DATA_DIR`` in ``run_script.sh`` is set to the HDF5 file's directory, so that it doesn't reads the complete dataset everytime and simply use the file.

For **Imagenet-Carnivores** dataset, we provide the zip files in ``data`` folder. Unzip the files there, which will correspond to the path in ``DATA_DIR``. The dataset was originally provided by the authors of Shahbazi et. al. [Link](https://github.com/mshahbazi72/transitional-cGAN) which they used for their experiements.

For **AnimalFace**, the dataset can be downloaded from [here](https://vcla.stat.ucla.edu/people/zhangzhang-si/HiT/exp5.html). After downloading the zip file of dataset, unzip the file and remove the ``Natural`` folder from it, as it contains 5-6 sample images.

## Requirements
For installing all the requirements use the following command:

```
conda env create -f environment.yml -n studiogan
```

<!-- For starting, first login to your WANDB account using your personal API key. (Obtained from WANDB account)
 ```bash
wandb login PERSONAL_API_KEY
 ``` -->

## Running Experiments
To run the experiements, the scripts have been provided in ``run_script.sh`` file.

Depending on the experiment setting that you want to keep for a run(experiment), changes can be done in the ``run_script.sh`` file.

The scripts for 3 tasks and all 5 datasets used in the paper have been provided. 

Tasks include Training the model, Evaluating a trained model, Visualizing the results from a trained model. Change the ``TASK`` parameter to perform the required task. Similarly change the ``DATASET`` name for the required dataset.

Change your Personal API KEY ``WANBD_API_KEY`` in the ``WANDB_API`` parameter to connect to experiement to your WANDB account.

### Changing Configs for running different experiments.
For a specific dataset, changing the ``CFG`` parameter will read the specified config file in ``src/configs/{DATASET}``. These configs specify the exhuastive set of hyperparameters for the experiments. 

**NOTE**: Full path of the config file need to be specified.

For running the NoisyTwins experiment, keep the config as follows.
(suppose for iNat19)
```bash
CFG="src/configs/iNat19/StyleGAN2-SPD-ADA-LC-NoisyTwins.yaml"
```
Similarly for a baseline 
```bash
CFG="src/configs/iNat19/StyleGAN2-SPD-ADA-LC.yaml"
```

The ``DATA_DIR`` parameter can also be changed depending on the location of the dataset or HDF5 files of the dataset.

You can change the config files for tuning hyper-parameters or create your own custom config files and datasets and run experiments accordingly.
For hyper-parameters not specified in config file, their default values will be taken from the ``config.py`` file.

## Code Implementation of NoisyTwins in different files
Most of the code related to the noise augmentation has been added in ``src/models/stylegan2.py``. The code of Barlow Twins based NoisyTwins Regulariser has been present in ``src/utils/barlow.py`` and ``src/utils/sample.py`` for creating the twins and loss calculation. In ``src/worker.py``, the NoisyTwins loss in integrated with other losses in the main training loop.

## Citation
If you find our code or work useful in any way, please consider citing us:

```bibtex

```
Please contact ``harshr@iisc.ac.in`` in case you have any comments or suggestions.

## Acknowledgements 
Our code is based on StudioGAN and supports all the functionality offered by StudioGAN.

PyTorch-StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN





