# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/data_util.py

import os
import random

import torchvision
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.datasets import ImageFolder
from scipy import io
from PIL import ImageOps, Image
import torch
import torchvision.transforms as transforms
import h5py as h5
import numpy as np
import json 

import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, download_file_from_google_drive, extract_archive

from collections import Counter

class RandomCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """
    def __call__(self, img):
        size = (min(img.size), min(img.size))
        # Only step forward along this edge if it's the long edge
        i = (0 if size[0] == img.size[0] else np.random.randint(low=0, high=img.size[0] - size[0]))
        j = (0 if size[1] == img.size[1] else np.random.randint(low=0, high=img.size[1] - size[1]))
        return transforms.functional.crop(img, j, i, size[0], size[1])

    def __repr__(self):
        return self.__class__.__name__


class CenterCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """
    def __call__(self, img):
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__

class CUB200(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, 
                loader=default_loader, 
                download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train
        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        # if self.train:
        #     self.data = self.data[self.data.is_training_img == 1]
        # else:
        #     self.data = self.data[self.data.is_training_img == 0]
        
        if not self.train:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False
        
        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)
        # download_file_from_google_drive(self.file_id, self.root)
        # extract_archive(os.path.join(self.root, self.filename))

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

class LT_Dataset(Dataset):
    
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

class FewShot_Dataset(ImageFolder):
    def __init__(self, root, transform=None):
        self.root = root
        super().__init__(self.root)
        self.files=[]
        for folder in os.listdir(root):
            img_paths_per_class = [folder+"/"+fname for fname in os.listdir(root+"/"+folder) if fname.endswith('.jpg')]
            self.files += img_paths_per_class
        self.transform = transform
    
    def __len__(self):
        return len(self.files)

class Dataset_(Dataset):
    def __init__(self,
                 data_name,
                 data_dir,
                 train,
                 crop_long_edge=False,
                 resize_size=None,
                 random_flip=False,
                 normalize=True,
                 hdf5_path=None,
                 load_data_in_memory=False):
        super(Dataset_, self).__init__()
        self.data_name = data_name
        self.data_dir = data_dir
        self.train = train
        self.random_flip = random_flip
        self.normalize = normalize
        self.hdf5_path = hdf5_path
        self.load_data_in_memory = load_data_in_memory
        self.trsf_list = []

        if self.hdf5_path is None:
            if crop_long_edge:
                self.trsf_list += [CenterCropLongEdge()]
            if resize_size is not None:
                self.trsf_list += [transforms.Resize(resize_size, Image.LANCZOS)]
        else:
            self.trsf_list += [transforms.ToPILImage()]

        if self.random_flip:
            self.trsf_list += [transforms.RandomHorizontalFlip()]

        if self.normalize:
            self.trsf_list += [transforms.ToTensor()]
            self.trsf_list += [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        else:
            self.trsf_list += [transforms.PILToTensor()]

        self.trsf = transforms.Compose(self.trsf_list)

        self.load_dataset()

    def load_dataset(self):
        if self.hdf5_path is not None:
            with h5.File(self.hdf5_path, "r") as f:
                data, labels = f["imgs"], f["labels"]
                self.num_dataset = data.shape[0]
                if self.load_data_in_memory:
                    print("Load {path} into memory.".format(path=self.hdf5_path))
                    self.data = data[:]
                    self.labels = labels[:]

            counts = Counter(self.labels)
            self.img_num_list = [0] * len(counts)
            for i in range(len(counts)):
                self.img_num_list[i] = counts[i]
            return

        if self.data_name == "CIFAR10_LT":
            if self.train==True:
                self.data = IMBALANCECIFAR10(dataset_name=self.data_name, root=self.data_dir, train=self.train, download=True)
            else:
                self.data = IMBALANCECIFAR10(dataset_name=self.data_name, root=self.data_dir, train=self.train, download=True, imb_factor=1.0)
        elif self.data_name == "CIFAR10":
            self.data = CIFAR10(root=self.data_dir, train=self.train, download=True)
        elif self.data_name == "CIFAR100":
            self.data = CIFAR100(root=self.data_dir, train=self.train, download=True)
        elif self.data_name == "CUB200":
            self.data = CUB200(root=self.data_dir, train=self.train, download=True)
        elif self.data_name == "iNat19":
            txt = './data/iNat19/iNaturalist19_train.txt' if self.train == True else './data/iNat19/iNaturalist19_val.txt'
            self.data = LT_Dataset(self.data_dir, txt)
        elif self.data_name =="imagenet_lt":
            txt = './data/imagenet_lt/ImageNet_LT_train.txt' if self.train == True else './data/imagenet_lt/ImageNet_LT_val.txt'
            self.data = LT_Dataset(self.data_dir, txt)
        elif self.data_name in ["Imgnet_carniv", "AnimalFace_FS"]:
            assert self.train == True, "Imgnet_carniv and Animal Face only support train dataset"
            self.data = FewShot_Dataset(self.data_dir)
        else:
            mode = "train" if self.train == True else "valid"
            root = os.path.join(self.data_dir, mode)
            self.data = ImageFolder(root=root)
        
        self.labels = []
        for _,lbls in self.data:
            self.labels.append(lbls)

        counts = Counter(self.labels)
        self.img_num_list = [0] * len(counts)
        for i in range(len(counts)):
            self.img_num_list[i] = counts[i]

    def _get_hdf5(self, index):
        with h5.File(self.hdf5_path, "r") as f:
            return f["imgs"][index], f["labels"][index]

    def __len__(self):
        if self.hdf5_path is None:
            num_dataset = len(self.data)
        else:
            num_dataset = self.num_dataset
        return num_dataset

    def __getitem__(self, index):
        if self.hdf5_path is None:
            img, label = self.data[index]
        else:
            if self.load_data_in_memory:
                img, label = self.data[index], self.labels[index]
            else:
                img, label = self._get_hdf5(index)
            
            if img.shape[0]==3:    ## if img of shape CxHxW, change to HxWxC
                img, label = np.transpose(img, (1,2,0)), int(label)
        return self.trsf(img), int(label)

class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self,dataset_name, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        self.dataset_name = dataset_name
        np.random.seed(rand_number)
        self.img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)

        self.gen_imbalanced_data(self.img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        
        return cls_num_list


class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100



class IMBALANCELSUN(torchvision.datasets.LSUN):
    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, classes="val",
                 transform=None, target_transform=None, max_samples = None):
        super(IMBALANCELSUN, self).__init__(root, classes, transform, target_transform,)
        np.random.seed(rand_number)
        self.dataset_name = 'lsun'
        self.max_samples = max_samples
        self.img_num_list = self.get_img_num_per_cls(len(self.classes), imb_type, imb_factor)
        self.gen_imbalanced_data(self.img_num_list)
        
    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = self.max_samples
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0 + 1e-9)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_list):

        modified_self_indices = []
        count = 0
        for c in img_num_list:
            count += c
            modified_self_indices.append(count)

        self.indices = modified_self_indices
        self.length = count