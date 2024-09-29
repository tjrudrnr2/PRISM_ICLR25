#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import logging
from torchvision import datasets, transforms
from  torch.autograd import Variable
from torchvision.transforms import InterpolationMode
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
from torch.utils import data
from scipy.stats import bernoulli
from torch.distributions.bernoulli import Bernoulli
from scipy.spatial import distance
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

from itertools import chain
from pathlib import Path

def untrack_bn_statistics(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.track_running_stats = False


def track_bn_statistics(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.track_running_stats = True
        
def listdir(dname):
    # 해당 경로 하위의 모든 image파일 경로 
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames

transform_setting = {
                'vgg19-pytorch':
                {
                'interpolation' : InterpolationMode.BILINEAR, 
                'normalize':((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))},
                }

def compute_distance(dict1, dict2, dist_type="hd"):
    dist = 0
    mask_dict_1 = torch.tensor([], device=dict1[next(iter(dict1))].device)
    mask_dict_2 = torch.tensor([], device=dict2[next(iter(dict2))].device)
    for k, v in dict1.items():
        if 'scores' in k and 'bias' not in k:
            mask_dict_1 = torch.cat((mask_dict_1, v.flatten()), dim=0)
    for k, v in dict2.items():
        if 'scores' in k and 'bias' not in k:
            mask_dict_2 = torch.cat((mask_dict_2, v.flatten()), dim=0)
            
    if dist_type == "hd":      
        dist = (mask_dict_1 != mask_dict_2).float().mean().item()  
    elif dist_type == "cos":
        dot_product = torch.dot(mask_dict_1, mask_dict_2)
        
        norm_vector1 = torch.norm(mask_dict_1)
        norm_vector2 = torch.norm(mask_dict_2)
        
        dist = dot_product / (norm_vector1 * norm_vector2)
        dist = dist.item()
        dist = 1 - dist
        
    return dist

def compare_global_mask(args, global_model, aggregated_weights, globalmask_corr, epoch, device):
    ###########################
    ### Compute server mask ###
    ###########################
    server_mask = dict()
    aggregated_mask = dict()
    with torch.no_grad():
        for k, v in global_model.state_dict().items():
            if 'scores' in k:
                target = v.to(device)
                theta = torch.sigmoid(target)
                
                updates_s = Bernoulli(theta).sample().to(device)
                if args.dataset == 'mnist' and 'net.lastConv.scores' in k:
                    updates_s = updates_s.squeeze(0)
                if server_mask.get(k) is None:
                    server_mask[k] = updates_s
                else:
                    server_mask[k] += updates_s
            else:
                if server_mask.get(k) is None:
                    server_mask[k] = v.to(device)
                else:
                    server_mask[k] += v.to(device)
                    
    with torch.no_grad():
        for k, v in aggregated_weights.items():
            if 'scores' in k:
                target = v.to(device)
                theta = torch.sigmoid(target)
                
                updates_s = Bernoulli(theta).sample().to(device)
                if args.dataset == 'mnist' and 'net.lastConv.scores' in k:
                    updates_s = updates_s.squeeze(0)
                if aggregated_mask.get(k) is None:
                    aggregated_mask[k] = updates_s
                else:
                    aggregated_mask[k] += updates_s
            else:
                if aggregated_mask.get(k) is None:
                    aggregated_mask[k] = v.to(device)
                else:
                    aggregated_mask[k] += v.to(device)
                 
    hamming_distances = compute_distance(server_mask, aggregated_mask, dist_type=args.dist_type)
    globalmask_corr.append(hamming_distances)
    
    # 결과 출력
    print(f"Round {epoch} Distance: {hamming_distances}")
    
    return hamming_distances
    
def compare_mask(args, global_model, client_list, epoch, device):
    ###########################
    ### Compute server mask ###
    ###########################
    server_mask = dict()
    with torch.no_grad():
        for k, v in global_model.state_dict().items():
            if 'scores' in k:
                target = v.to(device)
                theta = torch.sigmoid(target)
                
                updates_s = bernoulli.rvs(theta.cpu().numpy())
                updates_s = np.where(updates_s == 0, 0.01, updates_s)
                updates_s = np.where(updates_s == 1, 0.09, updates_s)
                if args.dataset == 'mnist' and 'net.lastConv.scores' in k:
                    updates_s = np.expand_dims(updates_s, axis=0)
                if server_mask.get(k) is None:
                    server_mask[k] = torch.tensor(updates_s, device=device)
                else:
                    server_mask[k] += torch.tensor(updates_s, device=device)
            else:
                if server_mask.get(k) is None:
                    server_mask[k] = torch.tensor(v, device=device)
                else:
                    server_mask[k] += torch.tensor(v, device=device)
    
    client_masks = [client.upload_mask() for client in client_list]
    
    hamming_distances = [compute_distance(server_mask, client_mask) for client_mask in client_masks]
    
    for i, dist in enumerate(hamming_distances):
        print(f"Client {i + 1} Hamming Distance: {dist}")
    

def getNetImageSizeAndNumFeats(net, setEncToEval=False, verbose=False, image_size = 32, use_cuda = True, device=None, nc = 3):
    '''return two list: 
    - list of size of output (image) for each layer
    - list of size of total number of features (nFeatMaps*featmaps_height,featmaps_width) '''
    
    # classifier set to eval mode
    # if opt.setEncToEval is True, classifier is used when precomputing real dataset, and calculating generator.
    if setEncToEval:
        net.eval()

    if use_cuda:
        y, layers = net(Variable(torch.randn(1,nc,image_size,image_size).to(device)))
    else:
        y, layers = net(Variable(torch.randn(1,nc,image_size,image_size)))


    layer_img_size = []
    layer_num_feats = []
    for L in reversed(layers):
        if len(L.size()) == 4:
            layer_img_size.append(L.size(2))
            layer_num_feats.append(L.size(1)*L.size(2)*L.size(3))
        elif len(L.size()) == 3:    # vit
            print('L size', L.shape)
            layer_img_size.append(1)
            layer_num_feats.append(L.size(1)*L.size(2))

        elif len(L.size()) == 2:
            layer_img_size.append(1)
            layer_num_feats.append(L.size(1))
        elif len(L.size()) == 1:
            logging.info('only one layer')
            layer_num_feats.append(L.size(0))
        else:
            assert 0, 'not sure how to handle this layer size '+L.size()
    if verbose:
        logger.info("# Layer img sizes: {}".format(layer_img_size))
        logger.info("# Layer num feats: {}".format(layer_num_feats))

    return layer_img_size, layer_num_feats

class DefaultDataset(data.Dataset):
    def __init__(self, root,transform=None):
        self.samples=listdir(root)
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img
    def __len__(self):
        return len(self.samples)

class CustomCelebADataset(Dataset):
    def __init__(self, root_dir, split, transforms=None):
        self.image_folder = 'img_align_celeba'
        self.root_dir = root_dir

        self.annotation_file = 'list_eval_partition.csv'
        self.label_file = 'list_attr_celeba.csv'
        self.transform = transforms
        self.split = split
        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[self.split]
        
        df = pd.read_csv(
            'CELEBA_PATH' 
            + self.annotation_file)
        self.label_df = pd.read_csv(
            'CELEBA_PATH' 
                + self.label_file)
        self.filename = df.loc[df['partition']
                               == split_, :].reset_index(drop=True)
        
        split_idx = [i for i, (x, y) in enumerate(zip(self.filename['image_id'].values, 
                                                      self.label_df['image_id'].values)) if x == y]
        self.targets = self.label_df.loc[split_idx].values[:, 1:]
        
        self.length = len(self.filename)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = int(idx)
        img = Image.open(os.path.join('CELEBA_PATH', self.image_folder,
                         self.filename.iloc[idx, ].values[0])).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        # targets = self.targets[idx]
        targets = False
        
        # return img
        return img, targets
    
def getNetImageSizeAndNumFeats(net, setEncToEval=False, verbose=False, image_size = 32, use_cuda = False, nc = 3):
    
    '''return two list: 
    - list of size of output (image) for each layer
    - list of size of total number of features (nFeatMaps*featmaps_height,featmaps_width) '''
    
    # classifier set to eval mode
    # if opt.setEncToEval is True, classifier is used when precomputing real dataset, and calculating generator.
    if setEncToEval:
        net.eval()

    if use_cuda:
        y, layers = net(Variable(torch.randn(1,nc,image_size,image_size).cuda()))
    else:
        y, layers = net(Variable(torch.randn(1,nc,image_size,image_size)))


    layer_img_size = []
    layer_num_feats = []
    for L in reversed(layers):
        if len(L.size()) == 4:
            layer_img_size.append(L.size(2))
            layer_num_feats.append(L.size(1)*L.size(2)*L.size(3))
        elif len(L.size()) == 3:    # vit
            print('L size', L.shape)
            layer_img_size.append(1)
            layer_num_feats.append(L.size(1)*L.size(2))

        elif len(L.size()) == 2:
            layer_img_size.append(1)
            layer_num_feats.append(L.size(1))
        elif len(L.size()) == 1:
            logging.info('only one layer')
            layer_num_feats.append(L.size(0))
        else:
            assert 0, 'not sure how to handle this layer size '+L.size()
            
    if verbose:
        logger.info("# Layer img sizes: {}".format(layer_img_size))
        logger.info("# Layer num feats: {}".format(layer_num_feats))

    return layer_img_size, layer_num_feats



def save_embedding(numpy_dict, PATH):
    '''
    save the numpy
    '''
    # If you don't have embedding file, just make it.
    if os.path.isfile(PATH):
        pass
    else:
        np.savez(PATH, **numpy_dict)    

def load_embedding(PATH):
    '''
    load the numpy
    '''
    try:
        if not os.path.isfile(PATH):
            raise
        dict_a = np.load(PATH, allow_pickle=True)
        return dict_a
    except FileNotFoundError:
        print('can not load file {}'.format(PATH))


def make_directories(opt):
    # Making Directories for saving grid-images and Generator models.
    if opt.debug:
        return 0
    try:
        if opt.outf == 'check_debugging':
            pass
        else:
            logger.info("Make : {}/{} directory\n".format(opt.outf, 'images') +
                    "Make : {}/{} directory\n".format(opt.outf, 'models'))
            os.makedirs(os.path.join(opt.saveroot, opt.outf, 'images'), exist_ok=True)
            os.makedirs(os.path.join(opt.saveroot, opt.outf, 'models'), exist_ok=True)
    except OSError:
        pass
    
def split_dataset(dataset, test_num, random_sample=False):
    """
    Params:
        dataset: Dataset
        test_num: int, the number of images for evaluation metrics
        random_sample: bool, If this param is true, you sample randomly.
    Returns:
        dataset: splitted Dataset
    """
    if random_sample:
        # Random sampling from the testset 5000
        indices = torch.randperm(len(dataset))[:test_num]
    else:
        indices = torch.arange(0, test_num)
    dataset = Subset(dataset, indices)
    return dataset

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image.clone().detach(), torch.tensor(label)

def _split_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        
    return dict_users

def _split_noniid(args, dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    if args.dataset == 'celeba':
        male_indices = [idx for idx in range(len(dataset)) if dataset.targets[idx][20]==1]
        female_indices = [idx for idx in range(len(dataset)) if dataset.targets[idx][20]==-1]
        female_indices = female_indices[:len(male_indices)]
        idxs = np.concatenate((male_indices, female_indices), axis=0)
        
        num_shards = 10
        num_imgs = len(idxs) // num_shards

        dict_users = {i: np.array([]) for i in range(10)}
        for i in range(5):
            dict_users[i] = np.concatenate(
                (dict_users[i], male_indices[i*num_imgs:(i+1)*num_imgs]), axis=0)
        
        i = 5
        for i in range(5):
            dict_users[i+5] = np.concatenate(
                (dict_users[i+5], female_indices[i*num_imgs:(i+1)*num_imgs]), axis=0)
    else:
        # 60,000 training imgs -->  200 imgs/shard X 300 shards
        num_shards = args.divide*num_users
        num_imgs = len(dataset) // num_shards
        idx_shard = [i for i in range(num_shards)]
        dict_users = {i: np.array([]) for i in range(num_users)}
        idxs = np.arange(num_shards*num_imgs)
        if args.dataset == "cifar":
            labels = np.array(dataset.targets)
        else:
            labels = dataset.train_labels.numpy()

        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        # divide and assign 2 shards/client
        for i in range(num_users):
            rand_set = set(np.random.choice(idx_shard, args.divide, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        
    return dict_users


def _get_transform(args):
    train_transform, test_transform = [], []
    
    if args.dataset == 'celeba':
        train_transform.extend([transforms.CenterCrop(160)])
        test_transform.extend([transforms.CenterCrop(160)])
    
    train_transform.extend([
        transforms.Resize((args.imageSize, args.imageSize), interpolation=transform_setting[args.netEncType]['interpolation'])])
    
    train_transform.extend([transforms.ToTensor()])
    
    train_transform.extend([transforms.Normalize((0.5,), (0.5,))])
    
    test_transform.extend([
        transforms.Resize(
            (args.imageSize, args.imageSize),
            interpolation=transform_setting[args.netEncType]['interpolation']),
        ToOnlyTensor()])
    
    return train_transform, test_transform

def MNIST(args, train_transform, test_transform):
    data_dir = '../data/mnist/'
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=transforms.Compose(train_transform))

    test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                    transform=transforms.Compose(test_transform))
    
    return train_dataset, test_dataset

def FMNIST(args, train_transform, test_transform):
    data_dir = '../data/fmnist/'
    
    train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                       transform=transforms.Compose(train_transform))

    test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                    transform=transforms.Compose(test_transform))
    
    return train_dataset, test_dataset

def CIFAR10(args, train_transform, test_transform):
    data_dir = '../data/cifar/'
    
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=transforms.Compose(train_transform))

    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                    transform=transforms.Compose(test_transform))
         
    return train_dataset, test_dataset

def CelebA(args, train_transform, test_transform):
    data_dir = 'CELEBA_PATH'
    
    train_dataset = CustomCelebADataset(data_dir, split='train', 
                                            transforms=transforms.Compose(train_transform))

    test_dataset = CustomCelebADataset(data_dir, split='test',
                                          transforms=transforms.Compose(test_transform))
    
    test_dataset = split_dataset(test_dataset, args.test_num, True)
    
    return train_dataset, test_dataset

dataset_function = {
    'mnist' : MNIST,
    'fmnist' : FMNIST,
    'celeba' : CelebA,
    'cifar' : CIFAR10,
}
    
        
def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        args.nc = 1
    else:
        args.nc = 3
        
    train_transform, test_transform = _get_transform(args)
    
    Dataset_fn = dataset_function[args.dataset]
    
    train_dataset, test_dataset = Dataset_fn(args, train_transform, test_transform)
    
    if args.iid:
        user_groups = _split_iid(train_dataset, args.num_users)
    else:
        user_groups = _split_noniid(args, train_dataset, args.num_users)
        
    trainset_list = []
    for idx in range(args.num_users):
        trainloader = DataLoader(DatasetSplit(train_dataset, user_groups[idx]),
                            batch_size=args.local_bs, shuffle=True)
        trainset_list.append(trainloader)
    
    return trainset_list, test_dataset


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        if 'scores' in key:
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))

    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    print(f'    Dataset : {args.dataset}')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

try:
    import accimage
except ImportError:
    accimage = None

#########################################
#   For Normalizing and Quantization Dataset, ToOnlyTensor is standarization. 
#########################################
class ToOnlyTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pic):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            Tensor
        """
        default_float_dtype = torch.get_default_dtype()

        if isinstance(pic, np.ndarray):
            # handle numpy array
            if pic.ndim == 2:
                pic = pic[:, :, None]

            img = torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()
            # backward compatibility
            if isinstance(img, torch.ByteTensor):
                return img.to(dtype=default_float_dtype)
            else:
                return img

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic).to(dtype=default_float_dtype)

        # handle PIL Image
        mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
        img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))

        if pic.mode == "1":
            img = 255 * img
        img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
        # put it from HWC to CHW format
        img = img.permute((2, 0, 1)).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.to(dtype=default_float_dtype)
        else:
            return img