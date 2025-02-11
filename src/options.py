#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=150,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=100,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64,
                        help="local batch size: B")
    parser.add_argument('--optimizer', type=str, default='adam', help='')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate for Generator')
    parser.add_argument('--lrMovAvrg', type=float, default=1e-5, help='learning rate for NetMean, NetVar')
    parser.add_argument('--beta1', type=float, default=0.5, help='')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    # avgM
    parser.add_argument('--aggregation', type=str, default='avgM', help='How to aggregate')
    parser.add_argument('--bias', type=bool, default=False, help='')
 
    # vgg19-pytorch | clip_resnet50
    parser.add_argument('--netEncType', type=str, default='vgg19-pytorch', help='')
    parser.add_argument('--numClassesInFtrExt', type=int, default='1000', help='Feature output number of classes')
    parser.add_argument('--setEncToEval', type=bool, default=False, help='')
    parser.add_argument('--ngf', type=int, default=64, help='Generator channel')
    parser.add_argument('--notAdaptFilterSize', type=bool, default=False, help='Does not use a different number of filters for each conv. layer [Resnet generator only].')
    parser.add_argument('--useConvAtGSkipConn', type=bool, default=False, help='For Resnet generator, applies a conv. layer to the input before upsampling in the skip connection')
    parser.add_argument('--numLayersToFtrMatching', type=int, default=16, help='Number of layers of the encoder/feature extractor used to perform feature matching')
    
    # K_Normal | Constant | pretrain | primitive
    parser.add_argument('--init', type=str, default='Constant', help='')
    parser.add_argument('--score_init', type=str, default="ME_init", help='')
    parser.add_argument('--scale_fan', type=bool, default=False, help='')
    parser.add_argument('--affine', type=bool, default=True, help='')
    parser.add_argument('--sparsity', type=float, default=0, help='')
    
    # Evaluation
    parser.add_argument('--test_num', type=int, default=10000, help='')
    parser.add_argument('--vgg_embedder_batch_norm', type=bool, default=True, help='if you load vgg16 model, the vgg16 model has batch norm')
    parser.add_argument('--embedder_backbone', type=str, default='inceptionV3', help='')
    parser.add_argument('--channel_multiplier', type=int, default=1, help='')
    parser.add_argument('--local_mask_corr', action='store_true', help='')
    
    
    # Differential-privacy
    parser.add_argument('--dp_epsilon', type=float, default=0) ### if use, 9.8
    parser.add_argument('--dp_clip', type=float, default=0.1)
    parser.add_argument('--dp_delta', type=float, default=1e-5)

    # other arguments
    parser.add_argument('--num_scorelayer', type=float, default=0, 
                        help='number of directly sending layer score 0 ~ 1 value.')
    parser.add_argument('--server_ema', action='store_true', help='')
    parser.add_argument('--ema_beta', type=float, default=0.5, help='')
    parser.add_argument('--MADA', action='store_true', help='')
    parser.add_argument('--dist_type', type=str, default='hd', help='')
    
    ## mnist | fmnist | celeba  | cifar
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--imageSize', type=int, default=32, help='')
    parser.add_argument('--batchSize', type=int, default=64, help='')
    parser.add_argument('--gpunum', default=0)
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--split', type=str, default="shards", 
                        help='shards | dirichlet')
    parser.add_argument('--dir_alpha', type=float, default=0.005, help='')
    parser.add_argument('--divide', type=int, default=4,
                        help='The set of partition for non-iid. ex) if set to 4, dataset will be 4-class non-iid')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--nz', type=int, default=128, help='size of the noise')
    parser.add_argument('--nc', type=int, default=3, help='channel of the image')
    parser.add_argument('--seed', type=int, default=30, help='random seed')
    parser.add_argument('--save', type=bool, default=False, help='If true, save imgs and plots')
    parser.add_argument('--evalIter', type=int, default=10, help='Evaluation interval')
    parser.add_argument('--experiments', type=str, default='entropy_mask_test', help='experiment name for save path')
    args = parser.parse_args()
    return args