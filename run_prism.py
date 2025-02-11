import os
import torch
torch.set_num_threads(4)
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=2
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=2
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=2
import time
import numpy as np
import random

from src.options import args_parser
from utils.utils import get_dataset, exp_details
from utils.logging import LoggerSetting
from models.load_classifier import getClassifier, check_requires_grad
import torchvision.utils as vutils
import matplotlib
matplotlib._log.disabled = True
import wandb

import json


from dp.rdp import RDPAccountant
from dp.utils import *
from dp.analysis import rdp as privacy_analysis

def init():
    start_time = time.time()
    ### define log and paths
    path_project = os.path.abspath('..')
    log = LoggerSetting()
    args = args_parser()
    exp_details(args)
    
    ### Wandb
    if args.save:
        wandb.init(project='PROJECT',
                    name = args.experiments,
                    entity='ENTITY')
        wandb.config.update(args)

    ### Fix random seed
    random.seed(args.seed) # random
    np.random.seed(args.seed) # numpy
    os.environ["PYTHONHASHSEED"] = str(args.seed) # os
    # pytorch
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed) 
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
        
    device = f'cuda:{args.gpunum}'
    
    if args.save:
        img_path = f'results/{args.experiments}'
        if not os.path.exists(img_path):
            os.mkdir(img_path)
            os.mkdir(os.path.join(img_path, f'clients'))
    
    return start_time, log, args, path_project, device
               

def build_Server_Client(args, trainset_list, test_dataloader, classifier):
    from clients.prism import Client
    from servers.prism import Server
    args.lr = 0.1
    
    ### Create clients and server
    client_list = []
    for client_idx in range(args.num_users):
        client = Client(args, classifier, trainset_list[client_idx], client_idx, device)
        client_list.append(client)
    server = Server(args, classifier, client_list, test_dataloader, device)
    
    return server, client_list

def record(args, epoch, client_loss, global_loss, eval_dict, IS, fid, prdc_value, test_imgs):
    ######################
    #### Save Plot #######
    ######################
    Epoch=range(1,((epoch+1)*args.local_ep)+1)
    plot_path=f'results/{args.experiments}'

    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    
    #######################
    ####  Save Image ######
    #######################
    img_path = f'results/{args.experiments}'
    if not os.path.isdir(img_path):
        os.mkdir(img_path)
    
    #######################
    #### Wandb Logging ####
    #######################
    vutils.save_image(test_imgs[:],
            os.path.join(img_path,f'fake_samples_{epoch+1}.png'), normalize=True)
    
    wandb.log({'Epoch' : epoch,
            'Inception Score' : IS,
            'FID' : fid,
            'Precision' : prdc_value['precision'],
            'Recall' : prdc_value['recall'],
            'Density' : prdc_value['density'],
            'Coverage' : prdc_value['coverage']})
        
    wandb.log({"Fake samples": [wandb.Image('%s/fake_samples_%d.png' %
                            (img_path, epoch+1))]}, commit=True)
    
if __name__ == '__main__':
    
    start_time, log, args, path_project, device = init()

    # Load dataset        
    trainset_list, test_dataset = get_dataset(args)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchSize,
                                                 shuffle=True, num_workers=4)
    
    # Build Classifier
    classifier = getClassifier(args, log)
    
    # Build Server and Client
    server, client_list = build_Server_Client(args, trainset_list, test_dataloader, classifier)
    
    eval_dict = {
        'epoch' : [],
        'is' : [],
        'fid' : [],
        'precision' : [],
        'recall' : [],
        'density' : [],
        'coverage' : [],
    }

    for epoch in range(args.epochs):
        client_loss, global_loss = server.train(epoch)
        
        if epoch % args.evalIter == 0:
            # server.global_model.to(server.device)
            IS, fid, prdc_value, test_imgs = server.evaluate(epoch)
            # server.global_model.cpu()
            eval_dict['epoch'].append(epoch)
            eval_dict['is'].append(IS)
            eval_dict['fid'].append(fid)
            eval_dict['precision'].append(prdc_value['precision'])
            eval_dict['recall'].append(prdc_value['recall'])
            eval_dict['density'].append(prdc_value['density'])
            eval_dict['coverage'].append(prdc_value['coverage'])
            
            if args.save:
                with open(f'./results/{args.experiments}/eval_dict.json', 'w') as f:
                    json.dump(eval_dict, f, indent=4)
                record(args, epoch, client_loss, global_loss, eval_dict, IS, fid, prdc_value, test_imgs)
        print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
    
    exp_details(args)
    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
    