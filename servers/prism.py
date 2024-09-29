import torch
from torch import nn
import numpy as np
import torch.optim as optim
import os
from metrics.evaluate import get_embedding_real, get_embedding_fake
from metrics.prdc import iprdc
from metrics import fid_score
import os
from models.prism import ResnetG
import copy
from metrics.inception_score import inception_score
import torchvision.utils as vutils

from ema_pytorch import EMA

from torch.utils.data import DataLoader
from utils.utils import _get_transform, dataset_function
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from models.model_utils import dynamic_EMA
from utils.utils import compare_mask, compare_global_mask

class Server(object):
    def __init__(self, args, classifier, client_list, test_dataloader, device):
        self.args = args
        self.client_list = client_list
        self.client_loss = [[] for i in range(self.args.num_users)]

        self.Clf = classifier
        self.test_dataloader = test_dataloader

        self.device = device
        
        self.best_FID = 9999

        self.numFeaturesInEnc, _, _ = self.get_numFeaturesInEnc(self.args.numLayersToFtrMatching)
        self.NetMean = nn.Linear(self.numFeaturesInEnc, 1, bias=False)
        self.NetVar = nn.Linear(self.numFeaturesInEnc, 1, bias=False)
        
        self.NetMean.to(self.device)
        self.NetVar.to(self.device)
        self.optimizerMean = optim.Adam(self.NetMean.parameters(
        ), lr=self.args.lrMovAvrg, betas=(self.args.beta1, 0.999))
        self.optimizerVar = optim.Adam(
            self.NetVar.parameters(), lr=self.args.lrMovAvrg, betas=(self.args.beta1, 0.999))
        
        self.client_entropy_list = [[] for i in range(self.args.num_users)]

        # Set global_model
        self.global_model = ResnetG(self.args, args.nz, args.nc, args.ngf
                                    , args.imageSize, adaptFilterSize=not args.notAdaptFilterSize,
                    useConvAtSkipConn=args.useConvAtGSkipConn).to(self.device)
        
        if self.args.server_ema:
                self.ema = EMA(
                    self.global_model,
                    beta=self.args.ema_beta,
                    update_after_step=1,
                    update_every=1,
                    )
    
        self.layer = []    
        
        for k, v in self.global_model.state_dict().items():
            if 'scores' in k and 'bias' not in k:
                self.layer.append(k)
        
        if self.args.num_scorelayer > 0:
            score_proportion = int(len(self.layer)*self.args.num_scorelayer)
            self.score_layer = self.layer[-1 * score_proportion:]
            self.mask_layer = self.layer[:-1 * score_proportion]
                

        self.freeze_model_weights(self.global_model)
        self.unfreeze_model_subnet(self.global_model)
        
        self.optimizerG = optim.Adam(self.global_model.parameters(), lr=self.args.lr, betas=(self.args.beta1, 0.999))
        
        self.globalmask_corr = []
                
    def freeze_model_weights(self, model):

        for n, m in model.named_modules():
            if hasattr(m, "weight") and m.weight is not None:
                # print(f"==> No gradient to {n}.weight")
                m.weight.requires_grad = False
                if m.weight.grad is not None:
                    # print(f"==> Setting gradient of {n}.weight to None")
                    m.weight.grad = None

                if hasattr(m, "bias") and m.bias is not None:
                    # print(f"==> No gradient to {n}.bias")
                    m.bias.requires_grad = False

                    if m.bias.grad is not None:
                        # print(f"==> Setting gradient of {n}.bias to None")
                        m.bias.grad = None

    def unfreeze_model_subnet(self, model):

        for n, m in model.named_modules():
            if hasattr(m, "scores"):
                # print(f"==> Gradient to {n}.scores")
                m.scores.requires_grad = True

    def get_numFeaturesInEnc(self, numLayersToFtrMatching):

        """
        Params:
            opt: option parser
            curNetEnc: current Classifier network
        Returns:
            numFeaturesInEnc: Encoder에서 계산되는 Feature의 개수
            numFeaturesForEachSelectedLayer list: 한 개의 데이터가 feature matching되는 feature의 개수를 의미
            numLayersToFtrMatching int : Feature Matching 하려는 레이어 개수
        """
        numFeaturesInEnc = 0
        # computes the total number of features
        # number of features output for each layer, from the last to the first layer
        numFeaturesForEachEncLayer = self.Clf.numberOfFeaturesPerLayer

        numLayersToFtrMatching = min(
            numLayersToFtrMatching, len(numFeaturesForEachEncLayer))

        numFeaturesInEnc += sum(
            numFeaturesForEachEncLayer[-numLayersToFtrMatching:])
        numFeaturesForEachSelectedLayer = numFeaturesForEachEncLayer[-numLayersToFtrMatching:]

        # orders from last to first layer
        # numFeaturesForEachSelectedLayer의 순서를 바꾼다.
        numFeaturesForEachSelectedLayer = numFeaturesForEachSelectedLayer[::-1]

        return numFeaturesInEnc, numFeaturesForEachSelectedLayer, numLayersToFtrMatching

    def train(self, epoch):
        self.global_model.train()

        local_weights = []

        if self.args.frac < 1:
            m = max(int(self.args.frac * self.args.num_users), 1)
            idxs_users = np.random.choice(range(self.args.num_users), m, replace=False)
        else:
            idxs_users = range(self.args.num_users)

        global_loss = 0

        for idx in idxs_users:
            
            self.client_list[idx].gfmn.load_state_dict(self.global_model.state_dict())

            local_model, NetMean_w, NetVar_w, optimizerMean_w, optimizerVar_w, local_loss, lossNetG = self.client_list[idx].local_train(epoch, self.optimizerMean.state_dict(), self.optimizerVar.state_dict())

            local_weights.append(copy.deepcopy(local_model))
            self.client_loss[idx].extend(local_loss)

            global_loss += lossNetG

        
        if self.args.num_scorelayer > 0: 
            # PRISM-alpha
            print("Aggregation method : PRISM-alpha")
            global_weights = self.average_scrores_alpha(local_weights)
            self.global_model.load_state_dict(global_weights)
            global_weights = self.average_masks_alpha(epoch, idxs_users)
        else:
            # PRISM
            print("Aggregation method : PRISM")
            global_weights = self.average_masks(epoch, idxs_users)
        
        if self.args.global_mask_corr:
            ema_coef = compare_global_mask(self.args, self.global_model, global_weights, self.globalmask_corr, epoch, self.device)
            
        if self.args.server_ema:
            print(f"Global Model EMA Update and EMA beta is {self.args.ema_beta}")
            # self.ema.update()
            # self.global_model.load_state_dict(self.ema.ema_model.state_dict())
            ema_param = dynamic_EMA(self.args, global_weights,
                                    self.global_model.state_dict(), beta=self.args.ema_beta)
            self.global_model.load_state_dict(ema_param)
        elif self.args.dynamic_ema:
            print("Dynamic Global Model EMA Update")
            ema_coef = compare_global_mask(self.args, self.global_model, global_weights, self.globalmask_corr, epoch, self.device)
            ema_param = dynamic_EMA(self.args, global_weights,
                                    self.global_model.state_dict(), beta=ema_coef)
            self.global_model.load_state_dict(ema_param)
        else:
            # update global weights
            self.global_model.load_state_dict(global_weights)
            
        if self.args.mask_corr:
            compare_mask(self.args, self.global_model, self.client_list, epoch, self.device)

        return self.client_loss, global_loss

    def average_scrores_alpha(self, w):
        """`
        Returns the average of the weights.
        """
        w_avg = copy.deepcopy(self.global_model.state_dict())

        for key in w_avg.keys():
            if key in self.score_layer:
                print(f"Score communication : {key}")
                w_avg[key] = torch.zeros_like(w_avg[key])
                for i in range(0, len(w)):
                    w_avg[key] += w[i][key].to(self.device)
                w_avg[key] = torch.div(w_avg[key], len(w))

        return w_avg
        
    def average_masks(self, epoch, idxs_users):
        aggregated_weight = copy.deepcopy(self.global_model.state_dict())
        aggregate_dict = dict()


        for k, v in self.global_model.state_dict().items():
            if 'scores' in k:
                aggregate_dict[k] = torch.zeros_like(v)
            # aggregate_dict[k] = torch.zeros_like(v, dtype=torch.float32)

        with torch.no_grad():
            ## Mask aggregation
            for client_idx in idxs_users:
                sampled_mask = self.client_list[client_idx].upload_mask()
                for k, v in sampled_mask.items():
                    if 'scores' in k:
                        aggregate_dict[k] += v / len(idxs_users)
                    # aggregate_dict[k] += torch.div(v, len(idxs_users))

            ## Theta to score
            for k, v in aggregate_dict.items():
                if 'scores' in k:
                    aggregated_weight[k] = torch.tensor(torch.log(v / (1 - v)),
                                                     requires_grad=True, device=self.device)

        return aggregated_weight
    
    def average_masks_alpha(self, epoch, idxs_users):
        aggregated_weight = copy.deepcopy(self.global_model.state_dict())
        aggregate_dict = dict()


        for k, v in self.global_model.state_dict().items():
            if 'scores' in k:
                aggregate_dict[k] = torch.zeros_like(v)

        with torch.no_grad():
            ## Mask aggregation
            for client_idx in idxs_users:
                sampled_mask = self.client_list[client_idx].upload_mask()
                for k, v in sampled_mask.items():
                    if 'scores' in k and k in self.mask_layer:
                        print(k)
                        aggregate_dict[k] += v / len(idxs_users)

            ## Theta to score
            if self.args.sparsity == 0:
                for k, v in aggregate_dict.items():
                    if 'scores' in k and k in self.mask_layer:
                        aggregated_weight[k] = torch.tensor(torch.log(v / (1 - v)),
                                                        requires_grad=True, device=self.device)

        return aggregated_weight

    def evaluate(self, epoch):            
        ####################
        #    Evaluation    #
        ####################
        noise = torch.randn(self.args.batchSize, self.args.nz, 1, 1).to(self.device)
        test_imgs = self.global_model(noise)
        
        fid_dict_path = os.path.join('embeddings', 'npz', self.args.dataset + str(self.args.test_num) + '_' +  str(self.args.imageSize) + '_' + str(self.args.embedder_backbone) + '_test.npz')
        if not os.path.isfile(fid_dict_path):
            load_embedding_flag=False
        else:
            load_embedding_flag=True
            
        load_embedding_flag = False
        
        IS, _ = inception_score(test_imgs, self.device)


        real_pred = get_embedding_real(self.args, self.test_dataloader, load_embedding_flag, fid_dict_path, self.device)
        fake_pred = get_embedding_fake(self.args, self.global_model, self.device)

        for embedder in real_pred.keys():
            if load_embedding_flag:
                real_mu = real_pred[embedder][()]['mu']
                real_sigma = real_pred[embedder][()]['sigma']
            else:
                real_mu = real_pred[embedder]['mu']
                real_sigma = real_pred[embedder]['sigma']

            fake_mu = fake_pred[embedder]['mu']
            fake_sigma = fake_pred[embedder]['sigma']

            fid_val = fid_score.get_fid((real_mu, real_sigma), (fake_mu, fake_sigma))

            #prdc
            r_pred = real_pred[embedder]['pred']
            f_pred = fake_pred[embedder]['pred']
            prdc_value = iprdc(r_pred, f_pred)

            if self.best_FID > fid_val:
                self.best_FID = fid_val
                if self.args.save:
                    torch.save(self.global_model.state_dict(), f'./results/{self.args.experiments}/best_model.pth')

            if self.args.dataset != 'mnist':
                print(f"Epoch : {epoch}/{self.args.epochs} IS : {IS} FID : {fid_val} best_FID : {self.best_FID} Precision : {prdc_value['precision']} Recall : {prdc_value['recall']} Density : {prdc_value['density']} Coverage : {prdc_value['coverage']}")
            else:
                print(f"Epoch : {epoch}/{self.args.epochs} FID : {fid_val} best_FID : {self.best_FID} Precision : {prdc_value['precision']} Recall : {prdc_value['recall']} Density : {prdc_value['density']} Coverage : {prdc_value['coverage']}")

        return IS, fid_val, prdc_value, test_imgs