import torch
from torch import nn
import numpy as np
import torch.optim as optim
import os
import torchvision.utils as vutils
import os
import copy
import matplotlib.pyplot as plt

from models.prism import ResnetG
from scipy.stats import bernoulli
from utils.utils import compute_distance, track_bn_statistics, untrack_bn_statistics
from utils.sam_opt import SAM

class Client(object):
    def __init__(self, args, classifier, dataset, client_idx, device, noise_multiplier):
        self.args = args
        self.Clf = classifier
        self.client_idx = client_idx

        self.trainloader = dataset

        self.device = device
        self.Clf.to(self.device)

        # Default criterion set to NLL loss function
        self.criterionL2Loss = nn.MSELoss()

        # get ImageNetstat for Preprocessing        
        self.imageNetNormMinV, self.imageNetNormRangeV  = self.get_imageNet_stat()
            
        self.imageNetNormMinV = self.imageNetNormMinV.to(self.device)
        self.imageNetNormRangeV = self.imageNetNormRangeV.to(self.device)
        # get Number of Features from Classifier
        self.numFeaturesInEnc, self.numFeaturesForEachSelectedLayerm, self.numLayersToFtrMatching = self.get_numFeaturesInEnc(self.args.numLayersToFtrMatching)
        self.__build_NetMeanVar(self.numFeaturesInEnc)
        
        self.noise_multiplier = noise_multiplier
        

        # get models and optimizers
        self.gfmn = ResnetG(args, args.nz, args.nc, args.ngf,
                            args.imageSize, adaptFilterSize=not args.notAdaptFilterSize,
                    useConvAtSkipConn=args.useConvAtGSkipConn).to(self.device)

        self.freeze_model_weights(self.gfmn)
        self.unfreeze_model_subnet(self.gfmn)
                    
        self.__getOptimizers()

        self.epsilon = 0.01
        
        self.layer = []    
        for k, v in self.gfmn.state_dict().items():
            if 'scores' in k and 'bias' not in k:
                self.layer.append(k)
                    
        if self.args.num_scorelayer > 0:
            score_proportion = int(len(self.layer)*self.args.num_scorelayer)
            self.score_layer = self.layer[-1 * score_proportion:]
            self.mask_layer = self.layer[:-1 * score_proportion]
        else:
            self.mask_layer = self.layer
            self.score_layer = []

    def freeze_model_weights(self, model):

        # print("=> Freezing model weights")
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
        # print("=> Unfreezing model subnet")

        for n, m in model.named_modules():
            if hasattr(m, "scores"):
                # print(f"==> Gradient to {n}.scores")
                m.scores.requires_grad = True

    def get_imageNet_stat(self):
        # Preprocessing for ImageNet
        imageNetNormMean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        imageNetNormStd = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        imageNetNormMin = -imageNetNormMean / imageNetNormStd
        imageNetNormMax = (1.0 - imageNetNormMean) / imageNetNormStd
        imageNetNormRange = imageNetNormMax - imageNetNormMin

        # Variables used to renormalize the data to ImageNet scale.
        imageNetNormMinV = torch.FloatTensor(imageNetNormMin)
        imageNetNormRangeV = torch.FloatTensor(imageNetNormRange)
        imageNetNormMinV.resize_(1, 3, 1, 1)
        imageNetNormRangeV.resize_(1, 3, 1, 1)

        return imageNetNormMinV, imageNetNormRangeV

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

    # ------------------------------------------
    #   Building Net Mean and Variance for Adam Moving Average
    # ------------------------------------------
    def __build_NetMeanVar(self, numFeaturesInEnc):
        # creates Networks for moving Mean and Variance.
        self.NetMean = nn.Linear(numFeaturesInEnc, 1, bias=False)
        self.NetVar = nn.Linear(numFeaturesInEnc, 1, bias=False)

        self.NetMean.to(self.device)
        self.NetVar.to(self.device)

    def __getOptimizers(self):
        parametersG = set()
        parametersG |= set(self.gfmn.parameters())
        if self.args.use_sam:
            self.optimizerG = SAM(params=parametersG,
                                  base_optimizer=torch.optim.SGD,
                                  lr=self.args.lr,
                                  rho=self.args.rho)
        else:
            self.optimizerG = optim.Adam(parametersG, lr=self.args.lr, betas=(self.args.beta1, 0.999))
        self.optimizerMean = optim.Adam(self.NetMean.parameters(
        ), lr=self.args.lrMovAvrg, betas=(self.args.beta1, 0.999))
        self.optimizerVar = optim.Adam(
            self.NetVar.parameters(), lr=self.args.lrMovAvrg, betas=(self.args.beta1, 0.999))                            

    def compute_real_features(self):
        numExamplesProcessed = 0.0
        globalFtrMeanValues = []
        for i, data in enumerate(self.trainloader):
            # gets real images
            real_cpu, _ = data      # img, target

            real_cpu = real_cpu.to(self.device)

            # convert gray scale image into color image
            if real_cpu.shape[1]==1:
                real_cpu = real_cpu.expand(
                    real_cpu.shape[0], 3, real_cpu.shape[-1], real_cpu.shape[-1])

            numExamplesProcessed += real_cpu.size()[0]

            # extracts features for TRUE data
            allFtrsTrue = self.extractFeatures(real_cpu, detachOutput=True)


            if len(globalFtrMeanValues) < 1:
                globalFtrMeanValues = torch.sum(allFtrsTrue, dim=0).detach()
                featureSqrdValues = torch.sum(allFtrsTrue** 2, dim=0).detach()
            else:
                globalFtrMeanValues += torch.sum(allFtrsTrue, dim=0).detach()
                featureSqrdValues += torch.sum(allFtrsTrue ** 2, dim=0).detach()

        return numExamplesProcessed, globalFtrMeanValues, featureSqrdValues

    def precomputed_real_datas(self):
        # Computing Real Dataset
        numExamplesProcessed, globalFtrMeanValues, featureSqrdValues = self.compute_real_features()

        # variance = (SumSq - (Sum x Sum) / n) / (n - 1)
        globalFtrVarValues = (featureSqrdValues - (globalFtrMeanValues ** 2) / numExamplesProcessed) / (
            numExamplesProcessed - 1)

        globalFtrMeanValues = globalFtrMeanValues / numExamplesProcessed

        return globalFtrMeanValues, globalFtrVarValues

    def preprocess_fakeData(self, fakeData, imageNetNormRangeV, imageNetNormMinV):
        # normalize part
        if fakeData.shape[1] == 1:
            fakeData = fakeData.expand(fakeData.shape[0], 3, fakeData.shape[-1], fakeData.shape[-1])

        elif fakeData.shape[1] == 3:
            # normalizes the generated images using imagenet min-max ranges
            # newValue = (((fakeData - OldMin) * NewRange) / OldRange) + NewMin
            fakeData = (((fakeData + 1) * imageNetNormRangeV) / 2) + imageNetNormMinV
        return fakeData

    def extractFeatures(self, batchOfData, detachOutput=False):
        """
        Params:
            batchofData: Input Batch가 들어온다.
            curNetEnc: classifier 하나의 모델이 들어온다.
            detachOutput: loss 계산시 무시를 하고 싶다면 detachOutput=True로 바꿔준다.
        Returns:
            ftrs: [ N (BatchSize) * All of Feature Num] i.e. 64*1,212,416 개가 output으로 나온다.
        Applies feature extractor. Concatenate feature vectors from all selected layers.
        """
        # gets features from each layer of netEnc
        ftrs = []
        ftrsPerLayer = self.Clf(batchOfData)[1]

        for lId in range(1, self.numLayersToFtrMatching + 1):
            cLid = lId - 1  # gets features in forward order

            # vit == .contiguous() 추가
            ftrsOfLayer = ftrsPerLayer[cLid].contiguous().view(
                ftrsPerLayer[cLid].size()[0], -1)

            if detachOutput:
                ftrs.append(ftrsOfLayer.detach())
            else:
                ftrs.append(ftrsOfLayer)
        ftrs = torch.cat(ftrs, dim=1)
        return ftrs

    def Load_precomputed_realData(self):
        if self.args.iid == 1:
            precompute_path=f'./embeddings/precompute/enc-{self.args.netEncType}_{self.args.dataset}-{self.args.imageSize}_{self.args.iid}_{self.args.num_users}'
        else:
            precompute_path=f'./embeddings/precompute/enc-{self.args.netEncType}_{self.args.dataset}-{self.args.imageSize}_{self.args.iid}_{self.args.divide}_{self.args.num_users}'
        
        if os.path.isfile(os.path.join(precompute_path,f'mean_client{self.client_idx}.npy')):
            print("Loading Precomputed values")
            mean_load = np.load(os.path.join(precompute_path,f'mean_client{self.client_idx}.npy'))
            var_load = np.load(os.path.join(precompute_path,f'var_client{self.client_idx}.npy'))
            globalFtrMeanValues = torch.from_numpy(mean_load).to(self.device)
            globalFtrVarValues = torch.from_numpy(var_load).to(self.device)
        else:
            print("Thers is no precomputed_file. Precomputing and saving..")
            globalFtrMeanValues, globalFtrVarValues = self.precomputed_real_datas()
            globalFtrMeanValues_npy = globalFtrMeanValues.detach().cpu().numpy()
            globalFtrVarValues_npy = globalFtrVarValues.detach().cpu().numpy()

            if not os.path.isdir(precompute_path):
                os.mkdir(precompute_path)
            np.save(os.path.join(precompute_path,f'mean_client{self.client_idx}.npy'), globalFtrMeanValues_npy)
            np.save(os.path.join(precompute_path,f'var_client{self.client_idx}.npy'), globalFtrVarValues_npy)

        
        return globalFtrMeanValues, globalFtrVarValues

    def local_train(self, epoch, optimizerMean, optimizerVar):
        self.gfmn.to(self.device)
        self.gfmn.train()

        batch_loss = []
        
        if self.args.local_mask_corr:
            global_mask = copy.deepcopy(self.gfmn.state_dict())
            local_mask_corr = []

        # Precomputing Real Datasets
        globalFtrMeanValues, globalFtrVarValues = self.Load_precomputed_realData()
        
        # initialization
        lossNetG = 0

        avrgLossNetGMean = 0.0
        avrgLossNetGVar = 0.0
        avrgLossNetMean = 0.0
        avrgLossNetVar = 0.0

        for iter in range(self.args.local_ep):
            # Setting zero_Grad()
            self.Clf.zero_grad()
            self.gfmn.zero_grad()
            self.NetMean.zero_grad()
            self.NetVar.zero_grad()
            if self.args.use_sam:
                self.gfmn.apply(track_bn_statistics)

            # creates noise and fake Datas
            noise = torch.randn(self.args.local_bs, self.args.nz, 1, 1)
            noise = noise.to(self.device)
            fakeData = self.gfmn(noise)
            
            fakeData = self.preprocess_fakeData(fakeData, self.imageNetNormRangeV, self.imageNetNormMinV)
            ftrsFakeData= self.extractFeatures(fakeData, detachOutput=False)
            
            
            # uses Adam moving average
            # updates moving average of mean differences
            ftrsMeanFakeData = torch.mean(ftrsFakeData, 0)
            diffFtrMeanTrueFake = globalFtrMeanValues.detach() - ftrsMeanFakeData.detach()
            lossNetMean = self.criterionL2Loss(
                self.NetMean.weight, diffFtrMeanTrueFake.detach().view(1, -1))
            lossNetMean.backward()
            avrgLossNetMean += lossNetMean.item()
            self.optimizerMean.step()

            # updates moving average of variance differences
            ftrsVarFakeData = torch.var(ftrsFakeData, 0)
            diffFtrVarTrueFake = globalFtrVarValues.detach() - ftrsVarFakeData.detach()
            lossNetVar = self.criterionL2Loss(
                self.NetVar.weight, diffFtrVarTrueFake.detach().view(1, -1))
            lossNetVar.backward()
            avrgLossNetVar += lossNetVar.item()
            self.optimizerVar.step()

            # updates generator
            meanDiffXTrueMean = self.NetMean(globalFtrMeanValues.view(1, -1)).detach()
            meanDiffXFakeMean = self.NetMean(ftrsMeanFakeData.view(1, -1))

            varDiffXTrueVar = self.NetVar(globalFtrVarValues.view(1, -1)).detach()
            varDiffXFakeVar = self.NetVar(ftrsVarFakeData.view(1, -1))

            lossNetGMean = (meanDiffXTrueMean - meanDiffXFakeMean)
            avrgLossNetGMean += lossNetGMean.item()

            lossNetGVar = (varDiffXTrueVar - varDiffXFakeVar)
            avrgLossNetGVar += lossNetGVar.item()

            # compute loss generator
            lossNetG = lossNetGMean + lossNetGVar
            lossNetG.backward()
            if self.args.use_sam:
                self.optimizerG.first_step(zero_grad=True)
                self.gfmn.apply(untrack_bn_statistics)
                
                # creates noise and fake Datas
                noise = torch.randn(self.args.local_bs, self.args.nz, 1, 1)
                noise = noise.to(self.device)
                fakeData = self.gfmn(noise)
                
                fakeData = self.preprocess_fakeData(fakeData, self.imageNetNormRangeV, self.imageNetNormMinV)
                ftrsFakeData= self.extractFeatures(fakeData, detachOutput=False)
                
                ftrsMeanFakeData = torch.mean(ftrsFakeData, 0)
                ftrsVarFakeData = torch.var(ftrsFakeData, 0)
                
                meanDiffXTrueMean = self.NetMean(globalFtrMeanValues.view(1, -1)).detach()
                meanDiffXFakeMean = self.NetMean(ftrsMeanFakeData.view(1, -1))

                varDiffXTrueVar = self.NetVar(globalFtrVarValues.view(1, -1)).detach()
                varDiffXFakeVar = self.NetVar(ftrsVarFakeData.view(1, -1))

                lossNetGMean = (meanDiffXTrueMean - meanDiffXFakeMean)
                lossNetGVar = (varDiffXTrueVar - varDiffXFakeVar)
            
                lossNetG = lossNetGMean + lossNetGVar
                lossNetG.backward()
                self.optimizerG.second_step(zero_grad=True)
            else:
                self.optimizerG.step()
            
            # Visualization Batch Loss
            batch_loss.append(lossNetG.item())

            record_iter = 99

            if iter % (self.args.local_ep - 1) == 0:
                if self.args.save and epoch % self.args.evalIter == 0:
                    img_path = f'results/{self.args.experiments}/clients'

                    if self.client_idx == 0:
                        vutils.save_image(fakeData[:],
                                            f'{img_path}/local_samples_{epoch}.png', normalize=True)
    
                print(f"Epoch : {epoch} Client [{self.client_idx}] Locel Epoch {iter} / {self.args.local_ep}, Loss_Gz: %.6f Loss_GzVar: %.6f Loss_vMean: %.6f Loss_vVar: %.6f" %
                    (avrgLossNetGMean / record_iter, avrgLossNetGVar / record_iter,
                    avrgLossNetMean / record_iter, avrgLossNetVar / record_iter))
                

                avrgLossNetGMean = 0.0
                avrgLossNetGVar = 0.0
                avrgLossNetMean = 0.0
                avrgLossNetVar = 0.0
            
            if self.args.local_mask_corr and iter == (self.args.local_ep - 1) and epoch % self.args.evalIter == 0:
                server_mask = dict()
                local_mask = dict()
                with torch.no_grad():
                    for k, v in global_mask.items():
                        if 'scores' in k:
                            target = v.to(self.device)
                            theta = torch.sigmoid(target)
                            
                            updates_s = bernoulli.rvs(theta.cpu().numpy())
                            if self.args.dataset == 'mnist' and 'net.lastConv.scores' in k:
                                updates_s = np.expand_dims(updates_s, axis=0)
                            if server_mask.get(k) is None:
                                server_mask[k] = torch.tensor(updates_s, device=self.device)
                            else:
                                server_mask[k] += torch.tensor(updates_s, device=self.device)
                        else:
                            if server_mask.get(k) is None:
                                server_mask[k] = torch.tensor(v, device=self.device)
                            else:
                                server_mask[k] += torch.tensor(v, device=self.device)
                                
                with torch.no_grad():
                    for k, v in self.gfmn.state_dict().items():
                        if 'scores' in k:
                            target = v.to(self.device)
                            theta = torch.sigmoid(target)
                            
                            updates_s = bernoulli.rvs(theta.cpu().numpy())
                            if self.args.dataset == 'mnist' and 'net.lastConv.scores' in k:
                                updates_s = np.expand_dims(updates_s, axis=0)
                            if local_mask.get(k) is None:
                                local_mask[k] = torch.tensor(updates_s, device=self.device)
                            else:
                                local_mask[k] += torch.tensor(updates_s, device=self.device)
                        else:
                            if local_mask.get(k) is None:
                                local_mask[k] = torch.tensor(v, device=self.device)
                            else:
                                local_mask[k] += torch.tensor(v, device=self.device)
                            
                hamming_distances = compute_distance(server_mask, local_mask)
                local_mask_corr.append(hamming_distances)
                epochs = list(range(len(local_mask_corr)))
                
                # 결과 출력
                print(f"Round {epoch} Hamming Distance: {hamming_distances}")
                
        return self.gfmn.state_dict(), self.NetMean.state_dict(), self.NetVar.state_dict(), self.optimizerMean.state_dict(), self.optimizerVar.state_dict(), batch_loss, lossNetG

    def upload_mask(self):
        ### Mask upload to the server
        param_dict = dict()

        with torch.no_grad():
            for k, v in self.gfmn.state_dict().items():
                if 'scores' in k:
                    if self.noise_multiplier != 0:
                        dp_sensitivity = (1 - 2 * self.args.dp_clip) * np.sqrt(v.numel())
                        noise_multiplier = np.sqrt(2 * np.log(1.25 / self.args.dp_delta) / self.args.dp_epsilon)
                        sigma = np.sqrt(noise_multiplier * dp_sensitivity)
                        
                        dp_noise = torch.normal(mean=0.,
                                            std=sigma, size=v.shape, device=self.device)
                    else:
                        dp_noise = torch.zeros(v.shape, device=self.device)
                        
                    target = v.to(self.device)
                    theta = torch.sigmoid(target)
                    theta += dp_noise
                    
                    if self.noise_multiplier != 0:
                        theta = torch.clamp(target, self.args.dp_clip, 1-self.args.dp_clip)
                    
                    if k in self.mask_layer:
                        updates_s = bernoulli.rvs(theta.cpu().numpy())
                        updates_s = np.where(updates_s == 0, self.epsilon, updates_s)
                        updates_s = np.where(updates_s == 1, 1-self.epsilon, updates_s)
                        if self.args.dataset == 'mnist' and 'net.lastConv.scores' in k:
                            updates_s = np.expand_dims(updates_s, axis=0)
                        if param_dict.get(k) is None:
                            param_dict[k] = torch.tensor(updates_s, device=self.device)
                        else:
                            param_dict[k] += torch.tensor(updates_s, device=self.device)
                    elif k in self.score_layer:
                        if self.noise_multiplier != 0:
                            if param_dict.get(k) is None:
                                param_dict[k] = torch.tensor(theta, device=self.device)
                            else:
                                param_dict[k] += torch.tensor(theta, device=self.device)
                        else:
                            if param_dict.get(k) is None:
                                param_dict[k] = torch.tensor(v, device=self.device)
                            else:
                                param_dict[k] += torch.tensor(v, device=self.device)

                else:
                    if param_dict.get(k) is None:
                        param_dict[k] = torch.tensor(v, device=self.device)
                    else:
                        param_dict[k] += torch.tensor(v, device=self.device)


        return param_dict