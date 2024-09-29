
import torch
import torch.nn.functional as F
import metrics.inception as inception
import metrics.inception_tensorflow as inception_tensorflow
from metrics.vgg16 import vgg16, vgg16_bn
import math
from metrics.util import prepare_generated_img, get_noise
import torchvision.utils as vutils


try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

class EvalModel():
    def __init__(self, eval_embedder, batch_size, device, test_num, vgg_embedder_batch_norm=None):
        super(EvalModel, self).__init__()
        self.embedder_desc = eval_embedder
        self.device = device
        self.test_num = test_num


        self.load_Model(self.embedder_desc, vgg_embedder_batch_norm)
        self.setting_normalize()
        self.setting_resize()

        if batch_size is None:
            self.batch_size = 32
        else:
            self.batch_size = batch_size
        self.eval()

    def setting_resize(self):
        if self.embedder_desc in ['vgg16']:
            self.resize_img = 224
        elif self.embedder_desc in ['inceptionV3', 'inceptionV3_tensorflow']:
            self.resize_img = 299

    def setting_normalize(self):
        if self.embedder_desc in ['vgg16', 'inceptionV3']:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        elif self.embedder_desc in ['inceptionV3_tensorflow']:
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        elif self.embedder_desc in ['clip_resnet50']:
            mean, std = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]

        # mean = [0.48145466, 0.4578275, 0.40821073]
        # std = [0.26862954, 0.26130258, 0.27577711]
        
        self.mean = torch.Tensor(mean).view(1, 3, 1, 1).to(self.device)
        self.std = torch.Tensor(std).view(1, 3, 1, 1).to(self.device)



    def eval(self):
        self.eval_embedder.eval()

    def load_Model(self, embedder_desc, vgg_embedder_batch_norm=None):
        if embedder_desc == 'inceptionV3':
            self.eval_embedder = inception.InceptionV3((3,))
        elif embedder_desc == 'inceptionV3_tensorflow':
            self.eval_embedder = inception_tensorflow.InceptionV3()
        elif embedder_desc == 'vgg16':
            if vgg_embedder_batch_norm:
                self.eval_embedder = vgg16_bn(pretrained=True, progress=True)
            else:
                self.eval_embedder = vgg16(pretrained=True, progress=True)
        self.eval_embedder = self.eval_embedder.to(self.device)

    def get_embeddings_from_loaders(self, dataloader):
        features_list = []

        total_instance = len(dataloader.dataset)
        
        num_batches = math.ceil(float(total_instance) / float(self.batch_size))

        data_iter = iter(dataloader)

        start_idx = 0
        with torch.no_grad():
            for _ in tqdm(range(0, num_batches)):
                
                try:
                    images = next(data_iter)[0].clone().detach().to(self.device)
                    # images = torch.tensor(next(data_iter)[0].clone().detach()).to(self.device)
                    images = self.resize_and_normalize(images).to(self.device)
                except StopIteration:
                    break

                
                features = self.eval_embedder(images).to(self.device)
                features = features.detach().cpu().numpy()
                features_list[start_idx:start_idx+ features.shape[0]] = features
                start_idx = start_idx + features.shape[0]
                
        return features_list


    def get_embeddings_from_generator(self, args, generator, nz, device):
        features_list = []
        total_instance = self.test_num
        num_batches = math.ceil(float(total_instance) / float(self.batch_size))


        latent = get_noise(self.test_num, nz, device)

        start_idx = 0
        with torch.no_grad():
            for _ in tqdm(range(0, num_batches)):
                images = prepare_generated_img(args, generator, latent[start_idx:start_idx + self.batch_size], num_batches, device)
                
                images = torch.FloatTensor(images).to(self.device)
                images = self.resize_and_normalize(images)
                
                features = self.eval_embedder(images).to(self.device)
                features = features.detach().cpu().numpy()
                features_list[start_idx:start_idx+ features.shape[0]] = features
                start_idx = start_idx + features.shape[0]
                    
                if start_idx == self.test_num:
                    break
            
            return features_list

    def resize_and_normalize(self, x):

        # Resize imageSize -> 299 or 224
        x = F.interpolate(x,
                            size=(self.resize_img, self.resize_img),
                            mode='bilinear',
                            align_corners=False)
        
        # Convert pixel range 0 ~ 255 to 0 ~ 1 using x /  255.
        x = x/255.

        # Convert pixel range 0 ~ 1 to -1 ~ 1 using z-score
        x = (x - self.mean) / self.std
    
        return x

        




