import torch
import math
from tqdm import tqdm
import numpy as np
import torchvision.utils as vutils

def prepare_generated_img(args, netG, latent, num_batches, device):
    with torch.no_grad():
        images = netG(latent)
        images = quantize_images(images)
        
    return images


def get_noise(test_num, nz, device):
    """
    test_num: 평가 때 생성하는 이미지 개수
    nz: 학습한 Generator의 latent dim
    device: testing for CUDA
    """
    latent = torch.randn(test_num, nz, 1, 1, device=device)
    return latent

def quantize_images(x):
    # -1 ~ 1 -> 0 ~ 255
    x = (x + 1)/2
    x = (255.0*x + 0.5).clamp(0.0, 255.0)
    x = x.detach().cpu().numpy().astype(np.uint8)
    return x