from metrics.features import EvalModel
from metrics import fid_score
import os
import logging
import numpy as np
log = logging.getLogger(__name__)



#######################################################
#           Calculate FID Score  for real samples
#######################################################
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

def save_embedding(numpy_dict, PATH):
    '''
    save the numpy
    '''
    # If you don't have embedding file, just make it.
    if os.path.isfile(PATH):
        pass
    else:
        np.savez(PATH, **numpy_dict)   
        
def get_embedding_real(args, dataloader, load_embedding_flag, fid_dict_path, device):
    if load_embedding_flag:
        print("Load existing numpy dict")
        embed_dict = load_embedding(fid_dict_path)
    else:
        print("Calculating numpy dict")
        embed_dict = calculate_embedding_from_loaders(dataloader=dataloader, 
                                                    test_num=args.test_num, 
                                                    device=device, 
                                                    vgg_embedder_batch_norm=args.vgg_embedder_batch_norm,
                                                    embedder_backbone=args.embedder_backbone)
        
        save_embedding(embed_dict, fid_dict_path)
        
    return embed_dict


def get_embedding_fake(args, model, device):
    embed_dict = calculate_embedding_from_generator(args=args, netG=model,
                                    nz=args.nz,
                                    test_num=args.test_num,
                                    device=device,
                                    vgg_embedder_batch_norm=args.vgg_embedder_batch_norm,
                                    embedder_backbone=args.embedder_backbone)
    return embed_dict

def calculate_embedding_from_loaders(*, dataloader, test_num, device, vgg_embedder_batch_norm, embedder_backbone):
    embed_list = [embedder_backbone]
    embed_dict = {}


    for embedder in embed_list:
        embed_dict[embedder] = {}
        embed_model = EvalModel(embedder, batch_size=64, device=device, test_num = test_num, vgg_embedder_batch_norm=vgg_embedder_batch_norm)

        # real_pred
        embed_dict[embedder]['pred'] = embed_model.get_embeddings_from_loaders(dataloader)

        # mu, sigma
        embed_dict[embedder]['mu'], embed_dict[embedder]['sigma'] = fid_score.getMean_and_Sigma(embed_dict[embedder]['pred'])
    return embed_dict


def calculate_embedding_from_generator(*, args, netG, nz, test_num, device, vgg_embedder_batch_norm=None, embedder_backbone):
    embed_list = [embedder_backbone]
    embed_dict = {}

    for embedder in embed_list:
        embed_dict[embedder] = {}
        embed_model = EvalModel(embedder, batch_size=128, device=device, test_num = test_num, vgg_embedder_batch_norm=vgg_embedder_batch_norm)
        
        # fake_pred
        embed_dict[embedder]['pred'] = embed_model.get_embeddings_from_generator(args, netG, nz, device)

        # mu, sigma
        embed_dict[embedder]['mu'], embed_dict[embedder]['sigma'] = fid_score.getMean_and_Sigma(embed_dict[embedder]['pred'])
    return embed_dict