import logging
import torch.nn as nn
logger = logging.getLogger('base')


def create_model(opt, channel):
    from .model import DDPM as M
    m = M(opt, channel)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m


def create_fusion_model(opt):
    from .Fusion_model import DFFM as M
    m = M(opt)
    logger.info('Fusion Model [{:s}] is created.'.format(m.__class__.__name__))
    return m


def create_cross_attention(opt):
    from models.transformer_cam_CrossFuse import cross_encoder
    device = 'cuda' if opt['gpu_ids'] is not None else 'cpu'
    img_size = 64
    patch_size = 2
    part_out = 256
    embed_dim = part_out * patch_size
    num_patches = int(img_size / patch_size) * int(img_size / patch_size)
    depth_self = 1
    depth_cross = 1
    Cross_Attention = nn.DataParallel(cross_encoder(img_size, patch_size, embed_dim, num_patches, depth_self, depth_cross)).to(device)
    return Cross_Attention