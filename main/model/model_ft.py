import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Union, Optional, Tuple, Dict, Any

from monai.networks.nets.swin_unetr import SwinTransformer, SwinUNETR
from monai.networks.nets.swin_unetr import MERGING_MODE
from monai.networks.blocks import UnetrBasicBlock, UnetOutBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep
from monai.networks.nets import ResNet

from run.options import kargs as args

from model.losses import Pretrain_Loss

from collections.abc import Sequence
from typing_extensions import Final
from monai.utils.deprecate_utils import deprecated_arg
from monai.utils import ensure_tuple_rep, look_up_option

from monai.networks.nets import resnet101, resnet50
from monai.networks.blocks import TransformerBlock

class SwinVitWrapper(nn.Module):
    def __init__(self, 
                 in_channels,
                 emb_dim,
                 sw_layer,
                 num_heads,
                 use_v2
                 ):
        super(SwinVitWrapper, self).__init__()
        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=emb_dim,
            window_size=(7,7,7),
            patch_size=(2,2,2),
            depths= sw_layer,
            num_heads = num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
            use_checkpoint=False,
            spatial_dims=3,
            use_v2=use_v2,
        )

    def forward(self, x):
        return self.swinViT(x)[4]


class MRI_CLS(nn.Module):
    def __init__(self, 
                 task_dict: dict,
                 agg_mode: str,
                 in_channels: int = 1,
                 seq_num = None,
                 seperate_encoders: Optional[bool] = False,
                 ):
        '''
        seq_num: number of sequences, required when agg_mode is 'cat' or separate_encoders is True
        seperate_encoders: if True, each sqeuence has its own encoder, otherwise all sequences share the same encoder.
        taskdict:{'task1': num_classes, 'task2': num_classes}
        agg_mode: 'trans' or 'mil' or 'cat'
        
        '''
        super(MRI_CLS, self).__init__()
        
        self.in_channels = in_channels
        self.task_dict = task_dict
        self.agg_mode = agg_mode
        self.seq_num = seq_num
        self.seperate_encoders = seperate_encoders
        # model structure
        self.emb_dim = args.emb_dim
        self.latent_dim = args.latent_dim
        sw_layer = args.sw_layer
        num_heads = args.num_heads
        use_v2 = args.use_v2

        if seperate_encoders:
            assert seq_num is not None, "seq_num should be provided when seperate_encoders is True"
            encoder_num = seq_num
        else:
            encoder_num = 1

        self.encoders = nn.ModuleList([])
        for i in range(encoder_num):
            self.encoders.append(
                SwinVitWrapper(
                    in_channels=in_channels,
                    emb_dim=self.emb_dim,
                    sw_layer=sw_layer,
                    num_heads=num_heads,
                    use_v2=use_v2
                )
            )

        if agg_mode == 'trans':
            self.latent_agg = TransformerBlock(hidden_size=self.latent_dim, mlp_dim=self.latent_dim, with_cross_attention=True,num_heads=4)
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.latent_dim))
            
        elif agg_mode == 'mil':
            self.latent_agg = nn.Identity()

        elif agg_mode == 'cat':
            assert seq_num is not None, "seq_num should be provided when agg_mode is 'cat'"
            self.latent_agg = nn.Linear(self.latent_dim * seq_num, self.latent_dim)
        
        else:
            raise ValueError(f"Unknown aggregation mode: {agg_mode}")
        
        self.cls_head = nn.ModuleDict()
        for task_name, cls_num in task_dict.items():
            self.cls_head[task_name] = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim//2),
                nn.LayerNorm(self.latent_dim//2),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(self.latent_dim//2, cls_num),
            )

    def forward(self, input):
        # encode
        if isinstance(input, list): # [[B, C, D, H, W], [B, C, D, H, W], ...], len(x) = seq_num
            batch_size = input[0].shape[0]
            seq_num = len(input)
            if self.seq_num is not None:
                assert seq_num == self.seq_num, f"seq_num should be {self.seq_num} defined in model init, but got {seq_num}"

            if self.seperate_encoders:
                seq_features = []
                for i, seq in enumerate(input):
                    latent_f = self.encoders[i](seq)  
                    seq_features.append(latent_f)
                # seq_features : [[B, C, ...], [B, C], ...]
                seq_features = torch.cat(seq_features, dim=0) # [B*seq_num, C, ...]

            else:

                x = torch.concat(input, dim=0)  # [B*seq_num, C, D, H, W]
                seq_features = self.encoders[0](x)  # [B*seq_num, C]


        elif isinstance(input, torch.Tensor): # [B, C, D, H, W]
            assert self.seperate_encoders is False, "seperate_encoders should be False when input is a single tensor"
            seq_num = 1
            batch_size = input.shape[0]
            seq_features = self.encoders[0](input) # [B*seq_num, C, D, H, W]

        seq_features = F.adaptive_avg_pool3d(seq_features, (1, 1, 1))
        seq_features  = seq_features.flatten(1)  # [B*seq_num, C]
        seq_features = seq_features.reshape(batch_size, seq_num, self.latent_dim) # [B, seq_num, C]  

        if self.agg_mode == 'trans':
            cls_token = self.cls_token.repeat(seq_features.shape[0], 1, 1) # [B, 1, C]
            seq_features = torch.cat([cls_token, seq_features], dim=1)  # [B, seq_num+1, C]
            f = self.latent_agg(seq_features)[:, 0, :]  # [B, C]

        elif self.agg_mode == 'mil':
            f = seq_features.max(dim=1).values  # [B, C]

        
        elif self.agg_mode == 'cat':
            seq_features = seq_features.reshape(batch_size, self.latent_dim * seq_num)
            f = self.latent_agg(seq_features)  # [B, C]

        self.seq_features = seq_features  # for debug
        self.f = f  # for debug
        out_dict = {}
        for task_name, cls_head in self.cls_head.items():
            out_dict[task_name] = cls_head(f)

        return out_dict
    
class MRI_SEG(nn.Module):
    def __init__(self, 
                 out_channels:int,
                 in_channels: int = 1,
                 ):
        
        super(MRI_SEG, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels        
        # model structure
        self.emb_dim = args.emb_dim
        self.latent_dim = args.latent_dim
        use_v2 = args.use_v2
        sw_layer = args.sw_layer
        num_heads = args.num_heads

        self.backbone = SwinUNETR(
            img_size = (96, 96, 96), # this argument does not really work, but it is required by the constructor
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            depths = sw_layer,
            num_heads = num_heads,
            feature_size = self.emb_dim,
            use_checkpoint = True,
            use_v2 = use_v2,
        )

        self.swinViT = self.backbone.swinViT

    def forward(self, x):
        out = self.backbone(x)
        return out
