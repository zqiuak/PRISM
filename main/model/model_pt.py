import numpy as np
from functools import partial
from typing_extensions import Final

import torch
from torch import Tensor
from typing import Tuple
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets.swin_unetr import SwinTransformer
from monai.networks.blocks import UnetOutBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep, look_up_option

from run.options import kargs as args

from util.basicutils import set_req_grad

from dataset.pretrain.ds import PART_TASK_DICT
META_TASK_DICT = {'Repetition time': 1, 'Echo time': 1, 'Flip angle': 1}


class ModDecoder(nn.Module):
    # Modulated Decoder for UnetrUpBlock, Deprecated, use ModConv3d instead
    def __init__(self, w_dim, decoder:UnetrUpBlock, *targs, **kwargs):
        raise DeprecationWarning('ModDecoder is deprecated, use ModConv3d instead')
        super().__init__(*targs, **kwargs)
        self.decoder = decoder
        old_conv = decoder.conv_block.conv1.conv
        self.w_dim = w_dim
        new_conv = ModConv3d(
            in_channels=old_conv.in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
            w_dim=w_dim,
            demod=True,
        )
        self.decoder.conv_block.conv1.conv = new_conv
        self.conv = self.decoder.conv_block.conv1.conv

    def set_w(self, w: Tensor):
        self.conv.set_w(w)

    def forward(self, x, skip_f):
        out = self.decoder(x, skip_f)
        return out
    
class ModConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride, padding, w_dim: int=768, demod = True, eps=1e-8, *targs, **kwargs):
        super(ModConv3d, self).__init__()
        '''
        default is regular conv3d, if want to modelulate, set_w() first.
        '''
        kernel_size = ensure_tuple_rep(kernel_size, 3)
        self.kernel_size = kernel_size 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        self.stride = stride
        self.padding = padding

        self.s = None
        self.projector = nn.Linear(w_dim, self.in_channels)
        self.modulate = False
        self.demodulate = demod
        self.eps = eps

    def set_w(self, w: Tensor):
        raise DeprecationWarning('set_w() is deprecated, input w in forward() instead')
        if self.s is not None:
            raise Warning('w is already set, it will be removed after every forward, please check if you need to set it again')
        
        self.modulate = True

    def forward(self, input:Tuple):
        x, w = input
        # if self.modulate:
        if w is not None:
            s = self.projector(w)
            s = s[:, None, :, None, None, None]

            conv_weight = self.weight[None, :, :, :, :, :]
            conv_weight = conv_weight * s
            if self.demodulate:
                sigma_inv = torch.rsqrt((conv_weight ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
                conv_weight = conv_weight * sigma_inv

            b, c, d, h, w = x.shape
            x = x.reshape(1, b * c, d, h, w)
            conv_weight = conv_weight.reshape(b * self.out_channels, self.in_channels, *self.kernel_size)
            x = F.conv3d(input=x, weight=conv_weight, stride=self.stride, padding=self.padding, groups=b)
            _, _, _D, _H, _W = x.shape
            x = x.reshape(b, -1, _D, _H, _W)
            self.s = None
            self.modulate = False
        else:
            x = F.conv3d(input=x, weight=self.weight, stride=self.stride, padding=self.padding)
        return x
        

class AdaIn(nn.Module):
    def __init__(self, w_dim, *targs, **kwargs):
        super().__init__(*targs, **kwargs)
        self.adin_affine = nn.ModuleList([
            nn.Linear(w_dim, 1),
            nn.Linear(w_dim, 1)])

    def forward(self, w, f):
        w0 = self.adin_affine[0](w)
        w1 = self.adin_affine[1](w)
        f = F.instance_norm(f)
        f = w0 * f + w1
        return f
    
class ReplaceW(nn.Module):
    def __init__(self, *targs, **kwargs):
        super().__init__(*targs, **kwargs)
        
    def forward(self, f, w):
        # w: [B, C] -> [B, C, D, H, W]
        w = w[:, :, None, None, None]
        w = w.expand_as(f)
        return w

class MixW(nn.Module):
    def __init__(self, latent_dim, *targs, **kwargs):
        super().__init__(*targs, **kwargs)
        self.mix_conv = nn.Conv3d(2*latent_dim, latent_dim, kernel_size=1)

    def forward(self, f, w):
        w = w[:, :, None, None, None]
        w = w.expand_as(f)
        mix_f = torch.cat([f, w], dim=1)
        mix_f = self.mix_conv(mix_f)
        return mix_f


class WMapping(nn.Module):
    def __init__(self, n_layer, w_dim, *targs, **kwargs):
        super().__init__(*targs, **kwargs)
        self.n_mapping_layer = n_layer 
        self.w_dim = w_dim
        mapping_layers = []
        for i in range(self.n_mapping_layer): 
            mapping_layers.append(
                nn.Sequential(
                    nn.Linear(self.w_dim, self.w_dim),
                    nn.ReLU(),
                )
            )
        self.noise_mapping = nn.Sequential(*mapping_layers)
        

    def forward(self, x):
        x = x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)
        x = self.noise_mapping(x)
        return x


class PT_MRI_FM(nn.Module):

    patch_size: Final[int] = 2

    def __init__(self, args = args):
        super(PT_MRI_FM, self).__init__()

        # pretext
        self.encoder_only = args.model_config.encoder_only
        self.disentangle = args.model_config.disentangle
        self.synthesize = args.model_config.synthesize
        self.replacew = args.model_config.replacew
        self.pred_meta = args.model_config.pred_meta
        self.pred_part = args.model_config.pred_part
        self.contrastive = args.model_config.contrastive
        self.mix_decoder = args.model_config.mix_decoder
        self.discriminate = args.model_config.discriminate

        # model params
        self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        self.emb_dim = args.model_config.emb_dim
        self.latent_dim = args.model_config.latent_dim
        self.depths = args.model_config.sw_layer
        self.num_heads = args.model_config.num_heads
        self.use_checkpoint = args.use_checkpoint
        self.normalize = True

        spatial_dims = args.spatial_dims
        img_size = args.roi_size
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_sizes = ensure_tuple_rep(self.patch_size, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)
        drop_rate = 0.0
        attn_drop_rate = 0.0
        feature_size = args.emb_dim
        dropout_path_rate = args.dropout_path_rate

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        self._check_input_size(img_size)

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")


        self.D = nn.ModuleDict()
        self.G = nn.ModuleDict()

        # construct swin transformer
        self.swinViT = SwinTransformer(
            in_chans=self.in_channels,
            embed_dim=self.emb_dim,
            window_size=window_size,
            patch_size=patch_sizes,
            depths=self.depths,
            num_heads=self.num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=self.use_checkpoint,
            spatial_dims=spatial_dims,
            use_v2=True,
        )
        self.D['swinViT'] = self.swinViT


        # encoder pre-text
        if self.pred_meta:
            self.meta_head = nn.Linear(self.latent_dim, len(META_TASK_DICT))
            self.D[f'meta_head'] =  self.meta_head
        if self.pred_part:
            self.part_head =  nn.Linear(self.latent_dim, len(PART_TASK_DICT))
            self.D['part_head'] = self.part_head

        if self.disentangle:

            self.structure_disentanglement = nn.Sequential(
                nn.Conv3d(self.latent_dim, self.latent_dim, kernel_size=1),
                nn.InstanceNorm3d(self.latent_dim),
                nn.LeakyReLU()
            )
            self.sequence_disentanglement = nn.Sequential(
                nn.Conv3d(self.latent_dim, self.latent_dim, kernel_size=1),
                nn.InstanceNorm3d(self.latent_dim),
                nn.LeakyReLU()
            )

            self.latent_agg = nn.Sequential(
                nn.Conv3d(2*self.latent_dim, self.latent_dim, kernel_size=1),
                nn.InstanceNorm3d(self.latent_dim),
                nn.LeakyReLU()
            )
            
            self.D['str_disen'] = self.structure_disentanglement
            self.D['seq_disen'] = self.sequence_disentanglement
            self.D['latent_agg'] = self.latent_agg



        if self.discriminate:
            self.disc_head = nn.Linear(self.latent_dim, 1)
            self.D['disc_head'] = self.disc_head


        # decoder pre-text
        if not self.encoder_only:

            self._construct_decoder(modconv=self.mix_decoder)
            self.up_decoder = nn.ModuleList([
                self.decoder1,
                self.decoder2,
                self.decoder3,
                self.decoder4,
                self.decoder5
            ])

            self.G['decoder'] = self.up_decoder

            self.recon_head = nn.Conv3d(self.emb_dim, self.out_channels, kernel_size=1, stride=1)
            self.G['recon_head'] = self.recon_head
            
        if self.synthesize:
            self.w_mapping = WMapping(n_layer=4, w_dim=self.latent_dim)
            
            if self.replacew:
                self.mix_w = ReplaceW()
            else:
                self.mix_w = MixW(self.latent_dim)

            self.G['wmap'] = self.w_mapping
            self.G['mix_w'] = self.mix_w

            # self.ada_in = AdaIn(w_dim=self.latent_dim)

        if self.contrastive:
            if self.disentangle:
                self.str_contrastive_head = nn.Linear(self.latent_dim, 512)
                self.seq_contrastive_head = nn.Linear(self.latent_dim, 512)

                self.D['str_contrastive_head'] = self.str_contrastive_head
                self.D['seq_contrastive_head'] = self.seq_contrastive_head
            else:
                self.contrastive_head = nn.Linear(self.latent_dim, 512)
                self.D['contrastive_head'] = self.contrastive_head



    def _construct_decoder(self, modconv = False):
        dim = self.latent_dim
        # if upsample == "large_kernel_deconv":
        #     self.conv = nn.ConvTranspose3d(dim, args.in_channels, kernel_size=(32, 32, 32), stride=(32, 32, 32))
        # elif upsample == "deconv":
        #     self.conv = nn.Sequential(
        #         nn.ConvTranspose3d(dim, dim // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        #         nn.ConvTranspose3d(dim // 2, dim // 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        #         nn.ConvTranspose3d(dim // 4, dim // 8, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        #         nn.ConvTranspose3d(dim // 8, dim // 16, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        #         nn.ConvTranspose3d(dim // 16, args.in_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        #     )
        # elif upsample == "vae":
        if modconv:
            conv_module = partial(ModConv3d, w_dim=self.latent_dim, demod=True)
        else:
            conv_module = nn.Conv3d

        self.decoder1 = nn.Sequential(
            conv_module(dim, dim // 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 2),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
        )
        self.decoder2 = nn.Sequential(
            conv_module(dim // 2, dim // 4, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 4),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
        )
        self.decoder3 = nn.Sequential(
            conv_module(dim // 4, dim // 8, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 8),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
        )
        self.decoder4 = nn.Sequential(
            conv_module(dim // 8, dim // 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 16),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
        )
        self.decoder5 = nn.Sequential(
            conv_module(dim // 16, dim // 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 16),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
        )


    @torch.jit.unused
    def _check_input_size(self, spatial_shape):
        img_size = np.array(spatial_shape)
        remainder = (img_size % np.power(self.patch_size, 5)) > 0
        if remainder.any():
            wrong_dims = (np.where(remainder)[0] + 2).tolist()
            raise ValueError(
                f"spatial dimensions {wrong_dims} of input image (spatial shape: {spatial_shape})"
                f" must be divisible by {self.patch_size}**5."
            )


    def gen_w(self, batch_size):
        z = torch.randn((batch_size, self.latent_dim)).to(self.w_mapping.parameters().__next__().device)
        w = self.w_mapping(z)
        return w

    # def set_synthesize_grad(self, flag:bool):
    #     raise DeprecationWarning('set_synthesize_grad() is deprecated')
    #     assert self.synthesize, 'synthesize must be True to set grad'

    #     if self.mix_decoder:
    #         for decoder in self.up_decoder:
    #             set_req_grad(decoder.conv.projector, flag)
        
        

    #     set_req_grad(self.w_mapping, flag)
    #     set_req_grad(self.mix_w, flag)
    #     set_req_grad(self.seq_contrastive_head, flag)
    #     set_req_grad(self.str_contrastive_head, flag)
        
    # def set_discr_head_grad(self, flag:bool):
    #     raise DeprecationWarning('set_discr_head_grad() is deprecated')
    #     set_req_grad(self.disc_head, flag)

    def _mix_noise(self, sequence_f, w):
        # synthesize:
        # add noise and fusion
        sequence_f = self.mix_w(sequence_f, w)
        return sequence_f
    
    def _mod_decoder(self, w):
        raise DeprecationWarning('mod_decoder is deprecated, use ModConv3d instead')
        # pass w to decoder 
        self.decoder1.set_w(w)
        self.decoder2.set_w(w)
        self.decoder3.set_w(w)
        self.decoder4.set_w(w)
        self.decoder5.set_w(w)
    
    def _disentangle(self, latent_f):
            # disentangle and fusion
            structural_f = self.structure_disentanglement(latent_f)
            sequence_f = self.sequence_disentanglement(latent_f)

            return structural_f, sequence_f
    
    def _aggreagte(self, latent_f, structural_f, sequence_f):
        res = latent_f
        cat_f = torch.cat((structural_f, sequence_f), dim=1)
        latent_f = self.latent_agg(cat_f) + res
        return latent_f

    def _encoder_forward(self, x, do_synthesize = False, w = None):
        # encode
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._check_input_size(x.shape[2:])
        hidden_states_out  = self.swinViT(x, self.normalize)
        latent_f = hidden_states_out[4]

        skip_f = (hidden_states_out[0], hidden_states_out[1], hidden_states_out[2], hidden_states_out[3])

        if self.disentangle:
            structural_f, sequence_f = self._disentangle(latent_f)

            if do_synthesize:
                assert w is not None, 'w must be provided for synthesis'
                assert w.shape[0] == latent_f.shape[0], 'w must have the same batch size as input'
                sequence_f = self._mix_noise(sequence_f, w)

            latent_f = self._aggreagte(latent_f, structural_f, sequence_f)

        else:
            structural_f = latent_f
            sequence_f = latent_f
            if do_synthesize:
                assert w is not None, 'w must be provided for synthesis'
                assert w.shape[0] == latent_f.shape[0], 'w must have the same batch size as input'
                latent_f = self._mix_noise(latent_f, w)

        return latent_f, skip_f, structural_f, sequence_f

    def _decoder_forward(self, x_in, latent_f, hidden_states_out, do_synthesize = False, w = None):
        '''
        swinunetr:
            x_in -> .encoder1 -> skip_in -->.decoder1 ---> out_f
            |                                   ^dec_0
            hs0  -> .encoder2 -> skip_0 --->.decoder2
            |                                   ^dec_1
            hs1  -> .encoder3 -> skip_1 --->.decoder3
            |                                   ^dec_2
            hs2  -> .encoder4 -> skip_2 --->.decoder4
            |                                   ^dec_3
            hs3  -------------------------->.decoder5
            |                                   ^
            hs4  -> .encoder10-> skip_4 --------^
        ssl:
        x_i
        |  
        hs0
        |  
        hs1
        |  
        hs2
        |  
        hs3
        |  
        hs4 -> latent_f-> structural_f, sequence_f-> latent_f->up^
        '''

        if do_synthesize:
            assert w is not None, 'w must be provided for synthesis'
            if self.mix_decoder:
                pass
                # self._mod_decoder(w)
            else:
                w = None
        
        # sw = self.fm.bb_model
        # skip_in = sw.encoder1(x_in)
        # skip_0 = sw.encoder2(hidden_states_out[0])
        # skip_1 = sw.encoder3(hidden_states_out[1])
        # skip_2 = sw.encoder4(hidden_states_out[2])
        # skip_4 = sw.encoder10(latent_f)

        # dec3 = self.decoder5(skip_4, hidden_states_out[3])
        # dec2 = self.decoder4(dec3, skip_2)
        # dec1 = self.decoder3(dec2, skip_1)
        # dec0 = self.decoder2(dec1, skip_0)
        # out_f = self.decoder1(dec0, skip_in)

        out = latent_f
        for decoder in self.up_decoder:
            if self.mix_decoder:
                out = decoder((out, w))
            else:
                out = decoder(out)
        out_f = out

        return out_f
    
    def _get_meta_pred(self, sequence_f):
        flatten_sequence_f = F.adaptive_avg_pool3d(sequence_f, (1,1,1)).flatten(start_dim=1)
        pred = self.meta_head(flatten_sequence_f) # [B, n_meta]
        meta_pred = {task:pred[:, i][:,None] for i, task in enumerate(META_TASK_DICT.keys())}
        
        return meta_pred
    
    def _get_part_pred(self, structural_f):
        
        flatten_structural_f = F.adaptive_avg_pool3d(structural_f, (1,1,1)).flatten(start_dim=1)
        part_pred = self.part_head(flatten_structural_f) # [B, n_part]
        # part_pred = F.softmax(part_pred, dim=1)
    
        return part_pred

    def _get_contrastive_f(self, latent_f, structural_f, sequence_f):
        
        if self.disentangle:
            flatten_structural_f = F.adaptive_avg_pool3d(structural_f, (1, 1, 1)).flatten(start_dim=1) # [B, C] 
            str_contrast_f = self.str_contrastive_head(flatten_structural_f)

            flatten_sequence_f = F.adaptive_avg_pool3d(sequence_f, (1, 1, 1)).flatten(start_dim=1)
            seq_contrast_f = self.seq_contrastive_head(flatten_sequence_f)

            contrast_f = {
                'seq': seq_contrast_f, # [B, C]
                'str': str_contrast_f, # [B, C]
            }
        
        else:
            flatten_latent = F.adaptive_avg_pool3d(latent_f, (1, 1, 1)).flatten(start_dim=1) # [B, C]
            latent_contrast_f = self.contrastive_head(flatten_latent)
            contrast_f = {
                'lat': latent_contrast_f, # [B, C]
            }

        return contrast_f
    
    def discriminate_forward(self, x) -> Tuple[Tensor, dict, dict]:

        latent_f, _, structural_f, sequence_f = self._encoder_forward(x, do_synthesize=False)

        if self.discriminate:
            # latent_f: [B, C, H, W, D]: 2,768,3,3,3
            flatten_latent_f = F.adaptive_avg_pool3d(latent_f, (1, 1, 1)).flatten(start_dim=1) # [B, C]
            disc_pred = self.disc_head(flatten_latent_f)
        else:
            disc_pred = None

        if self.pred_meta:
            meta_pred = self._get_meta_pred(sequence_f)
        else:
            meta_pred = None

        if self.pred_part:
            part_pred = self._get_part_pred(structural_f)
        else:
            part_pred = None

        if self.contrastive:
            contrast_f = self._get_contrastive_f(latent_f, structural_f, sequence_f)
        else:
            contrast_f = None

        return disc_pred, contrast_f, [meta_pred, part_pred]

    def recon_forward(self, x, do_synthesize = False, w = None) -> Tuple[Tensor, dict, dict]:
        '''
        out: out_img, contrast_f, meta_pred
        '''
        x = x.detach()
        latent_f, skip_f, structural_f, sequence_f = self._encoder_forward(x, do_synthesize, w)
          
        if self.pred_meta:
            meta_pred = self._get_meta_pred(sequence_f)
        else:
            meta_pred = None
        if self.pred_part:
            part_pred = self._get_part_pred(structural_f)
        else:
            part_pred = None

        # pre-text tasks:rotation, contrastive, reconstruction
        if self.contrastive:
            contrast_f = self._get_contrastive_f(latent_f, structural_f, sequence_f)
        else:
            contrast_f = None

        out_f = self._decoder_forward(x, latent_f, skip_f, do_synthesize, w)
        out_img = self.recon_head(out_f)
        
        return out_img, contrast_f, [meta_pred, part_pred]

    def _recon_and_meta_forward(self, real_img, masked_img, *targs, **kwargs):

        recon_img, _, meta_part_pred = self.recon_forward(masked_img)
        return recon_img, meta_part_pred 

    def _recon_and_discriminate_forward(self, real_img, masked_img, detach_D = False, *targs, **kwargs):

        real_disc_pred, _, real_meta_pred = self.discriminate_forward(real_img)
        recon_img, _, _ = self.recon_forward(masked_img)
        recon_disc_pred, _, _ = self.discriminate_forward(recon_img.detach() if detach_D else recon_img)

        return real_disc_pred, real_meta_pred, recon_img, recon_disc_pred

    def _syn_and_discriminate_forward(self, real_img, detach_D = False, *targs, **kwargs):

        w = self.gen_w(real_img.shape[0])
        fake_img, real_contrast_f, meta_pred = self.recon_forward(real_img, do_synthesize=True, w=w)
        fake_disc_pred, fake_contrast_f, _ = self.discriminate_forward(fake_img.detach() if detach_D else fake_img)
        return fake_img, real_contrast_f, meta_pred, fake_disc_pred, fake_contrast_f
    
    def do_recon(self, x):
        # test recon
        recon_img, _, _ = self.recon_forward(x)
        return recon_img

    def do_syn(self, x, w=None):
        if w is None:
            w = self.gen_w(x.shape[0])
        fake_img, _, _ = self.recon_forward(x, do_synthesize=True, w=w)
        return fake_img

    def forward(self, mode, real_img, masked_img = None, *targs, **kwargs):
        real_img = real_img.contiguous()
        masked_img = masked_img.contiguous() if masked_img is not None else None
        if mode == 'rm':
            return self._recon_and_meta_forward(real_img, masked_img, *targs, **kwargs)
        if mode == 'rd':
            return self._recon_and_discriminate_forward(real_img, masked_img, *targs, **kwargs)
        if mode == 'sd':
            return self._syn_and_discriminate_forward(real_img, *targs, **kwargs)
        else:
            raise ValueError(f'Unknown mode: {mode}')
