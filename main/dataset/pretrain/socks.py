import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from run.options import kargs as args
from util.ops import aug_rand

# from losses.loss import Contrast
from monai.losses import ContrastiveLoss, SSIMLoss
from monai.metrics import CumulativeAverage, ConfusionMatrixMetric
from monai.networks.blocks import UnetOutBlock

from dataset.pretrain.ds import PART_TASK_DICT
from model.model_pt import PT_MRI_FM, META_TASK_DICT



LOSS_WEIGHT = {'recon': 1.0, 'meta': 1.0, 'disc':1.0}
contrast_lossfunc = ContrastiveLoss()


def get_task_metrics():
    Task_Metrics = {}

    Task_Metrics['recon'] = {'l1': CumulativeAverage()}

    if args.model_config.pred_meta:
        for k in META_TASK_DICT.keys():
            Task_Metrics[k] = {'l1': CumulativeAverage()}
    if args.model_config.pred_part:
        Task_Metrics['part'] = {'acc': ConfusionMatrixMetric(metric_name='accuracy', get_not_nans=False)}

    if args.model_config.contrastive:
        Task_Metrics['contrast'] = {'l1': CumulativeAverage()}

    return Task_Metrics
        


def prepare_data(batch_data, mode, args):
    img = batch_data['image'].cuda(args.gpu)
    masked_img = aug_rand(img, args).detach().clone()
    meta_label = {k:batch_data[k].float().unsqueeze(1).cuda() for k in META_TASK_DICT.keys()}
    part_label = batch_data['part'].cuda(args.gpu)
    # part_label = F.one_hot(batch_data['part'], num_classes=len(PART_TASK_DICT)).long().cuda(args.gpu)

    return (img, masked_img), (img, meta_label, part_label)



def infer_model(model, data, mode, args):
    raise DeprecationWarning("Please use the XX_forward in the model")

def loss_fn(pred, label, stage, args):
    raise DeprecationWarning("Please call the loss function in below")
    recon_img, contrast_f, meta_pred, disc_pred = pred
    img, meta_label, part_label = label
    recon_label = img

    total_loss = 0

    if not args.encoder_only:
        total_loss += (Recon_Loss(recon_img, recon_label) * LOSS_WEIGHT['recon'])
    
    if args.model_config.pred_meta:
        total_loss += Meta_Loss(meta_pred, meta_label) * LOSS_WEIGHT['meta']
        total_loss += Part_Loss(part_pred, part_label) * LOSS_WEIGHT['meta']

    if args.model_config.contrastive:
        # contrast_loss = contrast_lossfunc(contrast_f, sync_contrast_f)
        # total_loss += contrast_loss
        pass

    return total_loss

def Recon_Loss(recon_img, recon_label):
    loss = F.l1_loss(recon_img, recon_label)
    # + 0.5*SSIMLoss(spatial_dims=3)(recon_img, recon_label)
    return loss

def Meta_Loss(meta_pred, meta_label):
    meta_loss = {}
    loss = 0
    for k in META_TASK_DICT.keys():
        no_label_meta = meta_label[k] == 0.0
        eps = torch.tensor(1e-8, dtype=meta_pred[k].dtype, device=meta_pred[k].device)
        meta_label[k] = meta_label[k].to(meta_pred[k].dtype)
        meta_label[k][no_label_meta] = meta_pred[k][no_label_meta].detach().clone()+ eps
        meta_loss[k] = F.l1_loss(meta_pred[k], meta_label[k])
        loss += (meta_loss[k] * LOSS_WEIGHT['meta'])

    return loss

def Part_Loss(part_pred, part_label):
    loss = F.cross_entropy(part_pred, part_label)
    return loss

def Disc_Loss(disc_pred, disc_label):
    '''
    cross entropy loss for discriminator
    '''
    loss = F.binary_cross_entropy_with_logits(disc_pred, disc_label)
    return loss

def Disc_real_loss(real_pred):
    real_loss = F.softplus(-real_pred)
    return real_loss.mean()

def Disc_fake_loss(fake_pred):
    fake_loss = F.softplus(fake_pred)
    return fake_loss.mean()

def D_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def G_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss

def Contrast_Loss(real_contrast_f, sync_contrast_f):
    ori_seq_f, ori_str_f = real_contrast_f['seq'], real_contrast_f['str']
    sync_seq_f, sync_str_f = sync_contrast_f['seq'], sync_contrast_f['str']
    # sim_matrix = F.cosine_similarity(ori_seq_f.detach().unsqueeze(1), ori_seq_f.detach().unsqueeze(0), dim=2)
    # seq_f_bank = ori_seq_f[sim_matrix.max(dim=0)[1]]
    # loss = contrast_lossfunc(ori_str_f, sync_str_f) + contrast_lossfunc(sync_seq_f, seq_f_bank)
    loss = contrast_lossfunc(ori_str_f, sync_str_f)
    return loss


@torch.no_grad()
def cal_metric(pred, label, mode, task_metrics, args, aggregate = False):
    if aggregate:
        result = {}
        for task in task_metrics.keys():
            for k, m in task_metrics[task].items():
                val = m.aggregate()
                if k == 'acc':
                    val = val[0]
                result[task] = {k: float(val)}
        eval_metric = 1/result['recon']['l1'] if result['recon']['l1'] else 0

        return result, eval_metric
    
    assert pred is not None and label is not None, "pred and label should not be None when aggregate is False"
    recon_img, contrast_f, meta_part_pred, disc_pred = pred
    meta_pred, part_pred = meta_part_pred
    
    img, meta_label, part_label = label
    recon_label = img


    if not args.encoder_only:
        batch_size = recon_img.shape[0]
        recon_l1 = F.l1_loss(recon_img, recon_label)
        task_metrics['recon']['l1'].append(recon_l1, batch_size)
        eval_metric = recon_l1.item()
        eval_metric = 1/eval_metric
    else:
        eval_metric = 0


    if args.model_config.pred_meta:
        for k in META_TASK_DICT.keys():
            meta_mask = meta_label[k]>0
            if meta_mask.sum() == 0:
                continue
            else:
                active_batch_size = meta_mask.sum()
                active_label = meta_label[k][meta_mask]
                active_pred = meta_pred[k][meta_mask]
                task_metrics[k]['l1'].append(F.l1_loss(active_pred, active_label), active_batch_size)

    
    if args.model_config.pred_part:
        part_pred = F.one_hot(part_pred.argmax(dim=1), num_classes=len(PART_TASK_DICT))
        part_label = F.one_hot(part_label, num_classes=len(PART_TASK_DICT))

        task_metrics['part']['acc'](part_pred, part_label)


    return task_metrics, eval_metric


