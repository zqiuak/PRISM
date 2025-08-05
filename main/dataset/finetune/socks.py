import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from model.model_ft import MRI_SEG

from monai.transforms import AsDiscrete
from .ds import INPUT_SIZE
from functools import partial


# background, RV (right ventricle), MYO (myocardium), LV (left ventricle)
SEG_CLASS = {'background': 0, 'RV': 1, 'MYO': 2, 'LV': 3}

def get_task_metrics():
    Task_Metrics = {}
    Task_Metrics['all'] = {
        'dice': DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)
    }
    return Task_Metrics

def prepare_data(data, mode, args):
    img = data["image"].cuda(args.gpu)  # [B, C, D, H, W]

    label = data["label"]  # [B, 1, D, H, W]
    if label.shape[1] ==1 and label.shape[1]!= len(SEG_CLASS):
        label = AsDiscrete(to_onehot=len(SEG_CLASS), dim=1)(label)  
        # label = label.squeeze(1)  # [B, D, H, W]
        # label = label.long()  # [B, D, H, W]
        # label = F.one_hot(label, num_classes=len(SEG_CLASS)).permute(0, 4, 1, 2, 3)
    label = label.cuda(args.gpu)
    return img, label

def infer_model(model, data, mode, args):
    if mode == 'train':
        seg_mask_pred = model(data)  # [B, C, D, H, W]
    else:
        with torch.no_grad():
            seg_mask_pred = sliding_window_inference(
                inputs=data,
                roi_size=INPUT_SIZE,
                sw_batch_size=args.sw_batch_size,
                predictor=model,
                overlap=0.75 if args.infer else args.sw_overlap,
            )
    return seg_mask_pred

def loss_fn(pred, label, mode='train', args=None):
    # Class weights can be adjusted for cardiac segmentation if needed
    # weight = torch.tensor([0.1, 1.0, 2.0, 1.0], dtype=torch.float32).to(pred.device)
    loss_func = DiceCELoss(include_background=False, softmax=True)
    loss = loss_func(pred, label)
    return loss

# def _post_process(pred):
#     # pred = F.one_hot(torch.argmax(pred, 1), num_classes=len(SEG_CLASS)).permute(0, 4, 1, 2, 3)
#     pred = torch.softmax(pred, dim=1)  # [B, D, H, W]
#     return pred

_post_process = AsDiscrete(argmax=True, to_onehot=len(SEG_CLASS), dim=1)

@torch.no_grad()
def cal_metric(pred, label, mode, task_metrics, args, aggregate=False):
    if aggregate:
        result = {}
        all_res = task_metrics['all']['dice'].aggregate()
        res_value = all_res[1:].mean().item()  # Skip background class
        result['all'] = {'dice': res_value}
        for cls_idx, cls_name in enumerate(SEG_CLASS.keys()):
            result[cls_name] = {'dice': all_res[cls_idx].item()}
        return result, res_value
    
    assert pred is not None and label is not None, "pred and label should not be None when aggregate is False"

    pred = _post_process(pred)  # [B, C, D, H, W] one-hot

    # Update metrics without aggregating
    eval_metric = task_metrics['all']['dice'](pred, label)

    batch_metric = {}
    for i, k in enumerate(SEG_CLASS.keys()):
        batch_metric[k] = eval_metric[0][i].item()
    
    return task_metrics, batch_metric

FT_MRI_FM = partial(MRI_SEG, out_channels=len(SEG_CLASS))