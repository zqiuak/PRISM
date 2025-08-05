import torch
import os
import numpy as np

from tqdm import tqdm
from monai.transforms import SplitDim
from PIL import Image

def count_para(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # in MB
    pytorch_total_params = pytorch_total_params / 1024 / 1024
    return pytorch_total_params

def add_data_to_dict(target_dict:dict, key:str, data):
    if key not in target_dict.keys():
        target_dict[key] = data
    else:
        target_dict[key+"_(1)"] = (data)
    return target_dict

def get_data_in_dict(target_dict, filter):
    r"""
    search with specific filter in dictionary
    return dictionary
    """
    result_dict = {}
    for k in target_dict.keys():
        if filter in k:
            result_dict[k] = target_dict[k]
    return result_dict

def backup_scripts(sourcepath, outpath):
    '''
    copy files to same directory structure
    skip "Result", "Labels" folder
    '''
    skip_list = ["Result", "Labels", "__pycache__", "weights", "SOTA", '.git', ".ipynb"]
    import os
    import shutil
    if not outpath.endswith('/'):
        outpath += '/'
    shutil.copytree(sourcepath, outpath, ignore=shutil.ignore_patterns(*skip_list))


def save_ckpt(model, epoch, args, filename="model.pt", optimizer=None, scheduler=None, global_step=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    if global_step is not None:
        save_dict["global_step"] = global_step
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def load(model, model_dict):
    # make sure you load our checkpoints
    if "state_dict" in model_dict.keys():
        state_dict = model_dict["state_dict"]
    else:
        state_dict = model_dict
    current_model_dict = model.state_dict()
    for k in current_model_dict.keys():
        if (k in state_dict.keys()) and (state_dict[k].size() == current_model_dict[k].size()):
            pass
        else:
            print('error weight:',k)
    new_state_dict = {
        k: state_dict[k] if (k in state_dict.keys()) and (state_dict[k].size() == current_model_dict[k].size()) else current_model_dict[k]
        for k in current_model_dict.keys()}
    model.load_state_dict(new_state_dict, strict=True)
    return model

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)

def distributed_all_gather(
    tensor_list, valid_batch_size=None, out_numpy=False, world_size=None, no_barrier=False, is_valid=None
):
    if world_size is None:
        world_size = torch.distributed.get_world_size()
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
    if not no_barrier:
        torch.distributed.barrier()
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            tensor_list_out.append(gather_list)
    return tensor_list_out

def set_req_grad(model:torch.nn.Module, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def save_3d_as_slice(img, out_dir, out_name, frame_dim = 3, out_ext='jpg'):
    os.makedirs(out_dir, exist_ok=True)
    slices = SplitDim(
                dim=frame_dim,
                keepdim=False,
                update_meta=False,
            )(img.cpu().numpy())

    for i, slice in enumerate(tqdm(slices)):
        # SaveImage(output_dir, output_ext='.png', output_postfix=f'slice_{i}', channel_dim=0, print_log=False, scale=255)(slice)
        Image.fromarray((slice.transpose(1, 2, 0) * 255).astype(np.uint8)).save(os.path.join(out_dir, f'slice{i}_{out_name}.{out_ext}'))