import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

from monai.data import Dataset
from monai.transforms import Compose
from torch.utils.data import DataLoader

from dataset.private_cancer.ds import pt_seqlist as pv_cancer_list, Loader as pv_cancer_loader
from dataset.private_knee.ds import pt_seqlist as pv_knee_list, Loader as pv_knee_loader
from dataset.picai.ds import pt_seqlist as picai_list, Loader as picai_loader
# from dataset.adni.ds import pt_seqlist as adni_list, Loader as adni_loader

from dataset.datautils import MetadataMap

def load_and_save_cache(seq_list, Loader, cache_dir, overwrite = False):
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    ds = Dataset(data=seq_list, transform=Compose(Loader))
    dl = DataLoader(ds, batch_size=1, num_workers=8, shuffle=False, collate_fn=lambda x: x)
    tq = tqdm(dl, total=len(ds))
    for i, data in enumerate(tq):
        data = data[0]
        image = data["image"]
        case_id = data["case_id"]
        seq_id = data["seq_id"]
        try:
            tr = image.meta[MetadataMap]
        except:
            tr = None
        try:
            te = image.meta[MetadataMap]
        except:
            te = None
        try:
            flip_angle = image.meta[MetadataMap]
        except:
            flip_angle = None
        if not overwrite and os.path.exists(os.path.join(cache_dir, f"{case_id}_{seq_id}.pt")):
            continue
        torch.save({"image": image, "tr": tr, "te": te, "flip_angle": flip_angle}, os.path.join(cache_dir, f"{case_id}_{seq_id}.pt"))