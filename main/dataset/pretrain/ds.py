import logging
import numpy as np
from torch.utils.data import ConcatDataset, random_split, Dataset as torch_dataset
import torch
import pandas as pd
from monai.transforms import *
from monai.data import CacheDataset, SmartCacheDataset, Dataset, DataLoader, DistributedSampler

from run.options import kargs as args
from dataset.adni.ds import get_pt_list as adni_getfunc, pt_seq_loader as adni_loader
from dataset.oasis3.ds import get_pt_list as oasis3_getfunc, pt_seq_loader as oasis3_loader
from dataset.fastmri.ds import get_pt_list as fastmri_getfunc, pt_seq_loader as fastmri_loader
from dataset.private_breast.ds import get_pt_list as pv_breast_getfunc, pt_seq_loader as pv_breast_loader
from dataset.private_cancer.ds import get_pt_list as pv_cancer_getfunc, pt_seq_loader as pv_cancer_loader
from dataset.private_knee.ds import get_pt_list as pv_knee_getfunc, pt_seq_loader as pv_knee_loader
from dataset.picai.ds import get_pt_list as picai_getfunc, pt_seq_loader as picai_loader


PART_TASK_DICT = {'head': 0, 'neck': 1, 'chest': 2, 'upper abdomen': 3, 'lower abdomen': 4, 'knee': 5, 'unkown': 6}

PART_CLASS_MAP = {
    'oai': 5,
    'oasis3': 1,
    'adni': 1,
    'picai': 4,
    'fastmri': {'Prostate': 4,
                'Brain': 0,
                'Knee': 5,
                'Breast': 2},
    'priv_knee': 5,
    'priv_breast': 2,
    'priv_cancer': {'直肠癌': 4, 
                    '宫颈癌': 4, 
                    '头颈癌': 1, 
                    '胰腺癌': 3, 
                    '乳腺癌': 2, 
                    '子宫内膜癌': 4, 
                    '肝细胞癌': 3, 
                    '胆管癌': 3, 
                    '膀胱癌': 4}
}

# number is now determined by the dataset
PRETRAIN_SCALE_MAP = {
    'all':{
        'oai': -1,
        'oasis3': -1,
        'adni': -1,
        'picai': -1,
        'fastmri': {'Prostate': -1,
                    'Brain': -1,
                    'Knee': -1,
                    'Breast': -1},
        'priv_knee': -1,
        'priv_breast': -1,
        'priv_cancer': {'直肠癌': -1, 
                        '宫颈癌': -1, 
                        '头颈癌': -1, 
                        '胰腺癌': -1, 
                        '乳腺癌': -1, 
                        '子宫内膜癌': -1, 
                        '肝细胞癌': -1, 
                        '胆管癌': -1, 
                        '膀胱癌': -1}
    },
    '10k':{
        'oai': -1,
        'oasis3': -1,
        'adni': -1,
        'picai': -1,
        'fastmri': {'Prostate': -1,
                    'Brain': -1,
                    'Knee': -1,
                    'Breast': -1},
        'priv_knee': -1,
        'priv_breast': -1,
        'priv_cancer': {'直肠癌': -1, 
                        '宫颈癌': -1, 
                        '头颈癌': -1, 
                        '胰腺癌': -1, 
                        '乳腺癌': -1, 
                        '子宫内膜癌': -1, 
                        '肝细胞癌': -1, 
                        '胆管癌': -1, 
                        '膀胱癌': -1}
    },
    '53k':{
        'oai': -1,
        'oasis3': -1,
        'adni': -1,
        'picai': -1,
        'fastmri': {'Prostate': -1,
                    'Brain': -1,
                    'Knee': -1,
                    'Breast': -1},
        'priv_knee': -1,
        'priv_breast': -1,
        'priv_cancer': {'直肠癌': -1, 
                        '宫颈癌': -1, 
                        '头颈癌': -1, 
                        '胰腺癌': -1, 
                        '乳腺癌': -1, 
                        '子宫内膜癌': -1, 
                        '肝细胞癌': -1, 
                        '胆管癌': -1, 
                        '膀胱癌': -1}
    },
    '336k':{
        'oai': -1,
        'oasis3': -1,
        'adni': -1,
        'picai': -1,
        'fastmri': {'Prostate': -1,
                    'Brain': -1,
                    'Knee': -1,
                    'Breast': -1},
        'priv_knee': -1,
        'priv_breast': -1,
        'priv_cancer': {'直肠癌': -1, 
                        '宫颈癌': -1, 
                        '头颈癌': -1, 
                        '胰腺癌': -1, 
                        '乳腺癌': -1, 
                        '子宫内膜癌': -1, 
                        '肝细胞癌': -1, 
                        '胆管癌': -1, 
                        '膀胱癌': -1}
    },
}

PRETRAIN_DS_MAP = {
    'oai': (oasis3_getfunc, oasis3_loader),
    'adni': (adni_getfunc, adni_loader),
    'oasis3': (oasis3_getfunc, oasis3_loader),
    'fastmri': (fastmri_getfunc, fastmri_loader),
    'priv_breast': (pv_breast_getfunc, pv_breast_loader),
    'priv_cancer': (pv_cancer_getfunc, pv_cancer_loader),
    'priv_knee': (pv_knee_getfunc, pv_knee_loader),
    'picai': (picai_getfunc, picai_loader),

}

DATA_KEYS = {'case_id':'unknown', 
            #  'seq_id': 0, 
             'image':0.0,
             'Repetition time':0.0, 
             'Echo time':0.0, 
             'Flip angle':0.0,
             'part': 6}



before_load_transforms = [
    # CopyItemsd(keys=["image"], times=1, names=["img_path"]),
]

loader = [LoadImaged(keys=["image"], reader="NumpyReader", image_only=True),]

train_transform_list = [
    SelectItemsd(keys=["image", 
                    #    'img_path', 'case_id', 'seq_id', 
                       'Repetition time', 'Echo time', 'Flip angle', 'part'], allow_missing_keys=False),
    OneOf([
        Orientationd(keys=["image"], axcodes="RAS"),
        Orientationd(keys=["image"], axcodes="RSA"),
        Orientationd(keys=["image"], axcodes="ARS"),
        Orientationd(keys=["image"], axcodes="ASR"),
        Orientationd(keys=["image"], axcodes="SRA"),
        Orientationd(keys=["image"], axcodes="SAR"),
    ]),
    RandAxisFlipd(keys=["image"], prob=0.5),
    SpatialPadd(keys=["image"], spatial_size=args.roi_size),
    RandSpatialCropd(keys=["image"], roi_size=args.roi_size, random_center=True, random_size=False),
    ToTensord(keys=["image"], track_meta=False),
]

val_transform_list = [
    SelectItemsd(keys=["image", 
                        #    'img_path', 'case_id', 'seq_id', 
                        'Repetition time', 'Echo time', 'Flip angle', 'part'], allow_missing_keys=False),
    SpatialPadd(keys=["image"], spatial_size=args.roi_size),
    RandSpatialCropd(keys=["image"], roi_size=args.roi_size, random_center=True, random_size=False),
    ToTensord(keys=["image"], track_meta=False),
]

def _get_one_dataset(datalist, transform):
    
    if args.cache_dataset:
        # print("Using MONAI Cache Dataset")
        train_ds = CacheDataset(data=datalist, transform=transform, cache_rate=0.5, num_workers=args.num_workers)
    elif args.smartcache_dataset:
        # print("Using MONAI SmartCache Dataset")
        train_ds = SmartCacheDataset(
            data=datalist,
            transform=transform,
            replace_rate=1.0,
            cache_num=2 * args.batch_size * args.sw_batch_size,
        )
    else:
        # print("Using generic dataset")
        train_ds = Dataset(data=datalist, transform=transform)


    return train_ds

def get_datasets(scale = args.ds):
    if args.rank == 0:
        print('************Get Data***************')
    assert scale in PRETRAIN_SCALE_MAP.keys(), f"Scale {scale} not found in PRETRAIN_SCALE_MAP"
    datasets = {}
    all_df = []
    for prefix, (datalist_func, _) in PRETRAIN_DS_MAP.items(): 
        datalist = datalist_func(scale, sample_size = PRETRAIN_SCALE_MAP[scale][prefix], )
        if args.rank == 0:
            print(f"Dataset {prefix} size: ", len(datalist))
        tmp_df = pd.DataFrame(datalist)
        if prefix == 'priv_cancer':
            tmp_df['part'] = tmp_df['cancer_type'].map(PART_CLASS_MAP[prefix])
        elif prefix == 'fastmri':
            tmp_df['part'] = tmp_df['type'].map(PART_CLASS_MAP[prefix])
        else:
            tmp_df['part'] = PART_CLASS_MAP[prefix]
        for k in DATA_KEYS.keys():
            if k not in tmp_df.columns:
                if args.rank == 0:
                    print(f"Warning: {k} not found in {prefix} dataset, using default value {DATA_KEYS[k]}")
                tmp_df[k] = DATA_KEYS[k]
        all_df.append(tmp_df)
    all_df = pd.concat(all_df, ignore_index=True)
    all_df = all_df[DATA_KEYS.keys()]

    train_df = all_df.sample(frac=0.9, random_state=42)
    val_df = all_df.drop(train_df.index).sample(frac=0.5, random_state=42)
    test_df = all_df.drop(train_df.index).drop(val_df.index)
    
    train_transform = Compose(before_load_transforms + loader + train_transform_list)
    train_ds = _get_one_dataset(train_df.to_dict(orient='records'), train_transform)
    val_transform = Compose(before_load_transforms + loader + val_transform_list)
    val_ds = _get_one_dataset(val_df.to_dict(orient='records'), val_transform)
    test_ds = _get_one_dataset(test_df.to_dict(orient='records'), val_transform)

    if args.rank == 0:
        print("Train dataset size: ", len(train_ds))
        print("Validation dataset size: ", len(val_ds))
        print("Test dataset size: ", len(test_ds))
        print('***************************')
    return train_ds, val_ds, test_ds

