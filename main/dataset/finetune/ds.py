import os
import sys
import pandas as pd
import re
from tqdm import tqdm
import json
import numpy as np
from run.pathdict import ACDC_Path
from run.options import kargs as args

from monai.transforms import *
from monai.data import NibabelReader, Dataset
from dataset.datautils import sample_by_case

DATASET_PREFIX = "acdc"
INPUT_SIZE = (32,192,192)
META_COLUMNS = ['Weight', 'Height', 'ED_frame', 'ES_frame']
EXCLUDE_PATIENTS = []


def _load_info_cfg(path):
    """Read metadata from Info.cfg file."""
    infodict = {}
    if os.path.exists(path):
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                k, v = line.strip().split(': ')
                infodict[k] = v
    return infodict

def _find_nii_in_folder(folder_path):
    """Find the first .nii file in a folder."""
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.endswith('.nii'):
                return os.path.join(root, f)
    return None

def create_seqfile(force_overwrite=False, output_path='./Labels/acdc_seqfile.csv'):
    """Create a CSV file listing all sequences and metadata, including both training and testing datasets."""
    print(f"Will save to: {output_path}")

    if os.path.exists(output_path) and not force_overwrite:
        print(f"{output_path} already exists, skipping creation.")
        return

    datasets = ['training', 'testing']
    seq_df = pd.DataFrame()
    case_id_index = 0

    for dataset in datasets:
        dataset_path = os.path.join(ACDC_Path.root, dataset)
        if not os.path.exists(dataset_path):
            print(f"Error: {dataset_path} not found")
            continue

        patient_dirs = sorted([d for d in os.listdir(dataset_path) if d.startswith('patient')])
        # print(f"Found {len(patient_dirs)} patients in {dataset} dataset")  # Debug

        for patient_id in tqdm(patient_dirs, desc=f"Processing {dataset}"):
            patient_path = os.path.join(dataset_path, patient_id)
            patient_number = patient_id.replace('patient', '')
            case_id = f"{DATASET_PREFIX}_{case_id_index}"
            case_id_index += 1
            infodict = _load_info_cfg(os.path.join(patient_path, 'Info.cfg'))

            # Get all frame entries (files only, not ground-truth)
            frame_entries = sorted([f for f in os.listdir(patient_path) if '_frame' in f and not '_gt' in f and f.endswith('.nii')])
            if not frame_entries:
                print(f"Warning: No frame entries found for {patient_id} in {dataset}")
                continue

            for frame_entry in frame_entries:
                frame_path = os.path.join(patient_path, frame_entry)
                try:
                    frame_id = frame_entry.split('_frame')[1].split('.')[0]
                except IndexError:
                    print(f"Error: Invalid frame_entry format: {frame_entry}")
                    continue

                # Flexible image path handling
                image_path = None
                if os.path.isfile(frame_path) and frame_path.endswith('.nii'):
                    image_path = frame_path
                elif os.path.isdir(frame_path) and frame_path.endswith('.nii'):
                    # Look for a .nii file inside the directory
                    found_nii = _find_nii_in_folder(frame_path)
                    if found_nii:
                        image_path = found_nii
                    else:
                        print(f"Warning: No .nii file found inside directory {frame_path}")
                        continue
                else:
                    print(f"Warning: Invalid frame_entry {frame_path}")
                    continue

                gt_entry = frame_entry.replace('.nii', '_gt.nii')
                gt_path_full = os.path.join(patient_path, gt_entry)
                gt_path = None
                if os.path.isfile(gt_path_full) and gt_path_full.endswith('.nii'):
                    gt_path = gt_path_full
                elif os.path.isdir(gt_path_full) and gt_path_full.endswith('.nii'):
                    found_gt = _find_nii_in_folder(gt_path_full)
                    if found_gt:
                        gt_path = found_gt
                    else:
                        print(f"Warning: No .nii file found inside GT directory {gt_path_full}")
                else:
                    if os.path.exists(gt_path_full):
                        print(f"Warning: GT entry exists but is not a valid .nii file or directory: {gt_path_full}")

                row = {
                    'case_id': case_id,
                    'seq_id': f"{case_id}_frame{frame_id}",
                    'image': image_path,
                    'label': gt_path,
                    'dataset': dataset,
                    **{k: infodict.get(k, 0.0 if k in ['Weight', 'Height'] else 0) for k in META_COLUMNS}
                }
                seq_df = pd.concat([seq_df, pd.DataFrame([row])], ignore_index=True)

    # Verify dataset inclusion
    print("Dataset counts:")
    print(seq_df['dataset'].value_counts())

    seq_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")



################## downstream tasks ##################

train_transforms = Compose([
    LoadImaged(keys=["image", "label"], reader='NibabelReader', ensure_channel_first=True),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    Transposed(keys=["image", "label"], indices=[0, 3, 1, 2]),
    CropForegroundd(
        keys=["image", "label"], source_key="image", k_divisible=INPUT_SIZE
    ),
    SpatialPadd(keys=["image", "label"], spatial_size=INPUT_SIZE),

    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=INPUT_SIZE,
        pos=1,
        neg=1,
        num_samples=args.sample_size,
        image_key="image",
        image_threshold=0,
    ),
    RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=2),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"], reader='NibabelReader', ensure_channel_first=True),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    
    Transposed(keys=["image", "label"], indices=[0, 3, 1, 2]),
    CropForegroundd(
        keys=["image", "label"], source_key="image", k_divisible=INPUT_SIZE
    ),
    SpatialPadd(keys=["image", "label"], spatial_size=INPUT_SIZE,
                mode="constant"),
])

def _to_dataset(data, transform, keys):
    """Create a MONAI Dataset for downstream tasks."""
    # Filter data to include only specified keys
    filtered_data = [
        {k: d[k] for k in keys if k in d and pd.notnull(d[k])}
        for d in data
    ]
    return Dataset(data=filtered_data, transform=transform)

def _to_cache_path(inpath):
    folder = os.path.dirname(inpath)
    filename = os.path.basename(inpath)
    folder = folder.replace(ACDC_Path.datapath, ACDC_Path.cache_dir)
    filename = filename.replace('.nii', '.npy')
    return os.path.join(folder, filename)


def get_ft_list(use_cache = False):
    """Get lists for training, validation, and testing."""
    seq_file = ACDC_Path.seq_file
    seq_pd = pd.read_csv(seq_file)
    seq_pd = seq_pd[~seq_pd['case_id'].isin(EXCLUDE_PATIENTS)]
    if use_cache:
        seq_pd['image'] = seq_pd['image'].apply(_to_cache_path)
        seq_pd['label'] = seq_pd['label'].apply(_to_cache_path)

    train_list = seq_pd[seq_pd['dataset'] == 'training'].to_dict('records')
    
    train_num = int(len(train_list)*0.8)  # Use 80% of training data for training
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(train_list))
    train_indices = indices[:train_num]
    val_indices = indices[train_num:]
    val_list = [train_list[i] for i in val_indices]
    train_list = [train_list[i] for i in train_indices]
    test_list = seq_pd[seq_pd['dataset'] == 'testing'].to_dict('records')
    
    return train_list, val_list, test_list

def get_ft_ds():
    """Get dataset for downstream tasks."""
    train_list, val_list, test_list = get_ft_list()
    
    train_ds = _to_dataset(
        data=train_list,
        transform=train_transforms,
        keys=['image', 'label'] 
    )
    val_ds = _to_dataset(
        data=val_list,
        transform=val_transforms,
        keys=['image', 'label']
    )
    test_ds = _to_dataset(
        data=test_list,
        transform=val_transforms,
        keys=['image', 'label']   # Labels may be absent in test set
    )
    
    return train_ds, val_ds, test_ds