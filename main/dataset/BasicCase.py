
import numpy as np
import pickle
import os
from torch.utils import data


from util.basicutils import *
from dataset.datautils import parse_plane_and_protocol



class MultiSequenceCase(data.Dataset):
    def __init__(self, datalist, 
                 sequence_transform,
                 other_transform = None,
                 seq_list_key = 'seq_list',
                 out_key = 'image_list',
                 *args, **kwargs):
        super().__init__()
        self.datalist = datalist
        self.sequence_transform = sequence_transform
        self.other_transform = other_transform
        self.seq_list_key = seq_list_key
        self.out_key = out_key


    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        data = self.datalist[idx]
        img_list = []
        for each in data[self.seq_list_key]:
            img = self.sequence_transform(each)
            if isinstance(img, list):
                img_list.extend(img)
            else:
                img_list.append(img)    
        data[self.out_key] = img_list
        if self.other_transform is not None:
            data = self.other_transform(data)
        return data
    


    

class BasicMultisequenceCase():
    r"""
    Basic Case class, define a simple patient case with multiple MRI sequence
    """
    def __init__(self, 
                 dataset_name:str,
                 case_id:str,
                 seq_name_list,
                 seq_path_list,
                 seq_loader_func,
                 cache_path:str = None,
                 cls_label:str = None,
                 *arg, **kwargs) -> None:
        self.dataset_name = dataset_name
        self.id = case_id
        self.seq_name_list = seq_name_list
        self.seq_path_list = seq_path_list
        self.seq_loader_func = seq_loader_func
        self.sequence_dict = {seq_id:None for seq_id in seq_name_list}
        self.cls_label = cls_label

        self.is_loaded = False

        # other info
        self.other_info = None

        # cache
        self.cache_path = cache_path
        if cache_path is not None:
            if not os.path.exists(cache_path):
                os.makedirs(cache_path, exist_ok=True)
            self.cache_file = os.path.join(cache_path, f"cache_case_{self.id}.pkl")
            self.cached_attr_list = ["sequence_dict", "other_info"]
        else:
            self.cache_file = None


    def save(self, overwrite = False):
        self._save_cache(overwrite)

    def load(self):
        try:
            self._load_cache()
        except:
            self._load_sequence()
        self.is_loaded  = True

    def flush_cache(self):
        self._load_sequence()
        self._save_cache(overwrite=True)

    def _load_sequence(self):
        for i, seq_name in enumerate(self.seq_name_list):
            seq_data = self.seq_loader_func(self, i)
            self.sequence_dict[seq_name] = Single_Sequence(sequence_id=seq_name, **seq_data)

    def get_sequence_by_filter(self, filter:None|str = None):
        result_list = []
        for seq in self.sequence_dict.values():
            if filter in seq.keywords or filter is None:
                result_list.append(seq)
        return result_list
    
    def __len__(self):
        return len(self.sequence_dict.keys())

    def _save_cache(self, overwrite):
        if self.cache_file is None:
            raise ValueError("Cache path is not defined")
        if os.path.exists(self.cache_file) and not overwrite:
            return
        cache_buffer = {}
        for attr_name in self.cached_attr_list:
            cache_buffer[attr_name] = getattr(self, attr_name)
        with open(self.cache_file, 'wb+') as outfile:
            pickle.dump(cache_buffer, outfile)

    def _load_cache(self):
        if self.cache_file is None:
            raise ValueError("Cache path is not defined")
        with open(self.cache_file, "rb") as infile:
            chache_buffer = pickle.load(infile)
        self.__dict__.update(chache_buffer)
        
    def __getitem__(self, indx):
        return self.sequence_dict[self.seq_name_list[indx]]
    
    def __get_by_id__(self, seq_id):
        return self.sequence_dict[seq_id]
            

class Single_Sequence():
    def __init__(self, sequence_id, img_data, info_dict = None, mask_data=None, cls_label=None):
        self.img = img_data
        self.mask = mask_data
        self.cls_label = cls_label
        self.id = sequence_id
        self.info = info_dict
        if info_dict is not None:
            if "series_description" in info_dict.keys():
                self.keywords = parse_plane_and_protocol(info_dict["series_description"])
            else:
                self.keywords = None
            if "series_id" in info_dict.keys():
                self.series_id = info_dict["series_id"]
            else:
                self.series_id = None
            if "ImageOrientationPatient" in info_dict.keys():
                self.plane = info_dict["ImageOrientationPatient"]
            else:
                self.plane = None

    def apply_aug(self, aug_func):
        self.img = aug_func(self.img)
        if self.mask is not None:
            self.mask = aug_func(self.mask)

    def __call__(self, *args, **kwds):
        return self.img
