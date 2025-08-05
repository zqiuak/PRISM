import SimpleITK as sitk
import numpy as np
import math
from run.options import kargs
import torch
import os, re


MetadataMap = {
    "StudyDescription":"0008|1030", 
    "SeriesDescription":"0008|103e", 
    "AcquisitionContrast":"0008|9209", 
    "PatientID":"0010|0020", 
    "ScanningSequence":"0018|0020", 
    "SequenceVariant":"0018|0021", 
    "ScanOptions":"0018|0022", 
    "SequenceName":"0018|0024", 
    "SliceThickness":"0018|0050", 
    "RepetitionTime":"0018|0080", 
    "EchoTime":"0018|0081", 
    "MagneticFieldStrength":"0018|0087", 
    "ProtocolName":"0018|1030", 
    "FlipAngle":"0018|1314",
    "PatientPosition":"0018|5100", 
    "PulseSequenceName":"0018|9005", 
    "SpectrallySelectedSuppression":"0018|9025", 
    "ImagePositionPatient":"0020|0032", 
    "ImageOrientationPatient":"0020|0037", 
    "PixelSpacing":"0028|0030"
    }





def check_sequence(folder):
    result_list = []
    has_valid_file = False
    for eachfile in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, eachfile)):
            result_list.extend(check_sequence(os.path.join(folder, eachfile)))

        elif eachfile.endswith(".jpg") or eachfile.endswith(".DS_Store"):
                continue
                
        else:
            has_valid_file = True
        
    if not has_valid_file:
        return result_list
    
    result_list.append(folder)
    return result_list

def get_dicom_info_in_dir(filepath, subject="") -> list:
    if len(os.listdir(filepath)) == 0 :
        return []
    
    infoDict_list = []
    
    has_valid_file = False
    # check any directory in the folder
    for eachfile in os.listdir(filepath): 
        if os.path.isdir(os.path.join(filepath, eachfile)):
            infoDict_list.extend( get_dicom_info_in_dir(os.path.join(filepath, eachfile), subject) )
        elif eachfile.endswith(".jpg") or eachfile.endswith(".DS_Store"):
            continue
        else:
            has_valid_file = True
            
    if not has_valid_file:
        return infoDict_list
    
    # find all dicom series
    seriesReader= sitk.ImageSeriesReader()
    seriesReader.LoadPrivateTagsOn()
    series_ids = seriesReader.GetGDCMSeriesIDs(filepath)
    for series_id in series_ids:
        series_file_names = seriesReader.GetGDCMSeriesFileNames(filepath, series_id)
        for eachfile in series_file_names:
            infoDict = get_dicom_info(os.path.join(filepath, eachfile))
            infoDict["subject"] = subject
            infoDict["series_id"] = series_id
            infoDict["folder"] = filepath
            if infoDict is not None:
                infoDict_list.append(infoDict)
                break

    return infoDict_list

def get_dicom_info(filepath)->dict:
    '''
    input: dicom path
    output: dicom info dict:{}

    '''
    try:
        reader = sitk.ImageFileReader()
        reader.LoadPrivateTagsOn()
        reader.SetFileName(filepath)
        reader.ReadImageInformation()
    except:
        return None
    infoDict = {
        "file":filepath.split("/")[-1],
        "subject": None,
        "series_id": None,
        "folder": None
        }

    for k, v in MetadataMap.items():
        try:
            infoDict[k] = reader.GetMetaData(v)
        except:
            infoDict[k] = "unknown"
    return infoDict

def parse_plane_and_protocol(SequenceDiscription:str, PatientOrientation:str):

    assert len(PatientOrientation) == 6, "PatientOrientation should be a tuple with 6 elements"

    
    if PatientOrientation == "unknown":
        return None
    
    patient_oirentation = [abs(round(float(x))) for x in PatientOrientation.split("\\")]
    if patient_oirentation == [1, 0, 0, 0, 1, 0]:
        seq_plane = "cor"
    elif patient_oirentation == [1, 0, 0, 0, 0, 1]:
        seq_plane = "sag"
    elif patient_oirentation == [0, 1, 0, 0, 0, 1]:
        seq_plane = "axi"
    else:
        Warning("Unknown patient orientation: ", PatientOrientation)
        return None
    
    seq_descript = SequenceDiscription.lower()
    seq_keywgs = re.split(r'[^a-zA-Z0-9]', seq_descript)
    for key in seq_keywgs:
        if key in ["ax", "sag", "cor", "axi", "sagittal", "coronal", "axial"]:
            seq_keywgs.remove(key)
    seq_keywgs.append(seq_plane)
    return set(seq_keywgs)

    # determine protocol
    # if "t1" in seq_descript:
    #     seq_protocol = "T1"
    # elif "t2" in seq_descript:
    #     seq_protocol = "T2"
    # elif "pd" in seq_descript:
    #     seq_protocol = "PDW"
    # elif "stir" in seq_descript:
    #     seq_protocol = "STIR"
    
def determine_plane_by_metadata(patient_orientation:str):
    image_ori = [float(x) for x in patient_orientation.split("\\")]
    # ref: https://stackoverflow.com/questions/70645577/translate-image-orientation-into-axial-sagittal-or-coronal-plane
    # image_ori = [round(x) for x in image_ori]
    # plane = np.cross(image_ori[0:3], image_ori[3:6])
    # plane = [abs(x) for x in plane]
    # if plane[0] == 1:
    #     return "sag"
    # elif plane[1] == 1:
    #     return "cor"
    # elif plane[2] == 1:
    #     return "axi"
    # else:
    #     return ""
    
    
    image_y = np.array([image_ori[0], image_ori[1], image_ori[2]])
    image_x = np.array([image_ori[3], image_ori[4], image_ori[5]])
    image_z = np.cross(image_x, image_y)
    abs_image_z = abs(image_z)
    main_index = list(abs_image_z).index(max(abs_image_z))
    if main_index == 0:
        return "sag"
    elif main_index == 1:
       return "cor"
    else:
        return "axi"
    

def whiting(img, MEAN, STDDEV, MAX_PIXEL_VAL):
        # img2 = (img - np.mean(img)) / np.std(img)
        img2 = (img - MEAN) / STDDEV
        img2 = ((img2 - np.min(img2)) / (np.max(img2) - np.min(img2)) * MAX_PIXEL_VAL)
        return img2.astype(np.uint8)
    
def read_dicom_series(filepath, re_spacing = None, normalize_func = None):
    r'''
    input: folder of dicom series
    re_spacing: privde (dx, dy, dz) to be target spacing

    output: ndarray
    '''
    
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(filepath)
    reader.SetFileNames(dicom_names)
    sitk_img = reader.Execute()

    if re_spacing is not None:
        # unify spacing
        ndx, ndy, ndz = re_spacing
        dx, dy, dz = sitk_img.GetSpacing()
        x, y, z = sitk_img.GetSize()
        nx, ny, nz = [math.ceil(x*dx/ndx), math.ceil(y*dy/ndy), math.ceil(z*dz/ndz)]
        # print("Image %s, original size: %d, %d, %d, spacing %3f, %3f, %3f"%(path, x, y, z, dx, dy, dz))
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing((ndx, ndy, ndz))
        resample.SetSize((nx, ny, nz))
        resample.SetOutputOrigin(sitk_img.GetOrigin())
        resample.SetOutputDirection(sitk_img.GetDirection())
        newImage = resample.Execute(sitk_img)
    else:
        newImage = sitk_img

    out_array = sitk.GetArrayFromImage(newImage)
    if normalize_func is not None:
        return normalize_func(out_array)
    else:
        return out_array/out_array.max()

def read_itk_img(path)->torch.Tensor:
    '''
    Read MRI series in given path, return the size as kargs.INPUT_DIM
    '''
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dicom_names)
    sitk_img = reader.Execute()

    # unify spacing
    ndx, ndy, ndz = kargs.Spacing
    dx, dy, dz = sitk_img.GetSpacing()
    x, y, z = sitk_img.GetSize()
    nx, ny, nz = [math.ceil(x*dx/ndx), math.ceil(y*dy/ndy), math.ceil(z*dz/ndz)]
    # print("Image %s, original size: %d, %d, %d, spacing %3f, %3f, %3f"%(path, x, y, z, dx, dy, dz))
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing((ndx, ndy, ndz))
    resample.SetSize((nx, ny, nz))
    resample.SetOutputOrigin(sitk_img.GetOrigin())
    resample.SetOutputDirection(sitk_img.GetDirection())
    newImage = resample.Execute(sitk_img)
    out_array = sitk.GetArrayFromImage(newImage)
    out_array = out_array/out_array.max()

    # CropResize
    SliceNum = kargs.SliceNum
    img_size = kargs.INPUT_DIM
    xl, xr = (nx//2-img_size//2, nx//2+img_size//2)
    yl, yr = (ny//2-img_size//2, ny//2+img_size//2)
    zl, zr = (nz//2-SliceNum//2, nz//2+SliceNum//2)
    pad_x = abs(xl) if xl<0 else 0
    pad_y = abs(yl) if yl<0 else 0
    pad_z = abs(zl) if zl<0 else 0
    out_array = np.pad(out_array, ((pad_z,pad_z),(pad_x,pad_x),(pad_y,pad_y)), mode="constant")
    out_array = out_array[zl+pad_z:zr+pad_z, xl+pad_x:xr+pad_x, yl+pad_y:yr+pad_y]
    out_tensor = torch.FloatTensor(out_array)
    return out_tensor


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def sample_by_case(df, case_size = None, seq_size = None):
    assert case_size is not None or seq_size is not None, "case_size or seq_size must be provided"
    case_list = df['case_id'].unique()
    if seq_size is not None:
        avg_seq_num = len(df) / len(case_list)
        case_num = int(seq_size // avg_seq_num)
        attempt = 0
        while attempt<10:
            # print('attempt', attempt)
            selected_case = np.random.choice(case_list, case_num, replace=False)
            selected_df = df[df['case_id'].isin(selected_case)]
            if abs(len(selected_df) - seq_size)/seq_size < 0.05:
                break
            if len(selected_df)< seq_size:
                case_num += 1
            else:
                case_num -= 1
            attempt += 1
    else:
        selected_case = np.random.choice(case_list, case_size, replace=False)
        selected_df = df[df['case_id'].isin(selected_case)]

    return selected_df

