'''
This file store path
'''

import os


# CodePath
CodePath = "./"
RootPath = CodePath
ResultPath = os.path.join(CodePath,"Result")
ExpFolder = os.path.join(ResultPath, "Exp")
DataRootPath = "./dataroot"
CachePath = "./cacheroot"
LabelFileDir = os.path.join(CodePath,"Labels/")

# Dataset Example:


################## ACDC Dataset #########################
class ACDC_Path_Class():
    def __init__(self):
        self.root = "./dataroot/ACDC/ACDC/database/"  # Root directory for ACDC dataset
        self.datapath = self.root  # training/ and testing/ subdirectories are here
        self.label_file = self.root  # No separate label CSVs; metadata in Info.cfg
        self.labelfilepath = os.path.join(LabelFileDir, "acdc/")
        self.seq_file = os.path.join(self.labelfilepath, "acdc_seqfile.csv")
        self.cache_dir = os.path.join(self.root, "cache/")
        self.sub_dataset = ["training", "testing"]
ACDC_Path = ACDC_Path_Class()
