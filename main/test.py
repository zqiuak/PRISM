
from run.pathdict import *
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from time import time
import torch


import multiprocessing as mp
from dataset.pretrain.ds import get_datasets
train_ds, val_ds, test_ds = get_datasets()


import time
import torch
import multiprocessing as mp
print(f"num of CPU: {mp.cpu_count()}")
from tqdm import tqdm
for num_workers in range(0, mp.cpu_count(), 2):  
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, num_workers=num_workers, batch_size=8, pin_memory=True)
    start = time.time()
    for epoch in range(0, 1):
        step = 0
        for i, data in enumerate(tqdm(train_loader, total=100)):
            # print("epoch: {}, batch: {}".format(epoch, i))
            step += 1
            if step > 100:
                break
    del train_loader
    end = time.time()
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))

exit()


from dataset.adni.ds import create_cache
create_cache()


exit()
 
from torch.utils.data import DataLoader
import SimpleITK as sitk
import logging
import os


class SimpleITKLogger(sitk.LoggerBase):
    """
    Adapts SimpleITK messages to be handled by a Python Logger object.

    Allows using the logging module to control the handling of messages coming
    from ITK and SimpleTK. Messages such as debug and warnings are handled by
    objects derived from sitk.LoggerBase.

    The LoggerBase.SetAsGlobalITKLogger method must be called to enable
    SimpleITK messages to use the logger.

    The Python logger module adds a second layer of control for the logging
    level in addition to the controls already in SimpleITK.

    The "Debug" property of a SimpleITK object must be enabled (if
    available) and the support from the Python "logging flow" hierarchy
    to handle debug messages from a SimpleITK object.

    Warning messages from SimpleITK are globally disabled with
    ProcessObject:GlobalWarningDisplayOff.
    """

    def __init__(self, logger: logging.Logger = logging.getLogger("SimpleITK")):
        """
        Initializes with a Logger object to handle the messages emitted from
        SimpleITK/ITK.
        """
        super(SimpleITKLogger, self).__init__()
        self._logger = logger

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger):
        self._logger = logger

    def __enter__(self):
        self._old_logger = self.SetAsGlobalITKLogger()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._old_logger.SetAsGlobalITKLogger()
        del self._old_logger

    def DisplayText(self, s):
        return
        # Remove newline endings from SimpleITK/ITK messages since the Python
        # logger adds during output.
        self._logger.info(s.rstrip())

    def DisplayErrorText(self, s):
        self._logger.error(s.rstrip())

    def DisplayWarningText(self, s):
        return
        self._logger.warning(s.rstrip())

    def DisplayGenericOutputText(self, s):
        return
        self._logger.info(s.rstrip())

    def DisplayDebugText(self, s):
        return
        self._logger.debug(s.rstrip())


# Enable all debug messages for all ProcessObjects, and procedures
sitk.ProcessObject.GlobalDefaultDebugOn()

# Construct a SimpleITK logger to Python Logger adaptor
sitkLogger = SimpleITKLogger()

# Configure ITK to use the logger adaptor
sitkLogger.SetAsGlobalITKLogger()

import sys
# Configure the Python root logger, enabling debug and info level messages.
logging.basicConfig(stream=sys.stdout,format="%(name)s (%(levelname)s): %(message)s", level=logging.DEBUG)

with SimpleITKLogger(logging.getLogger("Show")) as showLogger:
        
    df = pd.read_csv(private_knee_ft_seqfile)
    dl = cache_ds(df)
    # dl = DataLoader(cache_ds(df), batch_size=1, num_workers=20, collate_fn=lambda x:x, shuffle=False)
    for i, _ in enumerate(tqdm(dl)):
        if (i % 200 == 0):
            dl._save()
    dl._save()
exit()
