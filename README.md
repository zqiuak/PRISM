# Large-scale Multi-sequence Pretraining for Generalizable MRI Analysis in Versatile Clinical Applications

# Dataset structure

This folder contains the dataset used for training and evaluating the model. The dataset is organized into the following structure:

```
dataset name /
├── downloaded_code/
├── ds.py
├── socks.py
├── (ds_task1.py) 
├── (socks_task1.py) 
...

```
- `downloaded_code/`: This folder contains the code provided by the data provider. For example, it may contain the code for downloading the dataset or any other pre-processing steps.
- `ds.py`: This file contains the code for the dataset class. It is responsible for loading and processing the dataset. Details are provided below in the [ds.py](#dspy) section.
- `socks.py`: This file contains the code for training/validation. It is responsible for processing the batched data, model define, model inference, loss calculation, metrics calculation, etc. Details are provided below in the [socks.py](#sockspy) section.
- (Optional) `ds_taskX.py` and `socks_taskX.py`: If this dataset contains multiple tasks, the task-specific files are named accordingly. For example, if a dataset contains classification and segmentation tasks, it will have `ds_cls.py`, `ds_seg.py`, `socks_cls.py`, and `socks_seg.py`. 


## `ds.py`
This file is responsible for loading and processing the dataset.
If the dataset is used in pre-training, it will contain the following classes and functions:
- `get_pt_list()`: Appears when the dataset is used in pre-training. This function will return a list of data. It takes the following parameters:
  - `datascale`:str. The scale of the data. It can be `10k`, `50k`, `300k`, etc., used to identify the cached list.
  - `sample_size`:int. Actually how many samples are expected to draw from the dataset. If sample_size=-1, it means all the samples in the dataset will be used. 
  - `keys`:list. The keys of the data. Usually contains 'case_id', 'seq_id', 'image', etc.
- `pt_seq_loader`: This is usually a list of objects of the `monai.transforms.Compose` class. It is used to load the sequence.

If the dataset is used in down-stream tasks, it will contain the following classes and functions:
- `get_ft_ds()`: Appears when this dataset is used in down-stream tasks. This function is responsible for loading the dataset. We use `util.datautils.get_downstream_socks()` to get the dataset from this function.

Usually, this file contains the following helper functions and variables:
- `DATASET_PREFIX`: The prefix used for the dataset. It is used to identify the dataset in the code.
- `original_seq_loader`: This is usually a object of the `monai.transforms.Compose` class. It is used to load the original sequence of the dataset and pre-process it, commonly used in the `create_cache()` function.
- `create_cache()`: It is used for creating the cache for the dataset.
- `get_ft_list()`: To get the list of files in the dataset. Usually it returns train_list, val_list, test_list to serve the `get_ft_ds()`. Whether the list is sequence-based or case-based depends on the dataset.
- `train_transforms`, `val_transforms`: These are usually objects of the `monai.transforms.Compose` class. They are used to pre-process the data before feeding it into the model. Used in the `get_ft_ds()` function.

## `socks.py`
This file is responsible for training and validating the model. Usually it does contain the components that are used in the training and validation process. It is usually contains the following classes and functions:
- `FT_MRI_FM`: The model class. 
- `get_task_metrics()`: This function is used to get the evaluation metrics, it returns a dictionary in the format of {taskname: {metricname: metric class in `monai.metrics`}}.
- `prepare_data()`: Processing the batched data. It returns the data and label.
- `infer_model()`: How to infer the model with given data from the `prepare_data()` function. It returns the output of the model.
- `loss_fn()`: The loss calculation function. It returns the loss value.
- `cal_metric()`: The metric calculation function. If aggregate is False, it append the metric with given prediction and label to the `monai.metrics` class. If aggregate is True, it returns the aggregated metric. It also returns the value of current metric.

