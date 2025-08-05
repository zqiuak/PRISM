'''
statistical analysis utilities
'''


import numpy as np
from tqdm import tqdm
from scipy import stats

def compute_ci(datalist, alpha=0.05, num_samples=10000):

    data = np.array(datalist)
    n = len(data)
    
    bootstrap_stats = np.zeros(num_samples)
    
    for i in range(num_samples):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats[i] = np.mean(sample)
    
    lower = np.percentile(bootstrap_stats, 100 * alpha/2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha/2))
    mean = np.mean(bootstrap_stats)
    
    return (mean, lower, upper)

def compute_ci_from_dict(MetricDict_List:list, keys = None, *targs, **kwargs):
    '''
    MetricDict_List should be a list of dicts, each dict contains the metrics for one subject
    '''
    if keys is None:
        keys = MetricDict_List[0].keys()
    result_dict = {}
    for k in keys:
        datalist = [x[k] for x in MetricDict_List]
        datalist = [x for x in datalist if not np.isnan(x)]
        result_dict[k] = compute_ci(datalist, *targs, **kwargs)

    return result_dict
