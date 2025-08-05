# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pdb
import shutil
import time
import json

from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from torch.amp import GradScaler, autocast
from monai.metrics import CumulativeAverage
from run.options import HyperParams
from main.util.ci import compute_ci_from_dict


from util.basicutils import AverageMeter, distributed_all_gather, save_ckpt

from monai.data import decollate_batch
import pickle


def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    elif isinstance(x, list):
        return [to_cpu(i) for i in x]
    elif isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    else:
        return x

class Inferer():
    def __init__(self, loaders, adapters, model, args:HyperParams):


        self.loaders = loaders
        self.adapters = adapters
        self.args = args
        self.model = model
        self.prepare_data = adapters['prepare_data']
        self.infer_model = adapters['infer_model']
        self.loss_fn = adapters['loss_fn']
        self.cal_metric = adapters['cal_metric']
        self.task_metrics = adapters['task_metrics']

    def run(self):
        _, _, test_loader = self.loaders

        epoch_time = time.time()
                
        # test 

        test_loss, test_metric, metric_dict = self.val_epoch(test_loader, 'test')

        if self.args.rank == 0:
            print(
                "loss: {:.4f}".format(test_loss),
                "metric: {:.4f}".format(test_metric),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        # torch.cuda.empty_cache()


                
        
    @torch.no_grad()
    def val_epoch(self, loader, mode = 'val'):
        self.model.eval()
        loss_metric = CumulativeAverage()
        task_metrics = self.task_metrics()

        all_metrics = []
        outputs = []
        stack_size = 0
        with tqdm(loader, desc=f'{mode}', disable=self.args.gpu!=0) as pbar:
            for batch_data in pbar:
                data, target = self.prepare_data(batch_data, mode, self.args)
                # print('prepare data done')
                with autocast('cuda', enabled=self.args.amp):
                    pred = self.infer_model(self.model, data, mode, self.args)
                    # print('infer model done')
                    loss = self.loss_fn(pred, target, mode, self.args)
                    # print('loss_fn done')
                try:
                    if stack_size < 20:
                        outputs.append({'data': to_cpu(data), 'label': to_cpu(target), 'pred': to_cpu(pred)})
                        stack_size += 1
                except:
                    pass
                loss_metric.append(loss.detach().cuda())

                task_metrics, batch_metric = self.cal_metric(pred, target, mode, task_metrics, self.args)

                assert isinstance(batch_metric, dict), "batch_metric should be a dict"
                all_metrics.append(batch_metric)

                if self.args.debug:
                    break
        
        with open(os.path.join(self.args.logdir, f'{mode}_outputs.pkl'), 'wb') as f:
            pickle.dump(outputs[:10], f, protocol=pickle.HIGHEST_PROTOCOL,)

        # save metrics
        json.dump(all_metrics, open(os.path.join(self.args.logdir, f'{mode}_metrics.json'), 'w'), indent=4)
        # save metrics to csv
        df = pd.DataFrame(all_metrics)
        df.to_csv(os.path.join(self.args.logdir, f'{mode}_metrics.csv'), index=True)

        results_dir = self.args.logdir.replace('Exp/INFER', 'Results')
        results_dir = os.path.dirname(results_dir)
        a_name = self.args.logdir.split('-')[-1]
        os.makedirs(results_dir, exist_ok=True)
        df.to_csv(os.path.join(results_dir, f'{a_name}_metrics.csv'), index=True)

        # compute ci
        ci_dict = compute_ci_from_dict(all_metrics)
        json.dump(ci_dict, open(os.path.join(self.args.logdir, f'{mode}_ci.json'), 'w'), indent=4)

        # aggregate and print metrics
        metric_dict, eval_metric = self.cal_metric(None, None, mode, task_metrics, self.args, aggregate=True)
        if self.args.rank == 0:
            print(time.strftime("%Y-%m-%d %H:%M:%S"), metric_dict)

        loss = loss_metric.aggregate().item()
        return loss, eval_metric, metric_dict


