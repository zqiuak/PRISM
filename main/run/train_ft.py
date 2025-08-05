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

from tqdm import tqdm

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.amp import GradScaler, autocast
from monai.metrics import CumulativeAverage
from run.options import HyperParams


from util.basicutils import AverageMeter, distributed_all_gather, save_ckpt

from monai.data import decollate_batch



class Trainer():
    def __init__(self, loaders, adapters, optimizer, scheduler, model, args:HyperParams):

        self.writer = None
        if args.logdir is not None and args.rank == 0:
            self.writer = SummaryWriter(log_dir=args.logdir)
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "Writing Tensorboard logs to ", args.logdir)

        scaler = None
        if args.amp:
            scaler = GradScaler("cuda")

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.loaders = loaders
        self.adapters = adapters
        self.args = args
        self.model = model
        self.prepare_data = adapters['prepare_data']
        self.infer_model = adapters['infer_model']
        self.loss_fn = adapters['loss_fn']
        self.cal_metric = adapters['cal_metric']
        self.task_metrics = adapters['task_metrics']



    def run(self, start_epoch=0, global_step=0):
        train_loader, val_loader, test_loader = self.loaders
        best_eval = 0
        writer = self.writer
        no_improve = 0
        for epoch in range(start_epoch, self.args.max_epochs):

            # train
            epoch_time = time.time()
            if self.args.distributed:
                train_loader.sampler.set_epoch(epoch)
                torch.distributed.barrier()

            train_loss, train_metric = self.train_epoch(train_loader, epoch)

            if self.args.rank == 0:
                print(
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    "Train: {}/{}".format(epoch, self.args.max_epochs - 1),
                    "loss: {:.4f}".format(train_loss),
                    "metric: {:.4f}".format(train_metric),
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("train_loss", train_loss, epoch)
                    writer.add_scalar("train_metric", train_metric, epoch)
                    if self.args.lrdecay:
                        writer.add_scalar("lr", self.scheduler.get_last_lr(), epoch)

                            
            # torch.cuda.empty_cache()
                


            # val
            if (epoch + 1) % self.args.evaluate_every == 0:
                epoch_time = time.time()
                if self.args.distributed:
                    torch.distributed.barrier()
                    
                val_loss, eval_metric, _ = self.val_epoch(val_loader, epoch, 'val')

                if self.args.rank == 0:
                    print(
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        "Validation  {}/{}".format(epoch, self.args.max_epochs - 1),
                        "loss: {:.4f}".format(val_loss),
                        "metric: {:.4f}".format(eval_metric),
                        "time {:.2f}s".format(time.time() - epoch_time),
                    )
                    if writer is not None:
                        writer.add_scalar("val_loss", val_loss, epoch)
                        writer.add_scalar("val_metric", eval_metric, epoch)
                        if self.args.lrdecay:
                            writer.add_scalar("lr", self.scheduler.get_last_lr(), epoch)

                

                # test 
                # torch.cuda.empty_cache()
                val_metric = eval_metric
                if val_metric > best_eval:
                    if self.args.rank == 0:
                        print(time.strftime("%Y-%m-%d %H:%M:%S"), "new best ({:.6f} --> {:.6f}). ".format(best_eval, val_metric))
                        save_ckpt(self.model, epoch, self.args, filename="best_eval.pt", optimizer=self.optimizer, scheduler=self.scheduler)
                    best_eval = val_metric
                    no_improve = 0
                else:
                    no_improve += 1

                test_loss, test_metric, metric_dict = self.val_epoch(test_loader, epoch, 'test')

                if self.args.rank == 0:
                    print(
                        "Test  {}/{}".format(epoch, self.args.max_epochs - 1),
                        "loss: {:.4f}".format(test_loss),
                        "metric: {:.4f}".format(test_metric),
                        "time {:.2f}s".format(time.time() - epoch_time),
                    )
                    if writer is not None:
                        writer.add_scalar("test_loss", test_loss, epoch)
                        writer.add_scalar("test_metric", test_metric, epoch)
                        for task in metric_dict.keys():
                            for k,v in metric_dict[task].items():
                                writer.add_scalar(f"{task}_{k}", v, epoch)
                # torch.cuda.empty_cache()

            # save ckpt
            if self.args.rank == 0:
                save_ckpt(self.model, epoch, self.args, filename="checkpoint.pt", optimizer=self.optimizer, scheduler=self.scheduler)

            if no_improve > self.args.patience:
                if self.args.rank == 0:
                    print(time.strftime("%Y-%m-%d %H:%M:%S"), "No improvement for {} epochs, stop training.".format(self.args.patience))
                break
                

    def train_epoch(self, loader, epoch):
        self.model.train()
        loss_metric = CumulativeAverage()
        task_metrics = self.task_metrics()

        with tqdm(loader, desc=f'train epoch {epoch} in gpu {self.args.gpu}', disable=self.args.gpu!=0) as pbar:
            
            for i, batch_data in enumerate(pbar):

                data, target = self.prepare_data(batch_data, 'train', self.args)
                
                with autocast('cuda', enabled=self.args.amp):
                    pred = self.infer_model(self.model, data, 'train', self.args)
                    loss = self.loss_fn(pred, target, 'train', self.args)
                loss_metric.append(loss)

                if self.args.amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (i+1) % self.args.grad_accumulate == 0:
                    if self.args.amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        if self.args.grad_clip:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                        self.optimizer.step()
                        
                    self.optimizer.zero_grad()

                task_metrics, eval_metric = self.cal_metric(pred, target, 'train', task_metrics, self.args)
                    
                if self.args.debug:
                    break
                
        self.optimizer.zero_grad()
        if self.args.lrdecay:
            self.scheduler.step()


        # aggregate and print metrics
        metric_dict, eval_metric = self.cal_metric(None, None, 'train', task_metrics, self.args, aggregate=True)
        if self.args.rank == 0:
            print(time.strftime("%Y-%m-%d %H:%M:%S"), metric_dict)

        loss = loss_metric.aggregate().item()
        return loss, eval_metric
        
        
    @torch.no_grad()
    def val_epoch(self, loader, epoch, mode = 'val'):
        self.model.eval()
        loss_metric = CumulativeAverage()
        task_metrics = self.task_metrics()

        with tqdm(loader, desc=f'{mode} epoch {epoch} in gpu {self.args.gpu}', disable=self.args.gpu!=0) as pbar:
            for batch_data in pbar:
                data, target = self.prepare_data(batch_data, mode, self.args)
                # print('prepare data done')
                with autocast('cuda', enabled=self.args.amp):
                    pred = self.infer_model(self.model, data, mode, self.args)
                    # print('infer model done')
                    loss = self.loss_fn(pred, target, mode, self.args)
                    # print('loss_fn done')
                loss_metric.append(loss.detach().cuda())

                task_metrics, eval_metric = self.cal_metric(pred, target, mode, task_metrics, self.args)
                if self.args.debug:
                    break

        # aggregate and print metrics
        metric_dict, eval_metric = self.cal_metric(None, None, mode, task_metrics, self.args, aggregate=True)
        if self.args.rank == 0:
            print(time.strftime("%Y-%m-%d %H:%M:%S"), metric_dict)

        loss = loss_metric.aggregate().item()
        return loss, eval_metric, metric_dict

    