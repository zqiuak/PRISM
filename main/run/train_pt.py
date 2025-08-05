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
from monai.data import decollate_batch
from model.model_pt import PT_MRI_FM

from run.options import HyperParams
from run.lr_scheduler import WarmupCosineSchedule
from util.basicutils import AverageMeter, distributed_all_gather, save_ckpt, set_req_grad
from dataset.pretrain.socks import Recon_Loss, Meta_Loss, Part_Loss, Contrast_Loss, Disc_real_loss,Disc_fake_loss, G_nonsaturating_loss


class Trainer():
    def __init__(self, loaders, adapters, optimizer, scheduler, model:PT_MRI_FM, args:HyperParams):

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

        # init optimizer for discriminator and generator
        if self.args.distributed:
            self.G = self.model.module.G
            self.D = self.model.module.D
        else:
            self.G = self.model.G
            self.D = self.model.D

        self.optimizer_G = torch.optim.AdamW(self.G.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.optimizer_D = torch.optim.AdamW(self.D.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.lrdecay:
            self.scheduler_G = WarmupCosineSchedule(self.optimizer_G, warmup_steps=args.warmup_steps, t_total=args.num_steps)
            self.scheduler_D = WarmupCosineSchedule(self.optimizer_D, warmup_steps=args.warmup_steps, t_total=args.num_steps)
        
        if args.amp:
            self.scaler_G = GradScaler("cuda")
            self.scaler_D = GradScaler("cuda")

        self.eval_every = 500
        self.last_eval = 0
        
        self.loss_weight = self.args.model_config.loss_weights

    def run(self, start_epoch=0, global_step=0):
        train_loader, val_loader, test_loader = self.loaders
        best_eval = 0
        writer = self.writer
        no_improve = 0
        self.global_step = global_step

        for epoch in range(start_epoch, self.args.max_epochs):
            if self.global_step > self.args.num_steps:
                break

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
                    writer.add_scalar("train_metric", train_metric, epoch)

                            
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

                

            # test 
                # torch.cuda.empty_cache()
                val_metric = eval_metric
                if val_metric > best_eval:
                    if self.args.rank == 0:
                        print(time.strftime("%Y-%m-%d %H:%M:%S"), "new best ({:.6f} --> {:.6f}). ".format(best_eval, val_metric))
                        save_ckpt(self.model, epoch, self.args, filename="best_eval.pt", optimizer=self.optimizer, scheduler=self.scheduler, global_step=self.global_step)
                    best_eval = val_metric
                    no_improve = 0
                else:
                    no_improve += 1


                test_loss, test_metric, metric_dict = self.val_epoch(test_loader, epoch, 'test')

                if self.args.rank == 0:
                    print(
                        time.strftime("%Y-%m-%d %H:%M:%S"),
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
                save_ckpt(self.model, epoch, self.args, filename="checkpoint.pt", optimizer=self.optimizer, scheduler=self.scheduler, global_step=self.global_step)


            if no_improve > self.args.patience:
                
                print(time.strftime("%Y-%m-%d %H:%M:%S"), "Early stopping at epoch {}".format(epoch))
                break
                

    def train_epoch(self, loader, epoch):
        self.model.train()
        loss_metrics = {'all': CumulativeAverage()}
        task_metrics = self.task_metrics()

        with tqdm(loader, desc=f'train epoch {epoch} in gpu {self.args.gpu}', disable=self.args.gpu!=0) as pbar:
            for batch_data in pbar:
                data, target = self.prepare_data(batch_data, 'train', self.args)
                real_img, masked_img = data
                _, meta_label, part_label = target
            
                #*****************train recon and meta pred, old training
                if True:
                    self.optimizer.zero_grad()
                    set_req_grad(self.model, True)
                    
                    with autocast('cuda', enabled=self.args.amp):
                        recon_img, meta_part_pred = self.model('rm', real_img, masked_img)
                        pred = (recon_img, None, meta_part_pred, None)

                        losses = {'recon': Recon_Loss(recon_img, real_img)}

                        meta_pred, part_pred = meta_part_pred
                        if self.args.model_config.pred_meta:
                            losses['meta'] = Meta_Loss(meta_pred, meta_label)
                        if self.args.model_config.pred_part:
                            losses['part'] = Part_Loss(part_pred, part_label)

                        loss = sum([self.loss_weight[k] * v for k, v in losses.items()])
                            

                    loss_metrics['all'].append(loss.detach())
                    for k, v in losses.items():
                        if not k in loss_metrics:
                            loss_metrics[k] = CumulativeAverage()
                        loss_metrics[k].append(v.detach())

                    if self.args.amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()         
                        for name, param in self.model.named_parameters():
                            if param.requires_grad and param.grad is None:
                                print(f'{name} grad is None.')       
                        if self.args.grad_clip:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                        self.optimizer.step()
                        
                    self.optimizer.zero_grad()

                    task_metrics, eval_metric = self.cal_metric(pred, target, 'train', task_metrics, self.args)


                    
                # ********************Train discriminator and meta pred for recon
                if self.args.model_config.discriminate:

                    self.optimizer_D.zero_grad()
                    set_req_grad(self.D, True)
                    set_req_grad(self.G, False)
                    # if self.args.model_config.contrastive:
                        # set_req_grad(self.D['seq_contrastive_head'], False)
                        # set_req_grad(self.D['str_contrastive_head'], False)
                    with autocast('cuda', enabled=self.args.amp):
                        real_disc_pred, meta_part_pred, _, recon_disc_pred = self.model('rd', real_img, masked_img, detach_D = True)

                        meta_pred, part_pred = meta_part_pred

                        d_losses ={'disc': self.loss_weight['disc'] * Disc_real_loss(real_disc_pred) +  self.loss_weight['disc'] * Disc_fake_loss(recon_disc_pred)}

                        if self.args.model_config.pred_meta:
                            d_losses['meta'] = self.loss_weight['meta'] * Meta_Loss(meta_pred, meta_label)
                        if self.args.model_config.pred_part:
                            d_losses['part'] = self.loss_weight['part'] * Part_Loss(part_pred, part_label)


                        if self.args.model_config.synthesize:
                            _, _, _, fake_disc_pred, _ = self.model('sd', real_img, detach_D = True)
                            d_losses['disc'] += self.loss_weight['disc'] *Disc_fake_loss(fake_disc_pred)
                    
                    d_loss = 0.5 * sum([self.loss_weight[k] * v for k, v in d_losses.items()])


                    if self.args.amp:
                        self.scaler_D.scale(d_loss).backward()
                        self.scaler_D.step(self.optimizer_D)
                        self.scaler_D.update()
                    else:
                        d_loss.backward()
                        if self.args.grad_clip:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                        self.optimizer_D.step()
                        if self.args.lrdecay:
                            self.scheduler_D.step()
                    self.optimizer_D.zero_grad()
                            

                #***************** Train generator for recon
                if self.args.model_config.discriminate and self.global_step % 10 == 0:

                    self.optimizer_G.zero_grad()
                    set_req_grad(self.D, False)
                    set_req_grad(self.G, True)
                    with autocast('cuda', enabled=self.args.amp):
                            
                        # recon
                        _, _, recon_img, recon_disc_pred = self.model('rd', real_img, masked_img)
                        g_loss = self.loss_weight['disc'] * G_nonsaturating_loss(recon_disc_pred) \
                            +  self.loss_weight['recon'] * Recon_Loss(recon_img, real_img)

                        # synthesize
                        if self.args.model_config.synthesize:
                            fake_img, real_contrast_f, _, fake_disc_pred, fake_contrast_f = self.model('sd', real_img)
                            g_loss += self.loss_weight['disc'] * G_nonsaturating_loss(fake_disc_pred)

                            if self.args.model_config.contrastive:
                                g_loss += self.loss_weight['cont'] * Contrast_Loss(real_contrast_f, fake_contrast_f)

                        g_loss = 0.2 * g_loss

                        if self.args.amp:
                            self.scaler_G.scale(g_loss).backward()
                            self.scaler_G.step(self.optimizer_G)
                            self.scaler_G.update()
                        else:
                            g_loss.backward()
                            if self.args.grad_clip:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                            self.optimizer_G.step()
                            if self.args.lrdecay:
                                self.scheduler_G.step()
                        self.optimizer_G.zero_grad()

        
                if self.args.lrdecay:
                    self.scheduler.step()
                self.global_step += 1

                # ************** log
                if self.global_step % self.args.log_every == 0:
                    record_losses = {k: v.aggregate().item() for k, v in loss_metrics.items()}
                    if self.args.rank == 0:

                        save_ckpt(self.model, epoch, self.args, filename="checkpoint.pt", optimizer=self.optimizer, scheduler=self.scheduler, global_step=self.global_step)
                        
                        try:
                            recon_img = recon_img.detach()
                            recon_img = (recon_img-real_img.min()) / (recon_img.max()-real_img.min())
                            
                            # self.writer.add_scalar("train_loss", los, self.global_step)
                            for k, v in record_losses.items():
                                self.writer.add_scalar(f'train_loss_{k}', v, self.global_step)
                            slice_index = self.args.roi_size[0]//2
                            self.writer.add_images('train_real_img', real_img[:,:,slice_index], self.global_step)
                            self.writer.add_images('train_masked_img', masked_img[:,:,slice_index], self.global_step)
                            self.writer.add_images('train_recon_img', recon_img[:,:,slice_index], self.global_step)
                            
                            if self.args.model_config.synthesize:
                                fake_img = fake_img.detach()
                                fake_img = (fake_img-real_img.min()) / (fake_img.max()-real_img.min())

                                self.writer.add_images('train_fake_img', fake_img[:,:,slice_index], self.global_step)

                            if self.args.lrdecay:
                                self.writer.add_scalar("lr", self.scheduler.get_last_lr(), self.global_step)


                        except Exception as e:
                            print('summary write error:', e)

                            
                
                self.last_eval += 1
                if self.global_step>self.args.num_steps:
                    print('break by global step over num_steps, step:', self.global_step)
                    break

            self.last_eval = 0
                
        # aggregate and print metrics
        metric_dict, eval_metric = self.cal_metric(None, None, 'train', task_metrics, self.args, aggregate=True)
        if self.args.rank == 0:
            print(time.strftime("%Y-%m-%d %H:%M:%S"), metric_dict)

        loss = loss_metrics['all'].aggregate().item()
        return loss, eval_metric
        
        
    @torch.no_grad()
    def val_epoch(self, loader, epoch, mode = 'val'):
        self.model.eval()
        loss_metrics = {'all': CumulativeAverage()}
        task_metrics = self.task_metrics()

        with tqdm(loader, desc=f'{mode} epoch {epoch} in gpu {self.args.gpu}', disable=self.args.gpu!=0) as pbar:
            for batch_data in pbar:
                data, target = self.prepare_data(batch_data, mode, self.args)
                _, meta_label, part_label = target
                real_img, masked_img = data

                recon_img, meta_part_pred = self.model('rm', real_img, masked_img)
                if self.args.model_config.synthesize:
                    fake_img, _, _, _, _ = self.model('sd', real_img)

                pred = (recon_img, None, meta_part_pred, None)

                # loss = self.loss_fn(pred, target, mode, self.args)
                losses = {'recon': Recon_Loss(recon_img, real_img)}
                meta_pred, part_pred = meta_part_pred
                if self.args.model_config.pred_meta:
                    losses['meta'] = Meta_Loss(meta_pred, meta_label)
                if self.args.model_config.pred_part:
                    losses['part'] = Part_Loss(part_pred, part_label)

                loss = sum([self.loss_weight[k] * v for k, v in losses.items()])
                loss_metrics['all'].append(loss.detach())
                for k, v in losses.items():
                    if not k in loss_metrics:
                        loss_metrics[k] = CumulativeAverage()
                    loss_metrics[k].append(v.detach())

                task_metrics, eval_metric = self.cal_metric(pred, target, mode, task_metrics, self.args)
            
                if self.args.debug:
                    break

        # aggregate and print metrics
        metric_dict, eval_metric = self.cal_metric(None, None, mode, task_metrics, self.args, aggregate=True)
        if self.args.rank == 0:
            try:
                recon_img = recon_img.detach()
                recon_img = (recon_img-real_img.min()) / (recon_img.max()-real_img.min())
                slice_index = self.args.roi_size[0]//2
                self.writer.add_images(f'{mode}_real_img', real_img[:,:,slice_index], self.global_step)
                self.writer.add_images(f'{mode}_masked_img', masked_img[:,:,slice_index], self.global_step)
                self.writer.add_images(f'{mode}_recon_img', recon_img[:,:,slice_index], self.global_step)

                if self.args.model_config.synthesize:
                    fake_img = fake_img.detach()
                    fake_img = (fake_img-real_img.min()) / (fake_img.max()-real_img.min())
                    self.writer.add_images(f'{mode}_fake_img', fake_img[:,:,slice_index], self.global_step)

            except Exception as e:
                print('log error:', e)

            print(time.strftime("%Y-%m-%d %H:%M:%S"), metric_dict)

        loss = loss_metrics['all'].aggregate().item()
        return loss, eval_metric, metric_dict

    