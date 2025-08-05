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
import logging
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.parallel
import torch.utils.data.distributed

from run.lr_scheduler import LinearWarmupCosineAnnealingLR, WarmupCosineSchedule
from util.basicutils import backup_scripts, save_ckpt
from util.datautils import get_loader_and_adapter
from util.load_pt import load_pt

from run.options import kargs as args

# resource limit
import resource

# torch.multiprocessing.set_sharing_strategy('file_system')
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))
# print('Setting resource limit:', str(resource.getrlimit(resource.RLIMIT_NOFILE)))
# os.environ["MASTER_ADDR"] = "localhost"
# os.environ["MASTER_PORT"] = "28890"

def main(args):
    # set up the experiment
    args.amp = not args.noamp
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    if args.distributed:
        dist.init_process_group(
            backend='nccl', init_method=args.dist_url )
        print(
            f"Training in distributed mode with multiple processes, 1 GPU per process. This is process {args.local_rank} in GPU {args.gpu}, total {args.world_size}."
        )
    else:
        print("Training with a single process on 1 GPU.")
    torch.cuda.set_device(args.gpu)

    if args.rank == 0:
        backup_scripts(sourcepath=args.CodePath, outpath=os.path.join(args.logdir, 'scripts/'))



    # get model
    loaders, adapters = get_loader_and_adapter(args)
    if args.rank == 0:
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "Batch size is:", args.batch_size, "max_epochs", args.max_epochs)

    model = adapters['model']()
    if args.pretrained_weight != "":
        pretrained_weight = args.pretrained_weight
        try:
            model = load_pt(model, pretrained_weight, ckpt_vit_key='swinViT', model_vit_key='swinViT')
            if args.rank == 0:
                print(time.strftime("%Y-%m-%d %H:%M:%S"), "Loaded pretrained weights from", pretrained_weight)
        except Exception as e:
            if args.rank == 0:
                print(time.strftime("%Y-%m-%d %H:%M:%S"), "Failed to load pretrained weights from", pretrained_weight, "Error:", e)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.rank ==0:
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "Total trainable parameters count", pytorch_total_params/1e6, "M")

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict, strict=False)
    model = model.to('cuda')

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                          device_ids=[args.local_rank], 
                                                          find_unused_parameters=True if (args.stage == "PT" and args.model_config.synthesize) else False,
                                                          )

    # optimizer and lr scheduler
    if args.opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.opt))

    if args.lr_schedule == "warmup_cosine":
        if args.stage == "PT":
                scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)
        else:
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=args.warmup_steps, max_epochs=args.max_epochs
            )
    elif args.lr_schedule == "poly":

        def lambdas(epoch):
            return (1 - float(epoch) / float(args.max_epochs)) ** 0.9

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)

    elif args.lr_schedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    else:
        scheduler = None


    # checkpoint
    start_epoch = 0
    global_step = 0

    if args.resume is not None:
        if "optimizer" in checkpoint:
            optimizer_s_d = checkpoint["optimizer"]
            optimizer.load_state_dict(optimizer_s_d)
        if "scheduler" in checkpoint:
            scheduler_s_d = checkpoint["scheduler"]
            scheduler.load_state_dict(scheduler_s_d)
        if "epoch" in checkpoint and not args.debug:
            start_epoch = checkpoint["epoch"]
        if 'global_step' in checkpoint and not args.debug:
            global_step = checkpoint['global_step']
        if args.rank == 0:
            print(time.strftime("%Y-%m-%d %H:%M:%S"), f"=> loaded checkpoint {args.resume} (epoch {start_epoch})")

    # training
    if args.rank == 0:
        print("Start training\n***********************************")
    
    if args.stage == "PT":
        from run.train_pt import Trainer
        
    elif args.stage == "FT":
        from run.train_ft import Trainer
    trainer = Trainer(loaders, adapters, optimizer, scheduler, model, args)
    trainer.run(start_epoch=start_epoch, global_step=global_step)

    if args.rank == 0:
        print(time.strftime("%Y-%m-%d %H:%M:%S"), 'end training')
        save_ckpt(model, start_epoch, args, filename="finish_train.pt", optimizer=optimizer, scheduler=scheduler)
    if args.distributed:    
        torch.distributed.barrier()
        dist.destroy_process_group()



if __name__ == "__main__":
    if args.distributed:
        args.gpu = int(os.environ["LOCAL_RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.rank = int(os.environ["RANK"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
    if args.rank == 0:
        print(args)
        print("Initializing training\n***********************************")
    main(args)
