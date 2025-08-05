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

from run.options import kargs as args

# resource limit
import resource

def main(args):
    # set up the experiment
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
    assert args.resume is not None, "Please provide a checkpoint to resume from"
    checkpoint = torch.load(args.resume, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model = model.to('cuda')

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                          device_ids=[args.local_rank], 
                                                          find_unused_parameters=True
                                                          )


    if args.rank == 0:
        print("Start infering\n***********************************")
    
    from run.infer import Inferer
    
    inferer = Inferer(loaders, adapters,  model, args)
    inferer.run()

    if args.rank == 0:
        print(time.strftime("%Y-%m-%d %H:%M:%S"), 'end infer')
    if args.distributed:    
        torch.distributed.barrier()
        dist.destroy_process_group()



if __name__ == "__main__":
    if args.distributed:
        args.gpu = int(os.environ["LOCAL_RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        assert args.world_size  == 1, 'Onlye one GPU is supported for inference'
        args.rank = int(os.environ["RANK"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
    if args.rank == 0:
        print(args)
        print("Initializing infer\n***********************************")
    main(args)
