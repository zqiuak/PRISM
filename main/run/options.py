import torch
import os, argparse, sys
import time

from run.pathdict import CodePath, RootPath, CachePath, ExpFolder
from model.model_var import model_config



class HyperParams():
    def __init__(self):

        # exp args
        self.gpu = "0"
        self.stage = 'FT'
        self.infer = False
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        self.gpu = 0
        self.distributed = False
        self.pretrained_weight = ""
        self.dist_url = "env://"
        self.device = torch.device(f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu")

        #path args
        self.CodePath = CodePath
        self.RootPath = RootPath
        self.CachePath = CachePath
        self.ExpFolder = ExpFolder

        # task args
        # self.pretrain_ds = ["PrivateKnee", "OAI_ZIB", "OAI", "PrivateCancer"]
        # self.pretrain_ds = ["PrivateCancer"]
        # self.finetune_ds = ["MRNet"]

        # read data args 
        self.ds = "10k" # dataset name, '10l, 300k' if pretrain
        self.Spacing = (1.0, 1.0, 1.0)

        # model args
        # backbone: SwinUNETR, ResNet
        self.modelname = model_config.name
        self.model_config = model_config
        self.emb_dim = model_config.emb_dim
        self.latent_dim = model_config.latent_dim
        self.encoder_only = model_config.encoder_only
        self.sw_layer = model_config.sw_layer
        self.num_heads = model_config.num_heads

        self.spatial_dims = 3
        # input channels
        self.in_channels = 1
        # swin unetr output (mask) channels
        self.out_channels = 1
        # use gradient checkpointing to save memory
        self.use_checkpoint = True
        self.use_v2 = True
        self.alpha = 0.1
        self.dropout_path_rate = 0.0
        
        # train data args
        self.augment_data = True
        self.use_cache = True
        self.data_balance = False
        self.roi_x = 96
        self.roi_y = 96
        self.roi_z = 96
        # self.roi_size = (128, 128, 128)
        # self.roi_size = (192, 192, 192)
        self.batch_size = 2
        self.val_batch_size = 1 # batch size for validation and testing
        self.sample_size = 2 # how many samples per case during training
        self.sw_batch_size = 2
        self.sw_overlap = 0.75 # sliding window overlap
        self.monai_loader = False

        # training process args
        self.max_epochs = 200
        self.num_steps = 250000
        self.num_workers = 1
        self.grad_accumulate = 1 # accumulate gradient for how many batches
        self.log_every = 100
        self.evaluate_every = 1
        self.patience = 100
        self.resume = None
        self.grad_clip = False
        self.noamp = False
        
        # optimizer args
        self.opt = "adamw"
        self.lr = 4e-4
        self.gamma = 0.1
        self.momentum = 0.9
        self.warmup_steps = 1000
        self.weight_decay = 1e-3
        self.lrdecay = True
        self.max_grad_norm = 1.0
        self.lr_schedule = "warmup_cosine"
        

        self.debug = False

        self.parse_arguments()

        if self.logdir == "":
            self.logdir = self.ExpFolder+"/{}-{}-{}/".format(self.stage,self.modelname,time.strftime("%m%d-%H%M%S"))
        

        if self.debug:
            self.set_debug()
        
        
        # else
        self.attlist = [attr for attr in dir(self) if not callable(getattr(self,attr)) and not attr.startswith("__")]

        print("{}-{}-{}".format(self.stage,self.modelname,time.strftime("%m%d-%H%M%S")))
        # print('options initiated')

    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        
        parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")

        # Exp args
        parser.add_argument('--stage', type=str, default=self.stage, help='stage of the experiment')
        parser.add_argument('--infer', action='store_true', default=self.infer, help='inference mode')
        parser.add_argument('--logdir', type=str, default='', help='log directory')
        parser.add_argument('--log_every', type=int, default=self.log_every, help='log every n iterations')
        parser.add_argument('--ds', type=str, default=self.ds, help='dataset name')
        parser.add_argument('--debug', action='store_true', default=self.debug, help='debug mode')
        parser.add_argument('--num_workers', type=int, default=self.num_workers, help='number of workers')

        # Training args
        parser.add_argument("--evaluate_every", default=self.evaluate_every, type=int, help="evaluation frequency")
        parser.add_argument('--augment', default=self.augment_data, type=bool, help='augment the data')
        parser.add_argument('--gamma', type=float, default=self.gamma)
        parser.add_argument('--max_epochs', type=int, default=self.max_epochs)
        parser.add_argument("--num_steps", default=self.num_steps, type=int, help="number of training iterations")
        parser.add_argument('--batch_size', type=int, default=self.batch_size)
        parser.add_argument('--val_batch_size', type=int, default=self.val_batch_size, help='batch size for validation and testing')
        parser.add_argument('--sample_size', type=int, default=self.sample_size, help='how many samples per case during training')
        parser.add_argument("--sw_batch_size", default=self.sw_batch_size, type=int, help="number of sliding window batch size")
        parser.add_argument("--sw_overlap", default=self.sw_overlap, type=float, help="sliding window overlap")
        parser.add_argument('--monai_loader', choices=['true', 'false'], default=str(self.monai_loader), help='use monai loader')

        parser.add_argument('--lr', type=float, default=self.lr)
        parser.add_argument('--flush_history', type=int, choices=[0, 1], default=0)
        parser.add_argument('--save_model', type=int, choices=[0, 1], default=1)
        parser.add_argument('--patience', type=int, default=self.patience)
        parser.add_argument("--warmup_steps", default=self.warmup_steps, type=int, help="warmup steps")
        parser.add_argument('--gpu',type=str, default=self.gpu, help='gpu id')
        parser.add_argument('--grad_accumulate', type=int, default=self.grad_accumulate, help='gradient accumulation steps')

        # Model args
        parser.add_argument('--backbone', type=str, default=self.modelname, help='backbone model name')
        parser.add_argument('--roi_x', type=int, default=self.roi_x, help='roi x size')
        parser.add_argument('--roi_y', type=int, default=self.roi_y, help='roi y size')
        parser.add_argument('--roi_z', type=int, default=self.roi_z, help='roi z size')

        parser.add_argument("--in_channels", default=self.in_channels, type=int, help="input channels")
        parser.add_argument("--out_channels", default=self.out_channels, type=int, help="output channels")
        parser.add_argument("--dropout_path_rate", default=self.dropout_path_rate, type=float, help="drop path rate")
        parser.add_argument("--use_checkpoint", default=self.use_checkpoint, help="use gradient checkpointing to save memory")
        parser.add_argument("--no_v2", action='store_true', help="dont use v2 version of swinvit")
        

        parser.add_argument("--weight_decay", default=self.weight_decay, type=float, help="decay rate")
        parser.add_argument("--momentum", default=self.momentum, type=float, help="momentum")
        parser.add_argument("--lrdecay", default=self.lrdecay, help="enable learning rate decay (wormup + cosine annealing)")
        parser.add_argument("--max_grad_norm", default=self.max_grad_norm, type=float, help="maximum gradient norm")
        parser.add_argument("--opt", default=self.opt, type=str, help="optimization algorithm")
        parser.add_argument("--lr_schedule", default=self.lr_schedule, type=str)
        parser.add_argument("--resume", default=self.resume, type=str,
                            help="resume training")
        parser.add_argument("--grad_clip", action="store_true", 
        default=self.grad_clip, help="gradient clip")
        parser.add_argument("--distributed", action="store_true", default=self.distributed, help="distributed training")
        parser.add_argument("--noamp", action="store_true", default=self.noamp, help="do NOT use amp for training")
        parser.add_argument("--dist_url", default=self.dist_url, help="url used to set up distributed training")
        parser.add_argument("--smartcache_dataset", default=False, help="use monai smartcache Dataset")
        parser.add_argument("--cache_dataset", action="store_true", help="use monai cache Dataset")
        parser.add_argument("--pretrained_weight", default=self.pretrained_weight, type=str, help="pretrained weight path")


        def is_notebook() -> bool:
            try:
                shell = get_ipython().__class__.__name__
                if shell == 'ZMQInteractiveShell':
                    return True   # Jupyter notebook or qtconsole
                elif shell == 'TerminalInteractiveShell':
                    return False  # Terminal running IPython
                else:
                    return False  # Other type (?)
            except NameError:
                return False


        if is_notebook():
            args = parser.parse_args(args=[]) # jupyter
        else:
            args = parser.parse_args() # python


        args.monai_loader = True if args.monai_loader == 'true' else False
        
        self.input_args = args
        argslist = [attr for attr in dir(args) if not callable(getattr(args,attr)) and not attr.startswith("__")]

        for each in argslist:
            setattr(self, each, getattr(args, each))

        self.amp = not self.noamp
        self.use_v2 = not self.no_v2
        self.roi_size = (self.roi_x, self.roi_y, self.roi_z)

    def set_debug(self):
            self.max_epochs = 2
            self.evaluate_every = 1
            self.grad_accumulate = 2
            # self.num_workers = 2

    def __str__(self) -> str:
        try:
            # output all the attributes
            
            return '********args********\n'+str({k: getattr(self, k) for k in self.attlist})+"\n********args********\n"
        except:
            return ""

kargs = HyperParams()
