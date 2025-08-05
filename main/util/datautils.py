
from run.options import kargs as args
import importlib

if args.stage == 'FT' and args.monai_loader:
    from monai.data import DataLoader, DistributedSampler
else:
    from torch.utils.data import DistributedSampler, DataLoader


def get_downstream_socks(downstream_name:str):
    if args.stage == 'PT':
        ds_module = importlib.import_module(f"dataset.pretrain.ds")
        socks_module = importlib.import_module(f"dataset.pretrain.socks")
    else:
        ds_module = importlib.import_module(f"dataset.{downstream_name}.ds")
        socks_module = importlib.import_module(f"dataset.{downstream_name}.socks")

    train_ds, val_ds, test_ds = ds_module.get_ft_ds()

    return {'train_ds'  :   train_ds,
            "val_ds"    :   val_ds, 
            "test_ds"   :   test_ds},\
            {'prepare_data': socks_module.prepare_data,
            'infer_model':  socks_module.infer_model,
            'loss_fn'   :   socks_module.loss_fn,
            'cal_metric':   socks_module.cal_metric,
            'model'    :   socks_module.FT_MRI_FM,
            'task_metrics':socks_module.get_task_metrics
            }

    




def get_loader_and_adapter(args):
    if args.stage == 'PT':
        from dataset.pretrain.ds import get_datasets
        import dataset.pretrain.socks as socks_module
        train_ds, val_ds, test_ds = get_datasets(args.ds)
        adapters = {
            'prepare_data': socks_module.prepare_data,
            'infer_model':  socks_module.infer_model,
            'loss_fn'   :   socks_module.loss_fn,
            'cal_metric':   socks_module.cal_metric,
            'model'    :   socks_module.PT_MRI_FM,
            'task_metrics':socks_module.get_task_metrics
        }

        
    elif args.stage == 'FT':
        ds, adapters = get_downstream_socks(args.ds)
        train_ds = ds['train_ds']
        val_ds = ds['val_ds']
        test_ds = ds['test_ds']

    else:
        raise NotImplementedError(f"Stage {args.stage} not implemented yet")
    
    train_loader = get_train_loader(train_ds, args)
    val_loader = get_val_loader(val_ds, args)
    test_loader = get_val_loader(test_ds, args)

    return (train_loader, val_loader, test_loader), adapters



def get_train_loader(train_ds, args):

    if args.distributed:
        train_sampler = DistributedSampler(dataset=train_ds, shuffle=True)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers, sampler=train_sampler, drop_last=True, shuffle=(train_sampler is None)
    )

    # val_ds = Dataset(data=val_files, transform=val_transforms)
    # val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=True)

    return train_loader

def get_val_loader(val_ds, args):

    if args.distributed:
        val_sampler = DistributedSampler(dataset=val_ds, shuffle=False)
    else:
        val_sampler = None

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size if args.stage == 'PT' else args.val_batch_size, num_workers=args.num_workers, sampler=val_sampler, drop_last=True, shuffle=False
    )

    return val_loader