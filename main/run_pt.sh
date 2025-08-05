now=$(date +"%m%d-%H%M%S")
ds=all
logdir=Result/Exp/PT/$now-$ds
mkdir -p $logdir
export OMP_NUM_THREADS=2
export TORCH_DISTRIBUTED_DEBUG=DETAIL

torchrun --standalone --nproc_per_node=gpu train.py \
    --stage PT \
    --ds $ds \
    --lr 0.0005 \
    --num_steps 2000000 \
    --warmup_steps 1000 \
    --evaluate_every 1 \
    --batch_size  18 \
    --num_workers 1 \
    --weight_decay 0.1 \
    --distributed \
    --logdir $logdir | tee $logdir/$now.log

    