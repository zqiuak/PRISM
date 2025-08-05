#!/bin/bash

# Set your task and weight name here:
task='acdc'
weight_name='nopt'

# addition information to write to directory name
a='separate'

CONFIG_FILE=dataset/$task/config.ini
# check if the config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR]: Config $CONFIG_FILE not exist" >&2
    exit 1
fi


# read the configuration file and export variables
while IFS='=' read -r key value; do
    # skip empty lines and comments
    [[ -z "$key" ]] && continue
    [[ "$key" == \#* ]] && continue
    # import the variable into the current shell
    declare -g "CFG_$key"="$value"
done < "$CONFIG_FILE"


echo "=== Loading From Config  ==="
for var in $(compgen -v CFG_); do
    echo "$var=${!var}"
done
echo "============================"


# check ds
if [ -z "${CFG_ds}" ]; then
    echo "[ERROR]: parameter 'ds' does not exist in $CONFIG_FILE" >&2
    exit 1
fi


now=$(date +"%m%d-%H%M%S")
logdir="Result/Exp/FT/$CFG_ds/$now-$weight_name"

if [ -n "$a" ]; then
    logdir="$logdir-$a"
fi

if [ "$1" = "-d" ]; then
    debug="true"
    logdir="$logdir-debug"
else
    debug="false"
fi


mkdir -p $logdir
cp $CONFIG_FILE $logdir/config.ini

CMD="CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=gpu train.py"
CMD+=" --stage FT"
CMD+=" --distributed"
CMD+=" --ds ${CFG_ds}"
CMD+=" --logdir $logdir"


CFG_batch_size=${CFG_batch_size:-1}
CFG_num_workers=${CFG_num_workers:-0}
CFG_lr=${CFG_lr:-0.0001}
CFG_max_epochs=${CFG_max_epochs:-500}
CFG_warmup_steps=${CFG_warmup_steps:-10}
CFG_evaluate_every=${CFG_evaluate_every:-10}
CFG_sample_size=${CFG_sample_size:-2}
CFG_pretrained_weight=${CFG_pretrained_weight:-"weights/${weight_name}.pt"}



for var in $(compgen -v CFG_); do
    param_name="${var#CFG_}"  # remove the prefix 'CFG_'
    param_name=$(echo "$param_name")  
    
    # skip ds parameter
    if [ "$param_name" = "ds" ]; then
        continue
    fi
    
    param_value="${!var}"
    CMD+=" --$param_name $param_value"
done


if [[ "$weight_name" == "unetr" || "$weight_name" == "brainseg" ]]; then
    CMD+=" --no_v2"
fi

if [ "$debug" = "true" ]; then
    CMD+=" --debug"  # add debug flag
    echo "Debug mode is ON"
fi


echo "Running command: $CMD"
eval "$CMD" 2>&1 | tee "$logdir/$now.log"
