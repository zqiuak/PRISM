#!/bin/bash
# set here:

exp_path="./YourExpDir"



weight_name=${exp_path##*-}
a=''

ckpt_path="$exp_path/best_eval.pt"
CONFIG_FILE="$exp_path/config.ini"

# check if the config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR]: Config $CONFIG_FILE not exist" >&2
    exit 1
fi

# read the configuration file and export variables
while IFS='=' read -r key value; do
    # skip empty lines and comments
    [[ -z "$key" ]] && continue
    [[ -z "$key" || "$key" == \#* ]] && continue
    # import the variable into the current shell
    declare -g "CFG_$key"="$value"
done < "$CONFIG_FILE"

# check ds
if [ -z "${CFG_ds}" ]; then
    echo "[ERROR]: parameter 'ds' does not exist in $CONFIG_FILE" >&2
    exit 1
fi


now=$(date +"%m%d-%H%M%S")
logdir="Result/Exp/INFER/$CFG_ds/$now-$weight_name"

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

CMD="python infer.py"
CMD+=" --stage FT"
CMD+=" --infer"
CMD+=" --ds ${CFG_ds}"
# CMD+=" --no_v2"
CMD+=" --logdir $logdir"


CFG_batch_size=1
CFG_num_workers=0
CFG_lr=${CFG_lr:-0.0001}
CFG_max_epochs=1
CFG_evaluate_every=1
CFG_sample_size=1
CFG_resume=$ckpt_path



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

CMD+="| tee $logdir/$now.log"

echo "Running command: $CMD"
eval "$CMD"


