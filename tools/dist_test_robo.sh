#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --use_env \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test_corruption.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    ${@:4}

    # tools/dist_test_robo.sh projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_robo.py work_dirs/repdetr3d_eva02_800_bs2_seq_24e/iter_42192.pth 8 --eval bbox