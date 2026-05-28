#!/bin/bash
set -x
# source /opt/rh/devtoolset-8/enable
nproc_per_node=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TENSORBOARD_DIR=model/training_logs/qwen25_7b_sft_data_1200/
DATA_DIR=data/training_data/sft_data_1200
SAVE_DIR=model/qwen25-7b-sft-1200-data
MODEL_PATH=<cluster_path>/banma_aigc/model/huggingface.co/Qwen/Qwen2.5-7B-Instruct
# export LD_LIBRARY_PATH=<project_root>/envs/py310_verl_megatron/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=<project_root>/envs/py310_verl_niuwenzhe/lib:$LD_LIBRARY_PATH

export RAY_DEDUP_LOGS=1 
cd <project_root>/verl
export PYTHONPATH=$PYTHONPATH:$(pwd)
torchrun --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=2 \
    data.max_length=6000 \
    data.multiturn.tools_key=None \
    model.partial_pretrain=$MODEL_PATH \
    model.fsdp_config.model_dtype=bf16 \
    trainer.default_local_dir=$SAVE_DIR \
    trainer.project_name=qwen25-32b-sft \
    trainer.experiment_name=qwen25-32b-1200-data \
    trainer.logger=['console','tensorboard'] \
    trainer.total_epochs=1 \
    ulysses_sequence_parallel_size=2 \
    trainer.test_freq=-1 \
    use_remove_padding=true