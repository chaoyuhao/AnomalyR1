# cd src/open-r1-multimodal
# --lr_scheduler_type constant \
# --learning_rate 5e-7 \
# 

export DEBUG_MODE="true"
export CUDA_VISIBLE_DEVICES=0,1,2,3

IMAGE_ROOT="/home/chaoyuhao/workbench/MMAD/few-shot-dataset/"
RUN_NAME="GRPO-FSHOT-1"
export LOG_PATH="./debug_log_$RUN_NAME.txt"
export HF_ENDPOINT=https://hf-mirror.com

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo_rec.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name data_config/ad.yaml \
    --image_root $IMAGE_ROOT \
    --max_prompt_length 1024 \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 10 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true
