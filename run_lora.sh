export MODEL_NAME="stabilityai/stable-diffusion-2-inpainting"
export INSTANCE_DIR="data/upper_wear_M&B"
export Test_DIR="data/nt"
export OUT_DIR="out/T-shirt_5_03"
export INSTANCE_PROMPT="T-shirt"
export MODEL_DIR="models/T-shirt_11"
# # preprocess data
python preprocess_sam.py --instance_data_dir $INSTANCE_DIR \
                     --instance_prompt $INSTANCE_PROMPT

# # CUDA_VISIBLE_DEVICES=0
# accelerate launch --num_processes 1 train_lora.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$MODEL_DIR \
#   --instance_prompt=$INSTANCE_PROMPT \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=1 \
#   --learning_rate=5e-6    \
#   --lr_scheduler="cosine_with_restarts" \
#   --lr_warmup_steps=10 \
#   --max_train_steps=2000 \
#   --checkpointing_steps 1000
  

# From Source 
# CUDA_VISIBLE_DEVICES=0
# accelerate launch --num_processes 1 train_lora_org.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$MODEL_DIR \
#   --instance_prompt=$INSTANCE_PROMPT \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --learning_rate=1e-4 \
#   --max_train_steps=2000 \
#   --checkpointing_steps 1000

# python inference_lora.py --image_path $Test_DIR \
#                     --model_path $MODEL_DIR \
#                     --out_path $OUT_DIR \
#                     --instance_prompt $INSTANCE_PROMPT

# export INSTANCE_DIR="data/upper_wear_nautica"
# export OUT_DIR="out/T-shirt"
# export INSTANCE_PROMPT="T-shirt"
# export MODEL_DIR="models/T-shirt"

# # CUDA_VISIBLE_DEVICES=0
# accelerate launch --num_processes 1 train_lora.py \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$MODEL_DIR \
#   --instance_prompt=$INSTANCE_PROMPT \
  


