export CUDA_VISIBLE_DEVICES=0 
# export MODEL_NAME="models/blazer"
export MODEL_NAME="stabilityai/stable-diffusion-2-inpainting"
# export MODEL_NAME="diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
# export MODEL_NAME="runwayml/stable-diffusion-inpainting"

export INSTANCE_DIR="data/upper_wear_nautica"
export Test_DIR="data/nt"
export OUT_DIR="out/blazer"
export INSTANCE_PROMPT="T-Shirt"
export MODEL_DIR="models/blazer"

# preprocess data
python preprocess_sam.py --instance_data_dir $INSTANCE_DIR \
                     --instance_prompt $INSTANCE_PROMPT

accelerate launch --num_processes 1 train.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --use_8bit_adam \
  --output_dir=$MODEL_DIR \
  --instance_prompt=$INSTANCE_PROMPT \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6    \
  --lr_scheduler="cosine_with_restarts" \
  --lr_warmup_steps=10 \
  --max_train_steps=500 \
  
  # --mixed_precision=bf16 \
  

python infer_sam.py --image_path $Test_DIR \
                    --model_path $MODEL_DIR \
                    --out_path $OUT_DIR \
                    --instance_prompt $INSTANCE_PROMPT \


###################################################################try#####################################################
# # preprocess data
# python preprocess_sam.py --instance_data_dir $INSTANCE_DIR \
#                      --instance_prompt $INSTANCE_PROMPT

# CUDA_VISIBLE_DEVICES=0
# accelerate launch --num_processes 1 train.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$MODEL_DIR \
#   --instance_prompt=$INSTANCE_PROMPT \
#   --resolution=516 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=1 \
#   --learning_rate=5e-6    \
#   --lr_scheduler="cosine_with_restarts" \
#   --lr_warmup_steps=10 \
#   --max_train_steps=500 \
#   # --mixed_precision=bf16 \
  

# python infer_sam.py --image_path $Test_DIR \
#                     --model_path $MODEL_DIR \
#                     --out_path $OUT_DIR \
#                     --instance_prompt $INSTANCE_PROMPT \



#5e-6 
# 0.000005
# 0.000005
# 0.000001 