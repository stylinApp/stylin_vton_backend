export MODEL_NAME="stabilityai/stable-diffusion-2-inpainting"

export INSTANCE_DIR="data/trouser"
export Test_DIR="data/test_bottom"
export OUT_DIR="out/trouser"
export INSTANCE_PROMPT="trouser"
export MODEL_DIR="models/trouser"

# # preprocess data
# python preprocess_bottom_wear.py --instance_data_dir $INSTANCE_DIR \
#                      --instance_prompt $INSTANCE_PROMPT

# # CUDA_VISIBLE_DEVICES=0
# accelerate launch --num_processes 1 train_bottom_wear.py \
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
#   --max_train_steps=500 \

python infer_bottom_wear.py --image_path $Test_DIR \
                    --model_path $MODEL_DIR \
                    --out_path $OUT_DIR \
                    --instance_prompt $INSTANCE_PROMPT


#5e-6 