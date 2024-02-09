

### Installation
* Requirements
```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch 
pip install -r requirements.txt
```

* Initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with
  ```bash
  accelerate config default
  ```

* Run the following command to authenticate your token

  ```bash
  huggingface-cli login
  ```




### Preprocess Images
Please provide at least one images in .jpg format and instance prompt. The preprocess.py script will generate captions and instance masks.

```bash
python preprocess.py --instance_data_dir $INSTANCE_DIR \
                     --instance_prompt $INSTANCE_PROMPT
```

### Finetune
We then embed the instance images and prompt into stable diffusion model.

```bash
accelerate launch --num_processes 1 train.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$MODEL_DIR \
  --instance_prompt=$INSTANCE_PROMPT \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000
```

### Inference
Finally, you can provide new images to achieve image composition.

```bash
python inference.py --image_path $Test_DIR \
                    --model_path $MODEL_DIR \
                    --out_path $OUT_DIR \
                    --instance_prompt $INSTANCE_PROMPT
```


