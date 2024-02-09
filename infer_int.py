from PIL import Image, ImageDraw, ImageFilter
import requests
import numpy as np
import glob, os
import torch
from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler
from segment_anything import SamPredictor, sam_model_registry
from groundingdino.util.inference import load_model, load_image, predict, annotate
from GroundingDINO.groundingdino.util import box_ops
from PIL import Image
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from face_detection import *
from read_img_s3 import *
device = "cuda"
model_type = "vit_h"
predictor = SamPredictor(sam_model_registry[model_type](checkpoint="./weights/sam_vit_h_4b8939.pth").to(device=device))

groundingdino_model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")

def show_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))


#user_img = Image uploaded by user
#prompt = "a photo of" + Product Details
# model_path = Product ID
def tryon(UserImgUrl,prompt,ProductID,UserID,GarmentType):
    TEXT_PROMPT = GarmentType
    BOX_TRESHOLD = 0.3
    TEXT_TRESHOLD = 0.25
    model_path = 'models'+'/'+ProductID
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "stabilityai/stable-diffusion-2-inpainting"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_name)
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.unet.load_attn_procs(model_path)
    pipe = pipe.to(device)


    os.makedirs(f'UserImg/{UserID}', exist_ok=True)
    image_path = f'UserImg/{UserID}'

    image_path = save_image_locally(UserImgUrl, image_path)

    if os.path.isdir(image_path):
        img_paths = glob.glob(os.path.join(image_path, '*.jpg'))
        img_paths.extend(glob.glob(os.path.join(image_path, '*.png')))  # Add .png files
        img_paths.extend(glob.glob(os.path.join(image_path, '*.jpeg'))) 
        img_paths.sort()
    else:
        img_paths = [image_path]
        

    for img_path in img_paths:
        init_image = Image.open(img_path).convert("RGB")
        init_size = init_image.size
        init_image = init_image.resize((512, 512))
        src, img = load_image(img_path)

        boxes, logits, phrases = predict(
        model=groundingdino_model,
        image=img,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD)
        img_annnotated = annotate(image_source=src, boxes=boxes, logits=logits, phrases=phrases)[...,::-1]
        predictor.set_image(src)
        H, W, _ = src.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        new_boxes = predictor.transform.apply_boxes_torch(boxes_xyxy, src.shape[:2]).to(device)
        masks, _, _ = predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = new_boxes,
            multimask_output = False,
        )

        img_annotated_mask = Image.fromarray(show_mask(masks[0][0].cpu(), img_annnotated))
        original_img = Image.fromarray(src).resize((512, 512))
        img_annotated = Image.fromarray(img_annnotated)
        only_mask = Image.fromarray(masks[0][0].cpu().numpy()).resize((512, 512))

        mask = Image.fromarray(masks[0][0].cpu().numpy())  # Extract the mask

        # Convert the mask to 8-bit unsigned integer format and scale values
        mask_image = (np.array(mask) * 255).astype(np.uint8)
        mask_image = cv2.resize(mask_image, (512, 512))
        kernel_size = 10  # Adjust the kernel size based on your needs
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_image = cv2.dilate(mask_image, kernel, iterations=1)
        mask_image = cv2.GaussianBlur(mask_image, (25, 25), 0)
    
        # prompt = "a photo of Colourblocked black and white Round Neck T-shirt"
        prompt = "a photo of "+ prompt
        image = pipe(prompt=prompt, image=init_image, num_inference_steps = 50,
                    mask_image=mask_image
                    ).images[0]
    
        mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8))

        masked_image = Image.composite(init_image, mask_image_pil, mask_image_pil)
        # masked_image.save('out_try_1.jpg')
    
        target_img = detect_and_crop_face(img_path,image,img_path)
        return target_img,ProductID
        