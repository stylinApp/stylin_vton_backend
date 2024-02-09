from PIL import Image, ImageDraw, ImageFilter
import requests
import numpy as np
import glob, os
import torch
import argparse
from torchvision import transforms
from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

from segment_anything import SamPredictor, sam_model_registry
from groundingdino.util.inference import load_model, load_image, predict, annotate
from GroundingDINO.groundingdino.util import box_ops
from PIL import Image
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from face_detection import *
device = "cuda"
model_type = "vit_h"
predictor = SamPredictor(sam_model_registry[model_type](checkpoint="./weights/sam_vit_h_4b8939.pth").to(device=device))

groundingdino_model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
TEXT_PROMPT = "T-shirt"
BOX_TRESHOLD = 0.3
TEXT_TRESHOLD = 0.25
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



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of preprocessing daa.")
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to source directory.",
    )

    parser.add_argument(
        "--instance_prompt",
        type=str,
        required=True,
        help="Path to output directory.",
    )

    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Path to output directory.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to destinate directory.",
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "stabilityai/stable-diffusion-2-inpainting"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_name)
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.unet.load_attn_procs(args.model_path)
    pipe = pipe.to(device)

    os.makedirs(f'{args.out_path}', exist_ok=True)

    if os.path.isdir(args.image_path):
        img_paths = glob.glob(os.path.join(args.image_path, '*.jpg'))
        img_paths.extend(glob.glob(os.path.join(args.image_path, '*.png')))  # Add .png files
        img_paths.extend(glob.glob(os.path.join(args.image_path, '*.jpeg'))) 
        img_paths.sort()
    else:
        img_paths = [args.image_path]


    for img_path in img_paths:
        init_image = Image.open(img_path).convert("RGB")
        input_image = cv2.imread(img_path)
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
        kernel_size = 5  # Adjust the kernel size based on your needs
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_image = cv2.dilate(mask_image, kernel, iterations=1)
        # mask_image = cv2.erode(mask_image, kernel, iterations=1)
        mask_image = cv2.GaussianBlur(mask_image, (15, 15), 0)
        # mask_image = transforms.ToPILImage()(torch.sigmoid(preds)).convert("L").resize((512, 512))
        

        

        # inputs_clipseg = processor_clipseg(text=[args.instance_prompt], images=[init_image], padding="max_length", return_tensors="pt").to(device)
        # outputs = model_clipseg(**inputs_clipseg)
        # preds = outputs.logits.unsqueeze(0)[0].detach().cpu()
        # mask_image = transforms.ToPILImage()(torch.sigmoid(preds)).convert("L").resize((512, 512))
        # mask_image = mask_image.filter(ImageFilter.MaxFilter(21))
        #num_inference_steps = 20
        # prompt = f"a photo of {args.instance_prompt},high resolution "
        # prompt = "Generate a high-quality inpainted rendering of, White & Navy Color, Striped Cotton T-shirt using the trained model. Ensure precise reconstruction of facial features and strict containment within specified mask boundaries for optimal results."
        # prompt = '2'
        # prompt = "a photo of  black color  T-shirt" 
        # prompt = "a photo of White and green floral printed opaque Casual shirt ,has a spread collar, button placket, long regular sleeves, curved hem"
        # prompt = "a photo of Blue Solid Single-Breasted Blazer"
        # prompt = "a photo of blue solid regular-fit blazer, has a notched lapel, single-breasted, button closures on the front, three pockets, long sleeves, and a double-vented back hem"
        # prompt = "a photo of Colourblocked black and white Round Neck T-shirt"
        # prompt = "a photo of White & Navy Color Striped Cotton T-Shirt."
        # prompt= "a photo of vivid blue T-Shirt"
        # prompt = "a photo of Black T-shirt for men , Graphic printed ,Regular length,Round neck, Short, regular sleeves,Knitted cotton fabric"
        prompt = "a photo of White and navy blue striped T-shirt, has a round neck, short sleeves"
        # negative_prompt = "out of frame, edges on borders, edges on sholders,increased sholders, design on skin,mutated hands, poorly drawn hands,deformed hands,unnatural hand proportions,hard lines,contour, lowres, distortion while inpainting,text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate"
        negative_prompt = "(((out of frame))), (((bad inpainting))),(((pattern mismatch))),(((color difference))),(((noise at edges))),(((line at lower part))),(((lines on borders,(((noise near edges))),(((wider shoulder))),(((unrealistic shoulder))),(((sharp edges on shoulders))),edges on borders,(((sharp edges))), edges on shoulders, design on skin,mutated hands, poorly drawn hands,(((deformed hands))),unnatural hand proportions,lowres, (((distortion while inpainting))),worst quality, low quality, jpeg artifacts,ugly, duplicate,extra fingers, (((mutated hands))), (((poorly drawn hands))), poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, (((long neck))), username, watermark, signature"
        image = pipe(prompt=prompt, negative_prompt=negative_prompt,image=init_image, num_inference_steps = 50,
                    mask_image=mask_image
                    ).images[0]
        # image.save("output_image.jpg")

        # cat_image = Image.new('RGB', (512 * 2, 512))

        # masked_image = Image.composite(mask_image, init_image, mask_image)
        # cat_image.paste(init_image, (512*0, 0))
        # cat_image.paste(image, (512*1, 0))

        # cat_image.save(f"{args.out_path}/{os.path.basename(img_path)}")


        mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8))

        # Now use the converted mask_image_pil in the composite method
        masked_image = Image.composite(init_image, mask_image_pil, mask_image_pil)
        masked_image.save('out_try_1.jpg')
        # masked_image = Image.composite(init_image, mask_image_pil, mask_image_pil)

        # Rest of your code remains unchanged
        # cat_image = Image.new('RGB', (512 * 2, 512))
        # cat_image.paste(init_image, (512 * 0, 0))
        # cat_image.paste(image, (512 * 1, 0))
        # target_img_path = f"{args.out_path}/{os.path.basename(img_path)}"

        # numpy_array = np.array(pillow_image)

        # # Convert NumPy array to OpenCV image
        # opencv_image = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)

        # print("input_image------------------",type(input_image))
        # print("result from diffusion -------------------",type(image))
        # print("output after face detection ---------------",type(img_path))
        target_img = detect_and_crop_face(img_path,image,img_path)
        # print("++++++++++++++++++++++++++++++++++++++++++++",target_img)
        target_img.save(f"{args.out_path}/{os.path.basename(img_path)}")

        # print(f"T-shirt_mask/{os.path.basename(img_path)}")

        # print(mask_image)
        cv2.imwrite(f"T-shirt_mask/{os.path.basename(img_path)}",mask_image)