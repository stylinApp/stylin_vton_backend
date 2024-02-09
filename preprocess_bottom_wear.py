from segment_anything import SamPredictor, sam_model_registry
from groundingdino.util.inference import load_model, load_image, predict, annotate
from GroundingDINO.groundingdino.util import box_ops
from PIL import Image
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse

import os,glob,tqdm

# -----Set Image and CUDA

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of preprocessing daa.")
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        required=True,
        help="Path to source directory.",
    )

    parser.add_argument(
        "--instance_prompt",
        type=str,
        required=True,
        help="target object to be composed.",
    )

    args = parser.parse_args()

    return args
if __name__ == '__main__':
    args = parse_args()
    device = "cuda"

    # ------SAM Parameters
    model_type = "vit_h"
    predictor = SamPredictor(sam_model_registry[model_type](checkpoint="./weights/sam_vit_h_4b8939.pth").to(device=device))

    groundingdino_model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
    BOX_TRESHOLD = 0.3
    TEXT_TRESHOLD = 0.25


    img_files = glob.glob(os.path.join(args.instance_data_dir, "*.jpg"))

    
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
    for img_file in tqdm.tqdm(img_files):
        src, img = load_image(img_file)
        boxes, logits, phrases = predict(
            model=groundingdino_model,
            image=img,
            caption=args.instance_prompt,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
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


        # img_annotated_mask = Image.fromarray(show_mask(masks[0][0].cpu(), img_annnotated))
        # original_img = Image.fromarray(src).resize((512, 512))
        # img_annotated = Image.fromarray(img_annnotated)
        # only_mask = Image.fromarray(masks[0][0].cpu().numpy()).resize((512, 512))


        # mask = Image.fromarray(masks[0][0].cpu().numpy())  # Extract the mask

        # # Create a single figure and axis for the mask
        # fig, ax = plt.subplots(figsize=(15, 10))  # Adjust figsize as needed
        # # plt.title("Only Mask", fontsize=30)
        # ax.imshow(mask)
        # ax.axis('off')

        # # Save the mask directly
        # plt.savefig('mask.jpg')
    
        img_annotated_mask = Image.fromarray(show_mask(masks[0][0].cpu(), img_annnotated))
        original_img = Image.fromarray(src).resize((728, 728))
        img_annotated = Image.fromarray(img_annnotated)
        only_mask = Image.fromarray(masks[0][0].cpu().numpy()).resize((728, 728))

        mask = Image.fromarray(masks[0][0].cpu().numpy())  # Extract the mask

        # Convert the mask to 8-bit unsigned integer format and scale values
        mask_np = (np.array(mask) * 255).astype(np.uint8)
        # print(mask_np)
        # Save the mask using cv2
    
        cv2.imwrite(img_file[:-4]+ '.png', mask_np)