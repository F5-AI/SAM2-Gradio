import os
import cv2
import torch
import numpy as np
import gradio as gr
from tqdm import tqdm
from PIL import Image, ImageDraw

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import gc


if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


image_predictor = None
sam2_model = None


def get_sam2_config():
    return "sam2_hiera_l.yaml", "checkpoints/sam2_hiera_large.pt"


def get_sam2_model(device):
    global sam2_model
    if sam2_model:
        return sam2_model
    model_cfg, sam2_checkpoint = get_sam2_config()
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    return sam2_model


def get_image_predicator(device):
    global image_predictor
    if image_predictor:
        return image_predictor
    image_predictor = SAM2ImagePredictor(get_sam2_model(device))
    return image_predictor


def segment_one(img, mask_generator, seed=None):
    if seed is not None:
        np.random.seed(seed)
    masks = mask_generator.generate(img)
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    mask_all = np.ones((img.shape[0], img.shape[1], 3))
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            mask_all[m == True, i] = color_mask[i]
    result = img / 255 * 0.3 + mask_all * 0.7
    return result, mask_all


def generator_inference(device, input_image):
    mask_generator = SAM2AutomaticMaskGenerator(get_sam2_model(device))
    result, mask_all = segment_one(input_image, mask_generator)
    return result, mask_all


def predictor_inference(device, input_image, prompt_points):
    predictor = get_image_predicator(device)
    predictor.set_image(input_image)
    transformed_boxes = None

    if len(prompt_points) != 0:
        points_value = [p for p, _ in prompt_points]
        points = torch.Tensor(points_value).to(device).unsqueeze(1)
        lables_value = [int(l) for _, l in prompt_points]
        labels = torch.Tensor(lables_value).to(device).unsqueeze(1)
        transformed_points = points
    else:
        transformed_points, labels = None, None
    
    
    input_points = np.array([(p[0][0], p[0][1]) for p in prompt_points])
    input_labels = np.array([p[1] for p in prompt_points])
    box = None
    
    masks, scores, logits = predictor.predict(
        point_coords = input_points,
        point_labels = input_labels,
        box = box,
        multimask_output=False
    )
    
    mask = masks[0]    
    mask_all = np.ones((input_image.shape[0], input_image.shape[1], 3))
    # color_mask = np.random.random((1, 3)).tolist()[0]
    for i in range(3):
        mask_all[mask == True, i] = 1
        mask_all[mask == False, i] = 0.3
    img = input_image * mask_all / 255
    gc.collect()
    torch.cuda.empty_cache()
    
    mask_all = np.zeros((input_image.shape[0], input_image.shape[1]))
    mask_all[mask == True] = 1
    masked_image = np.zeros((input_image.shape[0], input_image.shape[1], 4))
    masked_image[:,:,0:3] = input_image / 255
    masked_image[:,:,3] = mask_all
    return img, masked_image


def image_inference(device, input_image, prompt_points=[]):
    if len(prompt_points) != 0:
        return predictor_inference(device, input_image, prompt_points)
    else:
        return generator_inference(device, input_image)
