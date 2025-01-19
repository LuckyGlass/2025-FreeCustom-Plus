import argparse
import os
import sys

import numpy as np
import json
import torch
from PIL import Image

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))


# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt

SAM_CHECKPOINT="sam_vit_h_4b8939.pth"
SAM_VERSION="vit_h"
GROUNDED_CHECKPOINT="groundingdino_swint_ogc.pth"
SAM_HQ_CHECKPOINT=None #default
USE_SAM_HQ=False #default
CONFIG="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
BOX_THRESHOLD=0.3
TEXT_THRESHOLD=0.25
DEVICE="cuda"
BERT_BASE_UNCASED_PATH=None #default
OUTPUT_DIR="outputs"


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image
    #print("image_pil.size: ", image_pil.size)
    
    # Resize the image to 512x512
    #image_pil_resized = image_pil.resize((512, 512))  # Resize image
    image_pil_resized = image_pil
    #print("image_pil_resized.size: ", image_pil_resized.size)
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil_resized, None)  # 3, h, w
    return image_pil_resized, image


def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        #mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
        mask_img[mask.cpu().numpy()[0] == 1] = value + idx + 1
    for idx, mask in enumerate(mask_list):
        #mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)

'''
generate_masks: 用于生成图片中的目标概念对象的mask
input: 
    image_path: 输入图片路径,str
    text_prompt: 文本提示,str
output: 
    masks: 生成的mask,就一个,shape为(1, 128, 128)
'''
def generate_masks(image_path, text_prompt):

    # cfg
    config_file = CONFIG  # change the path of the model config file
    grounded_checkpoint = GROUNDED_CHECKPOINT  # change the path of the model
    sam_version = SAM_VERSION
    sam_checkpoint = SAM_CHECKPOINT
    sam_hq_checkpoint = SAM_HQ_CHECKPOINT
    use_sam_hq = USE_SAM_HQ
    output_dir = OUTPUT_DIR
    box_threshold = BOX_THRESHOLD
    text_threshold = TEXT_THRESHOLD
    device = DEVICE
    bert_base_uncased_path = BERT_BASE_UNCASED_PATH

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil, image = load_image(image_path)
    #print("image.shape: ", image.shape)
    # load model
    model = load_model(config_file, grounded_checkpoint, bert_base_uncased_path, device=device)

    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, device=device
    )

    # initialize SAM
    if use_sam_hq:
        predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    #print("H: ", H) #4395
    #print("W: ", W) #4395
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )

    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    #print("masks[0].shape: ", masks[0].shape)
    #print("masks[0]: ", masks[0].cpu().numpy()) 
    
    # Resize each mask from 512x512 to 128x128 using cv2
    masks_resized = []
    
    
    for mask in masks:
        # Convert mask to uint8 before resizing
        #mask_resized = cv2.resize(mask.cpu().numpy()[0].astype(np.uint8), (128, 128))  # Resize to 128x128
        mask_resized = cv2.resize(mask.cpu().numpy()[0].astype(np.uint8), (128, 128))
        mask_resized = np.expand_dims(mask_resized, axis=0)  # Shape becomes (1, H, W)
        masks_resized.append(mask_resized)
    
    
    #print("masks_resized[0].shape: ", masks_resized[0].shape)
    #print("masks_resized[0]: ", masks_resized[0]) 

    for mask in masks_resized:
        show_mask(mask, plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)

    plt.axis('off')
    plt.savefig(
        os.path.join(output_dir, "grounded_sam_output.jpg"),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )
    
    # Convert list of masks to a tensor
    masks_resized_tensor = torch.stack([torch.tensor(mask) for mask in masks_resized])
    #print("masks_resized_tensor.shape: ", masks_resized_tensor.shape)
    #print("masks_resized_tensor: ", masks_resized_tensor)
    
    save_mask_data(output_dir, masks_resized_tensor, boxes_filt, pred_phrases)

    # 返回第一个mask,其实总共就1个mask,shape为(1, 128, 128)
    return masks_resized_tensor[0]


if __name__ == "__main__":
    mask=generate_masks("../FreeCustom/dataset/freecustom/multi_concept/cat_hinton/image/cat.jpg", "cat")
    #print("mask.shape: ", mask.shape)
    #print("mask: ", mask)