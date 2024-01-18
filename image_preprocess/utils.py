import cv2
import numpy as np
import torch
from PIL import Image
from rembg import remove
from segment_anything import SamPredictor, sam_model_registry

def sam_init(sam_checkpoint, device_id=0):
    model_type = "vit_h"

    device = "cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device)
    predictor = SamPredictor(sam)
    return predictor


def sam_out_nosave(predictor, input_image, *bbox_sliders):
    bbox = np.array(bbox_sliders)
    image = np.asarray(input_image)

    predictor.set_image(image)

    masks_bbox, scores_bbox, logits_bbox = predictor.predict(
        box=bbox, multimask_output=True
    )

    out_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    out_image[:, :, :3] = image
    out_image_bbox = out_image.copy()
    out_image_bbox[:, :, 3] = (
        masks_bbox[-1].astype(np.uint8) * 255
    )  # np.argmax(scores_bbox)
    torch.cuda.empty_cache()
    return Image.fromarray(out_image_bbox, mode="RGBA")


# contrast correction, rescale and recenter
def image_preprocess(input_image, save_path, lower_contrast=True, rescale=True):
    image_arr = np.array(input_image)
    in_w, in_h = image_arr.shape[:2]

    if lower_contrast:
        alpha = 0.8  # Contrast control (1.0-3.0)
        beta = 0  # Brightness control (0-100)
        # Apply the contrast adjustment
        image_arr = cv2.convertScaleAbs(image_arr, alpha=alpha, beta=beta)
        image_arr[image_arr[..., -1] > 200, -1] = 255

    ret, mask = cv2.threshold(
        np.array(input_image.split()[-1]), 0, 255, cv2.THRESH_BINARY
    )
    x, y, w, h = cv2.boundingRect(mask)
    max_size = max(w, h)
    ratio = 0.75
    if rescale:
        side_len = int(max_size / ratio)
    else:
        side_len = in_w
    padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
    center = side_len // 2
    padded_image[
        center - h // 2 : center - h // 2 + h, center - w // 2 : center - w // 2 + w
    ] = image_arr[y : y + h, x : x + w]
    rgba = Image.fromarray(padded_image).resize((256, 256), Image.LANCZOS)
    rgba.save(save_path)

def pred_bbox(image):
    image_nobg = remove(image.convert("RGBA"), alpha_matting=True)
    alpha = np.asarray(image_nobg)[:, :, -1]
    x_nonzero = np.nonzero(alpha.sum(axis=0))
    y_nonzero = np.nonzero(alpha.sum(axis=1))
    x_min = int(x_nonzero[0].min())
    y_min = int(y_nonzero[0].min())
    x_max = int(x_nonzero[0].max())
    y_max = int(y_nonzero[0].max())
    return x_min, y_min, x_max, y_max

def resize_image(input_raw, size):
    w, h = input_raw.size
    ratio = size / max(w, h)
    resized_w = int(w * ratio)
    resized_h = int(h * ratio)
    return input_raw.resize((resized_w, resized_h), Image.Resampling.LANCZOS)