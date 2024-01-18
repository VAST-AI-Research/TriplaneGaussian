from utils import image_preprocess, pred_bbox, sam_init, sam_out_nosave, resize_image
import os
from PIL import Image
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--ckpt_path", default="./checkpoints/sam_vit_h_4b8939.pth")
    args = parser.parse_args()

    # load SAM checkpoint
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    sam_predictor = sam_init(args.ckpt_path, gpu)
    print("load sam ckpt done.")

    input_raw = Image.open(args.image_path)
    # input_raw.thumbnail([512, 512], Image.Resampling.LANCZOS)
    input_raw = resize_image(input_raw, 512)
    image_sam = sam_out_nosave(
        sam_predictor, input_raw.convert("RGB"), pred_bbox(input_raw)
    )

    image_preprocess(image_sam, args.save_path, lower_contrast=False, rescale=True)
