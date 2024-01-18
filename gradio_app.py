import argparse
import os
import glob
import torch
from torch.utils.data import DataLoader
from copy import deepcopy
import tempfile
from functools import partial

CACHE_EXAMPLES = os.environ.get("CACHE_EXAMPLES", "0") == "1"
DEFAULT_CAM_DIST = 1.9

import gradio as gr
from image_preprocess.utils import image_preprocess, resize_image, sam_out_nosave, pred_bbox, sam_init
from gradio_splatting.backend.gradio_model3dgs import Model3DGS
from tgs.data import CustomImageOrbitDataset
from tgs.utils.misc import todevice
from tgs.utils.config import ExperimentConfig, load_config
from infer import TGS

from huggingface_hub import hf_hub_download
MODEL_CKPT_PATH = hf_hub_download(repo_id="VAST-AI/TriplaneGaussian", local_dir="./checkpoints", filename="model_lvis_rel.ckpt", repo_type="model")
# MODEL_CKPT_PATH = "checkpoints/model_lvis_rel.ckpt"
SAM_CKPT_PATH = "checkpoints/sam_vit_h_4b8939.pth"
CONFIG = "config.yaml"
EXP_ROOT_DIR = "./outputs-gradio"

os.makedirs(EXP_ROOT_DIR, exist_ok=True)

gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
device = "cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu"

print("device: ", device)

# init model
base_cfg: ExperimentConfig
base_cfg = load_config(CONFIG, cli_args=[], n_gpus=1)
base_cfg.system.weights = MODEL_CKPT_PATH
model = TGS(cfg=base_cfg.system).to(device)
print("load model ckpt done.")

HEADER = """
# Triplane Meets Gaussian Splatting: Fast and Generalizable Single-View 3D Reconstruction with Transformers

<div>
<a style="display: inline-block;" href="https://arxiv.org/abs/2312.09147"><img src="https://img.shields.io/badge/arxiv-2312.09147-B31B1B.svg"></a>
</div>

TGS enables fast reconstruction from single-view image in a few seconds based on a hybrid Triplane-Gaussian 3D representation.

This model is trained on Objaverse-LVIS (**~45K** synthetic objects) only. And note that we normalize the input camera pose to a pre-set viewpoint during training stage following LRM, rather than directly using camera pose of input camera as implemented in our original paper.

**Tips:**
1. If you find the result is unsatisfied, please try to change the camera distance. It perhaps improves the results.

**Notes:**
1. Please wait until the completion of the reconstruction of the previous model before proceeding with the next one, otherwise, it may cause bug. We will fix it soon.
2. We currently conduct image segmentation (SAM) by invoking subprocess, which consumes more time as it requires loading SAM checkpoint each time. We have observed that directly running SAM in app.py often leads to queue blocking, but we haven't identified the cause yet. We plan to fix this issue for faster segmentation running time later. 
"""

def assert_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image selected or uploaded!")

def preprocess(input_raw, sam_predictor=None):
    save_path = model.get_save_path("seg_rgba.png")
    input_raw = resize_image(input_raw, 512)
    image_sam = sam_out_nosave(
        sam_predictor, input_raw.convert("RGB"), pred_bbox(input_raw)
    )
    image_preprocess(image_sam, save_path, lower_contrast=False, rescale=True)
    return save_path

def init_trial_dir():
    trial_dir = tempfile.TemporaryDirectory(dir=EXP_ROOT_DIR).name
    model.set_save_dir(trial_dir)
    return trial_dir

@torch.no_grad()
def infer(image_path: str,
          cam_dist: float,
          only_3dgs: bool = False):
    data_cfg = deepcopy(base_cfg.data)
    data_cfg.only_3dgs = only_3dgs
    data_cfg.cond_camera_distance = cam_dist
    data_cfg.eval_camera_distance = cam_dist
    data_cfg.image_list = [image_path]
    dataset = CustomImageOrbitDataset(data_cfg)
    dataloader = DataLoader(dataset,
                batch_size=data_cfg.eval_batch_size, 
                num_workers=data_cfg.num_workers,
                shuffle=False,
                collate_fn=dataset.collate
            )

    for batch in dataloader:
        batch = todevice(batch, device)
        model(batch)
    if not only_3dgs:
        model.save_img_sequences(
            "video",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            delete=True,
        )

def run(image_path: str,
        cam_dist: float,
        save_path: str):
    infer(image_path, cam_dist, only_3dgs=True)
    gs = glob.glob(os.path.join(save_path, "3dgs", "*.ply"))[0]
    return gs

def run_video(image_path: str,
            cam_dist: float,
            save_path: str):
    infer(image_path, cam_dist)
    video = glob.glob(os.path.join(save_path, "video", "*.mp4"))[0]
    return video

def run_example(image_path, sam_predictor=None):
    save_path = init_trial_dir()
    seg_image_path = preprocess(image_path, sam_predictor)
    gs = run(seg_image_path, DEFAULT_CAM_DIST, save_path)
    video = run_video(seg_image_path, DEFAULT_CAM_DIST, save_path)
    return seg_image_path, gs, video

def launch(port):
    sam_predictor = sam_init(SAM_CKPT_PATH, gpu)
    print("load sam ckpt done.")

    with gr.Blocks(
        title="TGS - Demo"
    ) as demo:
        with gr.Row(variant='panel'):
            gr.Markdown(HEADER)
    
        with gr.Row(variant='panel'):
            with gr.Column(scale=1):
                input_image = gr.Image(value=None, image_mode="RGB", width=512, height=512, type="pil", sources="upload", label="Input Image")
                gr.Markdown(
                    """
                    **Camera distance** denotes the distance between camera center and scene center.
                    If you find the 3D model appears flattened, you can increase it. Conversely, if the 3D model appears thick, you can decrease it.
                    """
                )
                camera_dist_slider = gr.Slider(1.0, 4.0, value=DEFAULT_CAM_DIST, step=0.1, label="Camera Distance")
                img_run_btn = gr.Button("Reconstruction", variant="primary")

            with gr.Column(scale=1):
                with gr.Row(variant='panel'):
                    seg_image = gr.Image(value=None, width="auto", type="filepath", image_mode="RGBA", label="Segmented Image", interactive=False)
                    output_video = gr.Video(value=None, width="auto", label="Rendered Video", autoplay=True)
                output_3dgs = Model3DGS(value=None, label="3D Model")
        
        with gr.Row(variant="panel"):
            gr.Examples(
                examples=[
                    "example_images/green_parrot.webp",
                    "example_images/rusty_gameboy.webp",
                    "example_images/a_pikachu_with_smily_face.webp",
                    "example_images/an_otter_wearing_sunglasses.webp",
                    "example_images/lumberjack_axe.webp",
                    "example_images/medieval_shield.webp",
                    "example_images/a_cat_dressed_as_the_pope.webp",
                    "example_images/a_cute_little_frog_comicbook_style.webp",
                    "example_images/a_purple_winter_jacket.webp",
                    "example_images/MP5,_high_quality,_ultra_realistic.webp",
                    "example_images/retro_pc_photorealistic_high_detailed.webp",
                    "example_images/stratocaster_guitar_pixar_style.webp"
                ],
                inputs=[input_image],
                outputs=[seg_image, output_3dgs, output_video],
                cache_examples=CACHE_EXAMPLES,
                fn=partial(run_example, sam_predictor=sam_predictor),
                label="Examples",
                examples_per_page=40
            )

        trial_dir = gr.State()
        img_run_btn.click(
            fn=assert_input_image,
            inputs=[input_image],
        ).success(
            fn=init_trial_dir,
            outputs=[trial_dir],
        ).success(
            fn=partial(preprocess, sam_predictor=sam_predictor),
            inputs=[input_image],
            outputs=[seg_image],
        ).success(fn=run,
                inputs=[seg_image, camera_dist_slider, trial_dir],
                outputs=[output_3dgs],
        ).success(fn=run_video,
                inputs=[seg_image, camera_dist_slider, trial_dir],
                outputs=[output_video])

        launch_args = {"server_port": port}
        demo.queue(max_size=10)
        demo.launch(**launch_args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args, extra = parser.parse_known_args()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    launch(args.port)