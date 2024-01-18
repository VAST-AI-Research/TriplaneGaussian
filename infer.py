import torch
from dataclasses import dataclass, field
from einops import rearrange
import os
from torch.utils.data import DataLoader

import tgs
from tgs.models.image_feature import ImageFeature
from tgs.utils.saving import SaverMixin
from tgs.utils.config import parse_structured
from tgs.utils.ops import points_projection
from tgs.utils.misc import load_module_weights
from tgs.utils.typing import *

class TGS(torch.nn.Module, SaverMixin):
    @dataclass
    class Config:
        weights: Optional[str] = None
        weights_ignore_modules: Optional[List[str]] = None

        camera_embedder_cls: str = ""
        camera_embedder: dict = field(default_factory=dict)

        image_feature: dict = field(default_factory=dict)

        image_tokenizer_cls: str = ""
        image_tokenizer: dict = field(default_factory=dict)

        tokenizer_cls: str = ""
        tokenizer: dict = field(default_factory=dict)

        backbone_cls: str = ""
        backbone: dict = field(default_factory=dict)

        post_processor_cls: str = ""
        post_processor: dict = field(default_factory=dict)

        renderer_cls: str = ""
        renderer: dict = field(default_factory=dict)

        pointcloud_generator_cls: str = ""
        pointcloud_generator: dict = field(default_factory=dict)

        pointcloud_encoder_cls: str = ""
        pointcloud_encoder: dict = field(default_factory=dict)
    
    cfg: Config

    def load_weights(self, weights: str, ignore_modules: Optional[List[str]] = None):
        state_dict = load_module_weights(
            weights, ignore_modules=ignore_modules, map_location="cpu"
        )
        self.load_state_dict(state_dict, strict=False)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self._save_dir: Optional[str] = None

        self.image_tokenizer = tgs.find(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        )

        assert self.cfg.camera_embedder_cls == 'tgs.models.networks.MLP'
        weights = self.cfg.camera_embedder.pop("weights") if "weights" in self.cfg.camera_embedder else None
        self.camera_embedder = tgs.find(self.cfg.camera_embedder_cls)(**self.cfg.camera_embedder)
        if weights:
            from tgs.utils.misc import load_module_weights
            weights_path, module_name = weights.split(":")
            state_dict = load_module_weights(
                weights_path, module_name=module_name, map_location="cpu"
            )
            self.camera_embedder.load_state_dict(state_dict)

        self.image_feature = ImageFeature(self.cfg.image_feature)

        self.tokenizer = tgs.find(self.cfg.tokenizer_cls)(self.cfg.tokenizer)

        self.backbone = tgs.find(self.cfg.backbone_cls)(self.cfg.backbone)

        self.post_processor = tgs.find(self.cfg.post_processor_cls)(
            self.cfg.post_processor
        )

        self.renderer = tgs.find(self.cfg.renderer_cls)(self.cfg.renderer)

        # pointcloud generator
        self.pointcloud_generator = tgs.find(self.cfg.pointcloud_generator_cls)(self.cfg.pointcloud_generator)

        self.point_encoder = tgs.find(self.cfg.pointcloud_encoder_cls)(self.cfg.pointcloud_encoder)

        # load checkpoint
        if self.cfg.weights is not None:
            self.load_weights(self.cfg.weights, self.cfg.weights_ignore_modules)
    
    def _forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # generate point cloud
        out = self.pointcloud_generator(batch)
        pointclouds = out["points"]

        batch_size, n_input_views = batch["rgb_cond"].shape[:2]

        # Camera modulation
        camera_extri = batch["c2w_cond"].view(*batch["c2w_cond"].shape[:-2], -1)
        camera_intri = batch["intrinsic_normed_cond"].view(*batch["intrinsic_normed_cond"].shape[:-2], -1)
        camera_feats = torch.cat([camera_intri, camera_extri], dim=-1)

        camera_feats = self.camera_embedder(camera_feats)

        input_image_tokens: Float[Tensor, "B Cit Nit"] = self.image_tokenizer(
            rearrange(batch["rgb_cond"], 'B Nv H W C -> B Nv C H W'),
            modulation_cond=camera_feats,
        )
        input_image_tokens = rearrange(input_image_tokens, 'B Nv C Nt -> B (Nv Nt) C', Nv=n_input_views)

        # get image features for projection
        image_features = self.image_feature(
            rgb = batch["rgb_cond"],
            mask = batch.get("mask_cond", None),
            feature = input_image_tokens
        )

        # only support number of input view is one
        c2w_cond = batch["c2w_cond"].squeeze(1)
        intrinsic_cond = batch["intrinsic_cond"].squeeze(1)
        proj_feats = points_projection(pointclouds, c2w_cond, intrinsic_cond, image_features)

        point_cond_embeddings = self.point_encoder(torch.cat([pointclouds, proj_feats], dim=-1))
        tokens: Float[Tensor, "B Ct Nt"] = self.tokenizer(batch_size, cond_embeddings=point_cond_embeddings)

        tokens = self.backbone(
            tokens,
            encoder_hidden_states=input_image_tokens,
            modulation_cond=None,
        )

        scene_codes = self.post_processor(self.tokenizer.detokenize(tokens))
        rend_out = self.renderer(scene_codes,
                                query_points=pointclouds,
                                additional_features=proj_feats,
                                **batch)

        return {**out, **rend_out}
    
    def forward(self, batch):
        out = self._forward(batch)
        batch_size = batch["index"].shape[0]
        for b in range(batch_size):
            if batch["view_index"][b, 0] == 0:
                out["3dgs"][b].save_ply(self.get_save_path(f"3dgs/{batch['instance_id'][b]}.ply"))

            for index, render_image in enumerate(out["comp_rgb"][b]):
                view_index = batch["view_index"][b, index]
                self.save_image_grid(
                    f"video/{batch['instance_id'][b]}/{view_index}.png",
                    [
                        {
                            "type": "rgb",
                            "img": render_image,
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                )
        

if __name__ == "__main__":
    import argparse
    import subprocess
    from tgs.utils.config import ExperimentConfig, load_config
    from tgs.data import CustomImageOrbitDataset
    from tgs.utils.misc import todevice, get_device

    parser = argparse.ArgumentParser("Triplane Gaussian Splatting")
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--out", default="outputs", help="path to output folder")
    parser.add_argument("--cam_dist", default=1.9, type=float, help="distance between camera center and scene center")
    parser.add_argument("--image_preprocess", action="store_true", help="whether to segment the input image by rembg and SAM")
    args, extras = parser.parse_known_args()

    device = get_device()

    cfg: ExperimentConfig = load_config(args.config, cli_args=extras)
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(repo_id="VAST-AI/TriplaneGaussian", local_dir="./checkpoints", filename="model_lvis_rel.ckpt", repo_type="model")
    # model_path = "checkpoints/model_lvis_rel.ckpt"
    cfg.system.weights=model_path
    model = TGS(cfg=cfg.system).to(device)
    model.set_save_dir(args.out)
    print("load model ckpt done.")

    # run image segmentation for images
    if args.image_preprocess:
        segmented_image_list = []
        for image_path in cfg.data.image_list:
            filepath, ext = os.path.splitext(image_path)
            save_path = os.path.join(filepath + "_rgba.png")
            segmented_image_list.append(save_path)
            subprocess.run([f"python image_preprocess/run_sam.py --image_path {image_path} --save_path {save_path}"], shell=True)
        cfg.data.image_list = segmented_image_list

    cfg.data.cond_camera_distance = args.cam_dist
    cfg.data.eval_camera_distance = args.cam_dist
    dataset = CustomImageOrbitDataset(cfg.data)
    dataloader = DataLoader(dataset,
                        batch_size=cfg.data.eval_batch_size, 
                        num_workers=cfg.data.num_workers,
                        shuffle=False,
                        collate_fn=dataset.collate
                    )

    for batch in dataloader:
        batch = todevice(batch)
        model(batch)
    
    model.save_img_sequences(
        "video",
        "(\d+)\.png",
        save_format="mp4",
        fps=30,
        delete=True,
    )