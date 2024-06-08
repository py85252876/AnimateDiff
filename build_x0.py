import argparse
import datetime
import inspect
import os
from omegaconf import OmegaConf
import pandas as pd
import torchvision
import torch
import torchvision.transforms as transforms

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.models.sparse_controlnet import SparseControlNetModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.util import load_weights
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange, repeat

import csv, pdb, glob, math
from pathlib import Path
from PIL import Image
import numpy as np

def copy_videos_by_group(csv_file_path):
    
    df = pd.read_csv(csv_file_path)
    data = {}
    for col in df.columns:
        
        # 获取列中的非空值，这些值是视频的编号
        video_numbers = df[col].dropna().astype(int).tolist()
        data[f"{col}"] = video_numbers

    return data


@torch.no_grad()
def main(args,labels):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)
    
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"samples/{Path(args.config).stem}-{time_str}"
    os.makedirs(savedir)

    config  = OmegaConf.load(args.config)
    samples = []

    # create validation pipeline
    tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").cuda()
    vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").cuda()

    sample_idx = 0
    for model_idx, model_config in enumerate(config):
        inference_config = OmegaConf.load(model_config.get("inference_config", args.inference_config))
        unet = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)).cuda()

        # load controlnet model
        controlnet = controlnet_images = None
        if model_config.get("controlnet_path", "") != "":
            assert model_config.get("controlnet_images", "") != ""
            assert model_config.get("controlnet_config", "") != ""
            
            unet.config.num_attention_heads = 8
            unet.config.projection_class_embeddings_input_dim = None

            controlnet_config = OmegaConf.load(model_config.controlnet_config)
            controlnet = SparseControlNetModel.from_unet(unet, controlnet_additional_kwargs=controlnet_config.get("controlnet_additional_kwargs", {}))

            print(f"loading controlnet checkpoint from {model_config.controlnet_path} ...")
            controlnet_state_dict = torch.load(model_config.controlnet_path, map_location="cpu")
            controlnet_state_dict = controlnet_state_dict["controlnet"] if "controlnet" in controlnet_state_dict else controlnet_state_dict
            controlnet_state_dict.pop("animatediff_config", "")
            controlnet.load_state_dict(controlnet_state_dict)
            controlnet.cuda()

            image_paths = model_config.controlnet_images
            if isinstance(image_paths, str): image_paths = [image_paths]

            print(f"controlnet image paths:")
            for path in image_paths: print(path)
            assert len(image_paths) <= model_config.L

            image_transforms = transforms.Compose([
                transforms.RandomResizedCrop(
                    (model_config.H, model_config.W), (1.0, 1.0), 
                    ratio=(model_config.W/model_config.H, model_config.W/model_config.H)
                ),
                transforms.ToTensor(),
            ])

            if model_config.get("normalize_condition_images", False):
                def image_norm(image):
                    image = image.mean(dim=0, keepdim=True).repeat(3,1,1)
                    image -= image.min()
                    image /= image.max()
                    return image
            else: image_norm = lambda x: x
                
            controlnet_images = [image_norm(image_transforms(Image.open(path).convert("RGB"))) for path in image_paths]

            os.makedirs(os.path.join(savedir, "control_images"), exist_ok=True)
            for i, image in enumerate(controlnet_images):
                Image.fromarray((255. * (image.numpy().transpose(1,2,0))).astype(np.uint8)).save(f"{savedir}/control_images/{i}.png")

            controlnet_images = torch.stack(controlnet_images).unsqueeze(0).cuda()
            controlnet_images = rearrange(controlnet_images, "b f c h w -> b c f h w")

            if controlnet.use_simplified_condition_embedding:
                num_controlnet_images = controlnet_images.shape[2]
                controlnet_images = rearrange(controlnet_images, "b c f h w -> (b f) c h w")
                controlnet_images = vae.encode(controlnet_images * 2. - 1.).latent_dist.sample() * 0.18215
                controlnet_images = rearrange(controlnet_images, "(b f) c h w -> b c f h w", f=num_controlnet_images)

        # set xformers
        if is_xformers_available() and (not args.without_xformers):
            unet.enable_xformers_memory_efficient_attention()
            if controlnet is not None: controlnet.enable_xformers_memory_efficient_attention()

        pipeline = AnimationPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            controlnet=controlnet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
        ).to("cuda")

        pipeline = load_weights(
            pipeline,
            # motion module
            motion_module_path         = model_config.get("motion_module", ""),
            motion_module_lora_configs = model_config.get("motion_module_lora_configs", []),
            # domain adapter
            adapter_lora_path          = model_config.get("adapter_lora_path", ""),
            adapter_lora_scale         = model_config.get("adapter_lora_scale", 1.0),
            # image layers
            dreambooth_model_path      = model_config.get("dreambooth_path", ""),
            lora_model_path            = model_config.get("lora_model_path", ""),
            lora_alpha                 = model_config.get("lora_alpha", 0.8),
        ).to("cuda")

    # data  = torch.load("/bigtemp/trv3px/video_detection/AnimateDiff/predx_0/2000-experiment-1500.pt")  
    # data_1 = torch.load("/bigtemp/trv3px/video_detection/AnimateDiff/predx_0/2000-experiment-500.pt")
    # data = torch.cat((data, data_1), dim=0)
    # print(data.shape)
    # for key in list(labels.keys())[:2]:
    #     temp_data = labels[key]
    #     for temp in temp_data:
    #         for i in range(50):
    #             x_0 = data[temp:temp+1,i:i+1,:,:,:,:,].squeeze(0).to("cuda")
    #             video = pipeline.decode_latents(x_0)
    #             video = torch.from_numpy(video)
    #             # video = torch.from_numpy(video)
    #             # videos = rearrange(video, "b c t h w -> t b c h w")
    #             # outputs = []
    #             # rescale = False
    #             path = f"/bigtemp/trv3px/video_detection/AnimateDiff/train_detector/{key}/{temp}/{i}.mp4"
    #             # for x in videos:
    #             #     x = torchvision.utils.make_grid(x, nrow=6)
    #             #     x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
    #             #     if rescale:
    #             #         x = (x + 1.0) / 2.0  # -1,1 -> 0,1
    #             #     x = (x * 255).numpy().astype(np.uint8)
    #             #     outputs.append(x)

    #             os.makedirs(os.path.dirname(path), exist_ok=True)
    #             save_videos_grid(video, path)
    data = torch.load("/scratch/trv3px/video_detection/AnimateDiff/predx_0/unsafe.pt")
    for temp in range(200):
        for i in range(50):
            x_0 = data[temp:temp+1,i:i+1,:,:,:,:,].squeeze(0).to("cuda")
            video = pipeline.decode_latents(x_0)
            video = torch.from_numpy(video)
            # video = torch.from_numpy(video)
            # videos = rearrange(video, "b c t h w -> t b c h w")
            # outputs = []
            # rescale = False
            path = f"/scratch/trv3px/video_detection/AnimateDiff/unsafe/{i}/{temp}.mp4"
            # for x in videos:
            #     x = torchvision.utils.make_grid(x, nrow=6)
            #     x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            #     if rescale:
            #         x = (x + 1.0) / 2.0  # -1,1 -> 0,1
            #     x = (x * 255).numpy().astype(np.uint8)
            #     outputs.append(x)

            os.makedirs(os.path.dirname(path), exist_ok=True)
            save_videos_grid(video, path)

    # data  = torch.load("/bigtemp/trv3px/video_detection/AnimateDiff/predx_0/2000-experiment-1500.pt")  
    # data_1 = torch.load("/bigtemp/trv3px/video_detection/AnimateDiff/predx_0/2000-experiment-500.pt")
    # data = torch.cat((data, data_1), dim=0)
    # print(data.shape)
    # for key in list(labels.keys())[:2]:
    #     temp_data = labels[key]
    #     for temp in temp_data:
    #         for i in range(50):
    #             x_0 = data[temp:temp+1,i:i+1,:,:,:,:,].squeeze(0).to("cuda")
    #             video = pipeline.decode_latents(x_0)
    #             video = torch.from_numpy(video)
    #             path = f"/bigtemp/trv3px/video_detection/AnimateDiff/train_detector/{key}/{temp}/{i}.mp4"
    #             os.makedirs(os.path.dirname(path), exist_ok=True)
    #             save_videos_grid(video, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, default="models/StableDiffusion/stable-diffusion-v1-5",)
    parser.add_argument("--inference-config",      type=str, default="configs/inference/inference-v1.yaml")    
    parser.add_argument("--config",                type=str, required=True)

    parser.add_argument("--without-xformers", action="store_true")

    args = parser.parse_args()
    # labels = copy_videos_by_group("/bigtemp/trv3px/video_detection/AnimateDiff/train_detector/animate_output.csv")
    labels = []
    main(args, labels)
