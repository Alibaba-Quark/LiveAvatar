# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
import json
import time
import subprocess
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from decord import VideoReader
from PIL import Image
import torch.nn.functional as F
from safetensors import safe_open
from torchvision import transforms
from tqdm import tqdm
from peft import LoraConfig, inject_adapter_in_model
import subprocess
from diffusers import FlowMatchEulerDiscreteScheduler
from .wan_2_2.distributed.fsdp import shard_model
from .wan_2_2.distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .wan_2_2.distributed.util import get_world_size
from .causal_audio_encoder import AudioEncoder
from .causal_audio_encoder import AudioEncoder_Training
from .causal_model_s2v import CausalWanModel_S2V, sp_attn_forward_s2v
from .wan_2_2.modules.t5 import T5EncoderModel
from .wan_2_2.modules.vae2_1 import Wan2_1_VAE
from .wan_2_2.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .wan_2_2.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from ...utils.load_weight_utils import load_state_dict
from liveavatar.utils.router.utils import process_masks_to_routing_logits



def load_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


class WanS2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        sp_size=None,
        t5_cpu=False,
        init_on_cpu=True,
        drop_part_motion_frames=0.3,
        convert_model_dtype=False,
        is_training=False,
        single_gpu=False,
        offload_kv_cache=False
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu
        self.is_training = is_training
        self.num_train_timesteps = config.num_train_timesteps # 1000
        self.num_frames_per_block = config.num_frames_per_block
        self.param_dtype = config.param_dtype
        self.checkpoint_dir = checkpoint_dir
        self.drop_part_motion_frames = drop_part_motion_frames
        self.single_gpu = single_gpu
        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device,dtype=self.param_dtype)

        if self.is_training:
            from liveavatar.models.wan.flow_match import FlowMatchScheduler_Omni
            self.scheduler = FlowMatchScheduler_Omni(shift=5, sigma_min=0.0, extra_one_step=True)
            self.scheduler.set_timesteps(1000, training=True)
        else:
            if config.sample_solver == 'euler':
                self.sample_scheduler = FlowMatchEulerDiscreteScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=3)
            elif config.sample_solver == 'unipc':#default
                self.sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
            elif config.sample_solver == 'dpm++':
                self.sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
            else:
                raise NotImplementedError("Unsupported solver.")

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        if not dit_fsdp:
            self.noise_model = CausalWanModel_S2V.from_pretrained(
                checkpoint_dir,
                torch_dtype=self.param_dtype,
                device_map=self.device)
        else:
            self.noise_model = CausalWanModel_S2V.from_pretrained(
                checkpoint_dir, torch_dtype=self.param_dtype)
        
        self.noise_model.freqs.to(device=self.device)

        self.noise_model = self._configure_model(
            model=self.noise_model,
            use_sp=use_sp,
            sp_size=sp_size,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)
        self.noise_model.num_frame_per_block = self.num_frames_per_block

        if not self.is_training:
            self.audio_encoder = AudioEncoder(
                model_id=os.path.join(checkpoint_dir,
                                    "wav2vec2-large-xlsr-53-english"))
        else:
            self.audio_encoder = AudioEncoder_Training(
                model_id=os.path.join(checkpoint_dir
                ,
                                    "wav2vec2-large-xlsr-53-english"))

        if use_sp:
            self.sp_size = sp_size if sp_size is not None else get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt
        self.motion_frames = config.transformer.motion_frames
        self.drop_first_motion = config.drop_first_motion
        self.fps = config.sample_fps
        self.audio_sample_m = 0
        self.tgt_gpu_id = 0
    
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None, load_only=False, load_lora_weight_only=False):
        if not load_only:
            self.lora_alpha = lora_alpha
            if init_lora_weights == "kaiming":
                init_lora_weights = True
                
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                init_lora_weights=init_lora_weights,
                target_modules=lora_target_modules.split(","),
            )
            model = inject_adapter_in_model(lora_config, model)
                
        if pretrained_lora_path is not None:
            ori_pretrained_lora_path = pretrained_lora_path
            state_dict = load_state_dict(pretrained_lora_path)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            if load_lora_weight_only:
                new_state_dict = {}
                for key in state_dict.keys():
                    if 'lora' in key:
                        new_state_dict[key] = state_dict[key]
                state_dict = new_state_dict
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {ori_pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")
    
    def set_all_model_to_dtype_device(self, dtype, device):
        models = [
            self.noise_model,
            self.text_encoder.model,
            self.vae.model,
            self.audio_encoder.model
        ]
        
        for model in models:
            for param in model.parameters():
                param.data = param.data.to(device=device, dtype=dtype)
                if param.grad is not None:
                    param.grad = param.grad.to(device=device, dtype=dtype)
            
    
    def set_requires_grad(self, requires_grad=True):
        self.requires_grad = requires_grad
        self.noise_model.requires_grad_(requires_grad)
        self.text_encoder.model.requires_grad_(requires_grad)
        self.vae.model.requires_grad_(requires_grad)
        self.audio_encoder.model.requires_grad_(requires_grad)
    
    def set_eval(self):
        self.noise_model.eval()
        self.text_encoder.model.eval()
        self.vae.model.eval()
        self.audio_encoder.model.eval()
    
    def set_train(self):
        self.noise_model.train()
        self.text_encoder.model.train()
        self.vae.model.train()
        self.audio_encoder.model.train()
    
    def set_device_dtype(self, device, dtype):
        self.noise_model.to(device, dtype)
        self.text_encoder.model.to(device, dtype)
        self.vae.model.to(device, dtype)
        self.audio_encoder.model.to(device, dtype)
    
    def _configure_model(self, model, use_sp, sp_size, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            sp_size (`int`):
                Sequence parallel size to use instead of world_size.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)
        if use_sp:
            # Store sp_size in model for use in forward pass
            model.sp_size = sp_size
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward_s2v, block.self_attn)
                # Store sp_size in each block for access in forward pass
                block.sp_size = sp_size
            model.use_context_parallel = True

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def _process_timestep(self, timestep: torch.Tensor, type: str="causal_video") -> torch.Tensor:
        """
        copy from liveavatar/dmd.py:
        Pre-process the randomly generated timestep based on the generator's task type.
        Input:
            - timestep: [batch_size, num_frame] tensor containing the randomly generated timestep.
            - type: a string indicating the type of the current model (image, bidirectional_video, or causal_video).
        Output Behavior:
            - image: check that the second dimension (num_frame) is 1.
            - bidirectional_video: broadcast the timestep to be the same for all frames.
            - causal_video: broadcast the timestep to be the same for all frames **in a block**.
        """
        if type == "bidirectional_video":
            for index in range(timestep.shape[0]):
                timestep[index] = timestep[index, 0]
            return timestep
        elif type == "causal_video":
            # make the noise level the same within every motion block
            timestep = timestep.reshape(
                timestep.shape[0], -1, self.num_frames_per_block)
            timestep[:, :, 1:] = timestep[:, :, 0:1]
            timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep
        else:
            raise NotImplementedError("Unsupported model type {}".format(type))

    def get_size_less_than_area(self,
                                height,
                                width,
                                target_area=1024 * 704,
                                divisor=64):
        if height * width <= target_area:
            # If the original image area is already less than or equal to the target,
            # no resizing is needed—just padding. Still need to ensure that the padded area doesn't exceed the target.
            max_upper_area = target_area
            min_scale = 0.1
            max_scale = 1.0
        else:
            # Resize to fit within the target area and then pad to multiples of `divisor`
            max_upper_area = target_area  # Maximum allowed total pixel count after padding
            d = divisor - 1
            b = d * (height + width)
            a = height * width
            c = d**2 - max_upper_area

            # Calculate scale boundaries using quadratic equation
            min_scale = (-b + math.sqrt(b**2 - 2 * a * c)) / (
                2 * a)  # Scale when maximum padding is applied
            max_scale = math.sqrt(max_upper_area /
                                  (height * width))  # Scale without any padding

        # We want to choose the largest possible scale such that the final padded area does not exceed max_upper_area
        # Use binary search-like iteration to find this scale
        find_it = False
        for i in range(100):
            scale = max_scale - (max_scale - min_scale) * i / 100
            new_height, new_width = int(height * scale), int(width * scale)

            # Pad to make dimensions divisible by 64
            pad_height = (64 - new_height % 64) % 64
            pad_width = (64 - new_width % 64) % 64
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left

            padded_height, padded_width = new_height + pad_height, new_width + pad_width

            if padded_height * padded_width <= max_upper_area:
                find_it = True
                break

        if find_it:
            return padded_height, padded_width
        else:
            # Fallback: calculate target dimensions based on aspect ratio and divisor alignment
            aspect_ratio = width / height
            target_width = int(
                (target_area * aspect_ratio)**0.5 // divisor * divisor)
            target_height = int(
                (target_area / aspect_ratio)**0.5 // divisor * divisor)

            # Ensure the result is not larger than the original resolution
            if target_width >= width or target_height >= height:
                target_width = int(width // divisor * divisor)
                target_height = int(height // divisor * divisor)

            return target_height, target_width

    def prepare_default_cond_input(self,
                                   map_shape=[3, 12, 64, 64],
                                   motion_frames=5,
                                   lat_motion_frames=2,
                                   enable_mano=False,
                                   enable_kp=False,
                                   enable_pose=False):
        default_value = [1.0, -1.0, -1.0]
        cond_enable = [enable_mano, enable_kp, enable_pose]
        cond = []
        for d, c in zip(default_value, cond_enable):
            if c:
                map_value = torch.ones(
                    map_shape, dtype=self.param_dtype, device=self.device) * d
                cond_lat = torch.cat([
                    map_value[:, :, 0:1].repeat(1, 1, motion_frames, 1, 1),
                    map_value
                ],
                                     dim=2)
                cond_lat = torch.stack(
                    self.vae.encode(cond_lat.to(
                        self.param_dtype)))[:, :, lat_motion_frames:].to(
                            self.param_dtype)

                cond.append(cond_lat)
        if len(cond) >= 1:
            cond = torch.cat(cond, dim=1)
        else:
            cond = None
        return cond

    def encode_audio(self, audio_path, infer_frames):
        assert self.is_training is False
        z = self.audio_encoder.extract_audio_feat(
            audio_path, return_all_layers=True)
        audio_embed_bucket, num_repeat = self.audio_encoder.get_audio_embed_bucket_fps(
            z, fps=self.fps, batch_frames=infer_frames, m=self.audio_sample_m)
        audio_embed_bucket = audio_embed_bucket.to(self.device,
                                                   self.param_dtype)
        audio_embed_bucket = audio_embed_bucket.unsqueeze(0)
        if len(audio_embed_bucket.shape) == 3:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 1)
        elif len(audio_embed_bucket.shape) == 4:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 3, 1)
        return audio_embed_bucket, num_repeat
    
    def encode_audio_training(self, audio_tensor, infer_frames,fps,audio_sample_m = 0):
        assert self.is_training is True
        z = self.audio_encoder.extract_audio_feat_training(
            audio_tensor, return_all_layers=True)
        audio_embed_bucket, num_repeat = self.audio_encoder.get_audio_embed_bucket_fps(
            z, fps=fps, batch_frames=infer_frames, m=audio_sample_m)
        audio_embed_bucket = audio_embed_bucket.to(self.device,
                                                   self.param_dtype)
        audio_embed_bucket = audio_embed_bucket.unsqueeze(0)
        if len(audio_embed_bucket.shape) == 3:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 1)
        elif len(audio_embed_bucket.shape) == 4:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 3, 1)
        return audio_embed_bucket, num_repeat

    def read_last_n_frames(self,
                           video_path,
                           n_frames,
                           target_fps=16,
                           reverse=False):
        """
        Read the last `n_frames` from a video at the specified frame rate.

        Parameters:
            video_path (str): Path to the video file.
            n_frames (int): Number of frames to read.
            target_fps (int, optional): Target sampling frame rate. Defaults to 16.
            reverse (bool, optional): Whether to read frames in reverse order. 
                                    If True, reads the first `n_frames` instead of the last ones.

        Returns:
            np.ndarray: A NumPy array of shape [n_frames, H, W, 3], representing the sampled video frames.
        """
        vr = VideoReader(video_path)
        original_fps = vr.get_avg_fps()
        total_frames = len(vr)

        interval = max(1, round(original_fps / target_fps))

        required_span = (n_frames - 1) * interval

        start_frame = max(0, total_frames - required_span -
                          1) if not reverse else 0

        sampled_indices = []
        for i in range(n_frames):
            indice = start_frame + i * interval
            if indice >= total_frames:
                break
            else:
                sampled_indices.append(indice)

        return vr.get_batch(sampled_indices).asnumpy()

    def encode_prompt(self, input_prompt, n_prompt=None, offload_model=True):
        """
        编码文本提示词
        三种情况：只有正向，cfg但是提供空 neg，提供 neg
        
        Args:
            input_prompt (str): 正面文本提示词
            n_prompt (str): 负面文本提示词，默认为空字符串
            offload_model (bool): 是否在编码后将模型卸载到CPU，默认为True
            
        Returns:
            tuple: (context, context_null) - 编码后的正面和负面文本上下文
        """
        context_null = None
        if n_prompt == "": #case 3
            n_prompt = self.sample_neg_prompt
            
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            if n_prompt is not None:
                context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            if n_prompt is not None:
                context_null = self.text_encoder([n_prompt], torch.device('cpu'))
                context_null = [t.to(self.device) for t in context_null]
                
        return context, context_null

    def load_pose_cond(self, pose_video, num_repeat, infer_frames, size):
        HEIGHT, WIDTH = size
        if not pose_video is None:
            pose_seq = self.read_last_n_frames(
                pose_video,
                n_frames=infer_frames * num_repeat,
                target_fps=self.fps,
                reverse=True)

            resize_opreat = transforms.Resize(min(HEIGHT, WIDTH))
            crop_opreat = transforms.CenterCrop((HEIGHT, WIDTH))
            tensor_trans = transforms.ToTensor()

            cond_tensor = torch.from_numpy(pose_seq)
            cond_tensor = cond_tensor.permute(0, 3, 1, 2) / 255.0 * 2 - 1.0
            cond_tensor = crop_opreat(resize_opreat(cond_tensor)).permute(
                1, 0, 2, 3).unsqueeze(0)

            padding_frame_num = num_repeat * infer_frames - cond_tensor.shape[2]
            cond_tensor = torch.cat([
                cond_tensor,
                - torch.ones([1, 3, padding_frame_num, HEIGHT, WIDTH])
            ],
                                    dim=2)

            cond_tensors = torch.chunk(cond_tensor, num_repeat, dim=2)
        else:
            cond_tensors = [-torch.ones([1, 3, infer_frames, HEIGHT, WIDTH])]

        COND = []
        for r in range(len(cond_tensors)):
            cond = cond_tensors[r]
            cond = torch.cat([cond[:, :, 0:1].repeat(1, 1, 1, 1, 1), cond],
                             dim=2)
            cond_lat = torch.stack(
                self.vae.encode(
                    cond.to(dtype=self.param_dtype,
                            device=self.device)))[:, :,
                                                  1:].cpu()  # for mem save
            COND.append(cond_lat)
        return COND

    def get_gen_size(self, size, max_area, ref_image_path, pre_video_path):
        if not size is None:
            HEIGHT, WIDTH = size
        else:
            if pre_video_path:
                ref_image = self.read_last_n_frames(
                    pre_video_path, n_frames=1)[0]
            else:
                ref_image = np.array(Image.open(ref_image_path).convert('RGB'))
            HEIGHT, WIDTH = ref_image.shape[:2]
        HEIGHT, WIDTH = self.get_size_less_than_area(
            HEIGHT, WIDTH, target_area=max_area)
        return (HEIGHT, WIDTH)

    def _initialize_kv_cache(self, batch_size, dtype, device, kv_cache_size=13500):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        gpu_id : "1","2","3","4"
        """
        kv_cache1 = []

        for _ in range(self.noise_model.num_layers):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 40, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 40, 128], dtype=dtype, device=device),
                "cond_k": torch.zeros([batch_size, 3000, 40, 128], dtype=dtype, device=device), #hard-coded for 720*400 resolution
                "cond_v": torch.zeros([batch_size, 3000, 40, 128], dtype=dtype, device=device), #hard-coded for 720*400 resolution
                "cond_end": torch.tensor([0], dtype=torch.long, device=device) #dynamically updated 2640
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache
    
    def _move_kv_cache_to_working_gpu(self,moved_id, gpu_id=0):
        """
        Move the KV cache to the working GPU.
        move_id : "1","2","3","4"
        """
        if gpu_id == 0: #move to working device
            tgt_device =f"cuda:{gpu_id}"
        else: #offload
            tgt_device = "cpu" if self.single_gpu else f"cuda:{gpu_id}"
            
        kv_cache1 = self.kv_cache1[str(moved_id)]
        for layer in kv_cache1:
            layer["k"] = layer["k"].to(tgt_device)
            layer["v"] = layer["v"].to(tgt_device)
            layer["cond_k"] = layer["cond_k"].to(tgt_device)
            layer["cond_v"] = layer["cond_v"].to(tgt_device)
        self.kv_cache1[str(moved_id)] = kv_cache1

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.noise_model.num_layers):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 0, 40, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 0, 40, 128], dtype=dtype, device=device),
                "is_init": False
            })

        self.crossattn_cache = crossattn_cache  # always store the clean cache
    
    def _initialize_comm_group(self, num_gpus_dit=4, enable_vae_parallel=False):
        local_gpu_id = torch.distributed.get_rank()
        tgt_gpu_id = local_gpu_id + 1
        src_gpu_id = local_gpu_id - 1
        if local_gpu_id == num_gpus_dit - 1 + int(enable_vae_parallel):
            self.tgt_gpu = None
        else:
            self.tgt_gpu = tgt_gpu_id

        if local_gpu_id == 0:
            self.src_gpu = None
        else:
            self.src_gpu = src_gpu_id

    def generate(
        self,
        input_prompt=None,
        ref_image_path=None,
        audio_path=None,
        enable_tts=False,
        tts_prompt_audio=None,
        tts_prompt_text=None,
        tts_text=None,
        num_repeat=1,
        pose_video=None,
        generate_size=None,
        max_area=720 * 1280,
        infer_frames=80,
        shift=5.0,
        sample_solver='unipc',
        sampling_steps=40,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        offload_model=True,
        init_first_frame=False,
        use_dataset=False,
        dataset_sample_idx=0,
        drop_motion_noisy=False,
        num_gpus_dit=4,
        max_repeat=1000000,
        enable_vae_parallel=False,
        mask=None,
        input_video_for_sam2=None,
        enable_online_decode=False,
    ):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            ref_image_path ('str'):
                Input image path
            audio_path ('str'):
                Audio for video driven
            num_repeat ('int'):
                Number of clips to generate; will be automatically adjusted based on the audio length
            pose_video ('str'):
                If provided, uses a sequence of poses to drive the generated video
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            infer_frames (`int`, *optional*, defaults to 80):
                How many frames to generate per clips. The number should be 4n
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float` or tuple[`float`], *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
                If tuple, the first guide_scale will be used for low noise model and
                the second guide_scale will be used for high noise model.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            init_first_frame (`bool`, *optional*, defaults to False):
                Whether to use the reference image as the first frame (i.e., standard image-to-video generation)

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        # ------------------------------------Step 1: prepare conditional inputs--------------------------------------
        
        size = self.get_gen_size(
            size=None,
            max_area=max_area,
            ref_image_path=ref_image_path,
            pre_video_path=None)
        HEIGHT, WIDTH = size
        # HEIGHT, WIDTH = map(int, generate_size.split('*'))
        # size = (HEIGHT, WIDTH)
        channel = 3

        resize_opreat = transforms.Resize(min(HEIGHT, WIDTH))
        crop_opreat = transforms.CenterCrop((HEIGHT, WIDTH))
        tensor_trans = transforms.ToTensor()


        ref_image = np.array(Image.open(ref_image_path).convert('RGB'))

        # extract audio emb
        if enable_tts is True:
            audio_path = self.tts(tts_prompt_audio, tts_prompt_text, tts_text)
        # audio_emb, nr = self.encode_audio(audio_path, infer_frames=infer_frames)
        self.audio_encoder.model.to(device=self.device, dtype=self.param_dtype)
        self.audio_encoder.model.requires_grad_(False)
        self.audio_encoder.model.eval()
        
        if '+' in audio_path:
            audio_paths = audio_path.split('+')
            audio_embs = []
            nr_list = []
            
            for path in audio_paths:
                audio_emb_i, nr_i = self.encode_audio(path, infer_frames=infer_frames)
                audio_embs.append(audio_emb_i)
                nr_list.append(nr_i)
            
            min_frames = min(emb.shape[-1] for emb in audio_embs)
            audio_embs = [emb[..., :min_frames] for emb in audio_embs]
            nr = min(nr_list)
            audio_emb = torch.cat(audio_embs, dim=0)

            # Process SAM2 and generate routing_logits if video path is provided
            print(f"rank {dist.get_rank()} processing SAM2")
            input_video_for_sam2 = input_video_for_sam2 if input_video_for_sam2 is not None else ref_image_path
            routing_logits = None
            rank = dist.get_rank()
            
            # Broadcast video path to all ranks
            if rank == 0:
                video_path_bytes = input_video_for_sam2.encode('utf-8')
                path_length = torch.tensor([len(video_path_bytes)], dtype=torch.long, device=self.device)
            else:
                path_length = torch.tensor([0], dtype=torch.long, device=self.device)
            dist.broadcast(path_length, src=0)
            if rank == 0:
                path_tensor = torch.ByteTensor(list(video_path_bytes)).to(self.device)
            else:
                path_tensor = torch.zeros(path_length.item(), dtype=torch.uint8, device=self.device)
            dist.broadcast(path_tensor, src=0)
            video_path = path_tensor.cpu().numpy().tobytes().decode('utf-8')
            print(f"Rank {rank}: video_path: {video_path}")
            
            parent_dir = os.path.dirname(video_path)
            sam2_output_base = parent_dir  
            
            if rank == 0:
                sam2_cmd = [
                    "python",
                    "liveavatar/utils/router/sam2_tools.py",
                    "--video_folder", video_path,
                    "--output_path", sam2_output_base
                ]
                try:
                    subprocess.run(sam2_cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Rank {rank}: SAM2 processing failed: {e}")
                    raise e
                dist.barrier()
            else:
                dist.barrier()
            
            base_name = os.path.basename(video_path).split(".")[0]
            tracking_mask_results_dir = os.path.join(
                sam2_output_base,
                base_name,
                "tracking_mask_results"
            )
            print(f"Rank {rank}: Looking for masks in: {tracking_mask_results_dir}")
            
            target_shape = (1, infer_frames // 4, HEIGHT // 8, WIDTH // 8)
            routing_logits = process_masks_to_routing_logits(
                tracking_mask_results_dir,
                shape=target_shape
            )
            num_actors = routing_logits.shape[-1]
            routing_logits = routing_logits.reshape(1, infer_frames // 4, HEIGHT // 8 // 2, WIDTH // 8 // 2, num_actors)
            routing_logits = routing_logits.to(device=self.device, dtype=self.param_dtype)
            mask = routing_logits.permute(4,1,2,3,0)  # [num_actors, t, h, w, 1]

            # 按比例进行二维空间膨胀（允许重叠）
            def dilate_mask_by_ratio(mask_tensor: torch.Tensor, ratio: float = 0.3, thr: float = 0.5) -> torch.Tensor:
                # mask_tensor: [A, T, H, W, 1]，值域 [0,1]
                A, T, H, W, _ = mask_tensor.shape
                out = torch.zeros_like(mask_tensor)
                bin_mask = (mask_tensor > thr).to(dtype=mask_tensor.dtype)
                for a in range(A):
                    for t in range(T):
                        m2d = bin_mask[a, t, :, :, 0]
                        if m2d.any():
                            ys, xs = torch.where(m2d)
                            box_h = int(ys.max() - ys.min() + 1)
                            box_w = int(xs.max() - xs.min() + 1)
                            radius = max(1, int((ratio * max(box_h, box_w) + 0.9999)))
                            k = 2 * radius + 1
                            x = m2d[None, None, :, :]
                            x = F.max_pool2d(x, kernel_size=k, stride=1, padding=radius)
                            out[a, t, :, :, 0] = (x[0, 0] > 0).to(mask_tensor.dtype)
                        else:
                            out[a, t, :, :, 0] = m2d
                return out

            mask = dilate_mask_by_ratio(mask, ratio=0.1, thr=0.5)
            mask_bool = mask > 0.5
            total_count = mask_bool.sum(dim=0, keepdim=True)  # [1, t, h, w, 1], 统计该位置有多少角色为1
            others_present = (total_count - mask_bool.to(total_count.dtype)) > 0  # [num_actors, t, h, w, 1]
            mask = (~others_present).to(dtype=mask.dtype)
            m=(mask[0][0].detach().to(torch.float16).cpu().numpy()>0.5).astype(np.uint8)*255; Image.fromarray(m.squeeze()).save("tmp/mask/mask.png")
        else:
            audio_emb, nr = self.encode_audio(audio_path, infer_frames=infer_frames)

        
        self.audio_encoder.model.to("cpu")
        if num_repeat is None or num_repeat > nr:
            num_repeat = nr

        lat_motion_frames = (self.motion_frames + 3) // 4
        model_pic = crop_opreat(resize_opreat(Image.fromarray(ref_image)))

        ref_pixel_values = tensor_trans(model_pic)
        ref_pixel_values = ref_pixel_values.unsqueeze(1).unsqueeze(
            0) * 2 - 1.0  # b c 1 h w
        ref_pixel_values = ref_pixel_values.to(
            dtype=self.vae.dtype, device=self.vae.device)
        ref_latents = torch.stack(self.vae.encode(ref_pixel_values))

        # drop_first_motion = self.drop_first_motion
        drop_first_motion = False
        motion_latents = ref_pixel_values.repeat(1, 1, self.motion_frames, 1, 1)
        videos_last_frames = motion_latents.detach()
        motion_latents = torch.stack(self.vae.encode(motion_latents))
        
        if drop_motion_noisy:
            zero_motion_latents = torch.zeros_like(motion_latents)

        # get pose cond input if need
        COND = self.load_pose_cond(
            pose_video=pose_video,
            num_repeat=num_repeat,
            infer_frames=infer_frames,
            size=size) # list(1):[1,16,12,48,32]当num_repeat=1

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # process prompt
        context, context_null = self.encode_prompt(input_prompt, n_prompt, offload_model) #list(1):[len,4096]
        dataset_info = {}

        print("complete prepare conditional inputs")
        if sample_solver == 'euler':#default
            sample_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=3)
        else:
            raise NotImplementedError("Unsupported solver.")
        self._initialize_comm_group(num_gpus_dit=num_gpus_dit, enable_vae_parallel=enable_vae_parallel)
        in_dit_device = dist.get_rank() < num_gpus_dit
        dist.barrier() # wait all ranks to finish initialization


        #--------------------------------------Step 2: generate--------------------------------------
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
        ):
            out = []
            self.kv_cache1 = None
            active_nr = min(max_repeat, num_repeat)
            for r in range(active_nr):
            #-------------------------------------------rollout loop------------------------------------------------------
                #----------------------------------------------Step 2.1: clip-level init------------------------------------------------------ 

                if r==0 or in_dit_device:
                    seed_g = torch.Generator(device=self.device)
                    seed_g.manual_seed(seed + r)

                    lat_target_frames = (infer_frames + 3 + self.motion_frames
                                        ) // 4 - lat_motion_frames
                    target_shape = [lat_target_frames, HEIGHT // 8, WIDTH // 8]
                    frame_seq_length = HEIGHT // 8 * WIDTH // 8 // 2 // 2
                    clip_noise = [
                        torch.randn(
                            16,
                            target_shape[0],
                            target_shape[1],
                            target_shape[2],
                            dtype=self.param_dtype,
                            device=self.device,
                            generator=seed_g)
                    ]
                    clip_output = torch.zeros_like(clip_noise[0]) #[16,f,h,w]
                    max_seq_len = np.prod(target_shape) // 4
                    if self.kv_cache1 is None:
                        local_rank = torch.distributed.get_rank()
                        if local_rank < num_gpus_dit:
                            self._initialize_kv_cache(
                                    batch_size=1,
                                    dtype=self.param_dtype,
                                    device=f"cuda:{local_rank}",
                                    kv_cache_size=max_seq_len
                                )

                        self._initialize_crossattn_cache(
                            batch_size=1,
                            dtype=self.param_dtype,
                            device=self.device
                        )


                #----------------------------------------------Step 2.2: prepare clip-level cond---------------------------------
                if r==0 or in_dit_device:
                    clip_latents = deepcopy(clip_noise)
                    with torch.no_grad():
                        left_idx = r * infer_frames
                        right_idx = r * infer_frames + infer_frames
                        cond_latents = COND[r] if pose_video else COND[0] * 0
                        cond_latents = cond_latents.to(
                            dtype=self.param_dtype, device=self.device)
                        audio_input = audio_emb[..., left_idx:right_idx]
                    input_motion_latents = motion_latents.clone()

                    if offload_model or self.init_on_cpu:
                        self.noise_model.to(self.device)
                        torch.cuda.empty_cache()

                #-----------------------------------------------Temporal denoising loop in single clip---------------------------------
                # 2.2.0 prefill cond caching
                if (r==0 or r==1) and (dist.get_rank() != num_gpus_dit-1+int(enable_vae_parallel)): #考虑要不要r==1的时候替换一下ref cond，如果要的话clip r=0的时候还不能并行，要让每卡都有clean的latent0
                    if r==1:
                        ref_latents = torch.empty_like(ref_latents).type_as(clip_latents[0])
                        dist.broadcast(ref_latents, src=num_gpus_dit-1+int(enable_vae_parallel))
                    block_index = 0
                    block_latents = clip_latents[0][:, block_index *
                                    self.num_frames_per_block:(block_index + 1) * self.num_frames_per_block] #[16,f,h,w]
                    left_idx = block_index * (self.num_frames_per_block * 4)
                    right_idx = (block_index+1) * (self.num_frames_per_block * 4)
                    block_arg_c = {
                        'context': context[0:1], #list(1) torch.Size([19, 4096])
                        'seq_len': None,
                        'cond_states': cond_latents[:,:,block_index * 
                                        self.num_frames_per_block:(block_index + 1) * self.num_frames_per_block],
                        "motion_latents": input_motion_latents,
                        'ref_latents': ref_latents,
                        "audio_input": audio_input[..., left_idx:right_idx],
                        "motion_frames": [self.motion_frames, lat_motion_frames],
                        "drop_motion_frames": drop_first_motion and r == 0,
                        "sink_flag": True,
                    }
                    timestep = torch.ones(
                        [1, self.num_frames_per_block], device=self.device, dtype=self.param_dtype) * 0
                    self.noise_model( #update clean kv cache
                        [block_latents], t=timestep*0, **block_arg_c, 
                        kv_cache=self.kv_cache1, crossattn_cache=self.crossattn_cache,
                        current_start=block_index * self.num_frames_per_block * frame_seq_length,
                        current_end=(block_index + 1) * self.num_frames_per_block * frame_seq_length)
                        


                num_blocks = target_shape[0] // self.num_frames_per_block
                for block_index in range(num_blocks):
                    # 2.2.1 prepare block-level cond
                    if getattr(self, '_sampler_timesteps', None) is None:
                        sample_scheduler.set_timesteps(
                            sampling_steps, device=self.device)
                        self._sampler_timesteps = sample_scheduler.timesteps
                        self._sampler_sigmas = sample_scheduler.sigmas

                    timesteps = self._sampler_timesteps
                    sample_scheduler.timesteps = timesteps
                    sample_scheduler.sigmas = self._sampler_sigmas
                    sample_scheduler._step_index = dist.get_rank() 
                    sample_scheduler._begin_index = 0

                    block_latents = clip_latents[0][:, block_index *
                                self.num_frames_per_block:(block_index + 1) * self.num_frames_per_block] #[16,f,h,w]
                    if r==0 or in_dit_device:
                        left_idx = block_index * (self.num_frames_per_block * 4)
                        right_idx = (block_index+1) * (self.num_frames_per_block * 4)
                        block_arg_c = {
                            'context': context[0:1], #list(1) torch.Size([19, 4096])
                            'seq_len': None,
                            'cond_states': cond_latents[:,:,block_index * 
                                            self.num_frames_per_block:(block_index + 1) * self.num_frames_per_block],
                            "motion_latents": input_motion_latents,
                            'ref_latents': ref_latents,
                            "audio_input": audio_input[..., left_idx:right_idx],
                            "motion_frames": [self.motion_frames, lat_motion_frames],
                            "drop_motion_frames": drop_first_motion and r == 0,
                        }

                    for i, t in enumerate(tqdm(timesteps)):
                        # Modulo cycling: step i runs on rank (i % num_gpus_dit)
                        my_rank = dist.get_rank()
                        if my_rank >= num_gpus_dit:
                            # VAE rank skips all denoising steps
                            continue
                        if i % num_gpus_dit != my_rank:
                            continue

                        # Dynamic source: first step of each cycle starts fresh or receives from last DiT rank
                        is_first_in_cycle = (i % num_gpus_dit == 0)
                        is_very_first_step = (i == 0)

                        if is_very_first_step:
                            # Very first step: use initial block_latents
                            latent_model_input = block_latents
                        elif is_first_in_cycle:
                            # Start of new cycle: receive from last DiT rank (rank num_gpus_dit-1)
                            latent_model_input = torch.empty_like(block_latents)
                            dist.recv(latent_model_input, num_gpus_dit - 1)
                        elif self.src_gpu is not None:
                            # Normal step in cycle: receive from previous rank
                            latent_model_input = torch.empty_like(block_latents)
                            dist.recv(latent_model_input, self.src_gpu)
                        else:
                            latent_model_input = block_latents

                        timestep = [t] * self.num_frames_per_block
                        timestep = torch.tensor(timestep).to(self.device).unsqueeze(0)
                        

                        noise_pred_cond = self.noise_model(
                            [latent_model_input], t=timestep, **block_arg_c, 
                            kv_cache=self.kv_cache1, crossattn_cache=self.crossattn_cache,
                            current_start=block_index * self.num_frames_per_block * frame_seq_length + r * num_blocks * self.num_frames_per_block * frame_seq_length,
                            current_end=(block_index + 1) * self.num_frames_per_block * frame_seq_length + r * num_blocks *self.num_frames_per_block * frame_seq_length,
                            mask=mask)

                        noise_pred = [torch.cat(noise_pred_cond, dim=0)]

                        # Update scheduler step index for modulo cycling
                        sample_scheduler._step_index = i
                        temp_x0 = sample_scheduler.step(
                            noise_pred[0].unsqueeze(0),# [16,f,h,w]
                            t,
                            latent_model_input.unsqueeze(0), #[1,16,f,h,w]
                            return_dict=False,
                            generator=seed_g)[0]
                        block_latents = temp_x0.squeeze(0) #[16,num_frames_per_block,h,w]

                        # Dynamic target: last rank in cycle sends to rank 0, unless it's the final step
                        is_last_in_cycle = (my_rank == num_gpus_dit - 1)
                        is_final_step = (i == len(timesteps) - 1)

                        if is_final_step:
                            # Final step: send to VAE rank
                            if self.tgt_gpu is not None:
                                dist.send(block_latents.contiguous(), self.tgt_gpu)
                        elif is_last_in_cycle:
                            # End of cycle but not final: send to rank 0 for next cycle
                            dist.send(block_latents.contiguous(), 0)
                        elif self.tgt_gpu is not None and self.tgt_gpu < num_gpus_dit:
                            # Normal step: send to next rank (but not to VAE)
                            dist.send(block_latents.contiguous(), self.tgt_gpu)
 
                    if enable_vae_parallel and dist.get_rank() == num_gpus_dit-1+int(enable_vae_parallel):
                            vae_wait_start = time.time()
                            block_latents = torch.empty_like(block_latents)
                            dist.recv(block_latents, self.src_gpu)
                            torch.cuda.synchronize()
                            if time.time() - vae_wait_start < 0.01:
                                print(f"WARNING: VAE serves as a bottleneck!")

                    #----------------------------------------------Step 2.3: block-level postprocess for vae---------------------------------
                    if enable_vae_parallel and dist.get_rank() == num_gpus_dit-1+int(enable_vae_parallel):
                        if offload_model:
                            print(f"offloading model to cpu")
                            self.noise_model.cpu()
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                        if r == 0 and active_nr != 1:
                            if block_index == 0: #cache new ref
                                ref_latents = block_latents.unsqueeze(0)[:,:,0:1] # 更新attention sink anchor到generated image，broadcast到所有rank
                            elif block_index == num_blocks-1: #broadcast ref to all ranks
                                dist.broadcast(ref_latents.contiguous(), src=num_gpus_dit-1+int(enable_vae_parallel))
                            else:
                                pass

                        # decode to rgb
                        if not (drop_first_motion and r == 0):
                            decode_latents = torch.cat([motion_latents, block_latents.unsqueeze(0)], dim=2)
                        else:
                            decode_latents = torch.cat([ref_latents, block_latents.unsqueeze(0)], dim=2)
                        image = torch.stack(self.vae.decode(decode_latents))
                        image = image[:, :, -(infer_frames)//num_blocks:] # 3
                        
                        if r == 0 and block_index == 0:
                            image = image[:, :, 3:]#第一个clip第一个block保留0帧，后面3
                        
                        overlap_frames_num = min(self.motion_frames, image.shape[2]) # 12 
                        #todo:现在一次生成帧数必须大于motion_frames，所以需要长帧数训练，或者训练时候随机drop掉motion_frame的前73-48帧
                        videos_last_frames = torch.cat([
                            videos_last_frames[:, :, overlap_frames_num:], #61->16
                            image[:, :, -overlap_frames_num:] #12->3
                        ],
                                                    dim=2)
                        videos_last_frames = videos_last_frames[:,:,-9:]
                        # videos_last_frames = image[:, :, -overlap_frames_num:]
                        videos_last_frames = videos_last_frames.to(
                            dtype=motion_latents.dtype, device=motion_latents.device)
                        # motion_latents = block_latents.unsqueeze(0)[:,:,-4:]
                        motion_latents = torch.stack(
                            self.vae.encode(videos_last_frames)).type_as(clip_latents[0])
                        # if (drop_first_motion and r == 0):
                        #     motion_latents[:,:,:-overlap_frames_num//4] = 0
                            
                        out.append(image.cpu())

        #-------------------------------------- Step 3: full-video postprocess--------------------------------------
        if dist.is_initialized():
                dist.barrier()
        if dist.get_rank() == num_gpus_dit-1+int(enable_vae_parallel):
            videos = torch.cat(out, dim=2)
            del clip_noise, clip_latents, clip_output,block_latents
            self._sampler_timesteps = None
            self._sampler_sigmas = None
            # del sample_scheduler
            if offload_model:
                gc.collect()
                torch.cuda.synchronize()
            

            return videos[0], dataset_info
        else:
            return None,dataset_info
    

    def tts(self, tts_prompt_audio, tts_prompt_text, tts_text):
        if not hasattr(self, 'cosyvoice'):
            self.load_tts()
        speech_list = []
        from cosyvoice.utils.file_utils import load_wav
        import torchaudio
        prompt_speech_16k = load_wav(tts_prompt_audio, 16000)
        if tts_prompt_text is not None:
            for i in self.cosyvoice.inference_zero_shot(tts_text, tts_prompt_text, prompt_speech_16k):
                speech_list.append(i['tts_speech'])
        else:
            for i in self.cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k):
                speech_list.append(i['tts_speech'])
        torchaudio.save('tts.wav', torch.concat(speech_list, dim=1), self.cosyvoice.sample_rate)
        return 'tts.wav'

    def load_tts(self):
        if not os.path.exists('CosyVoice'):
            from wan.utils.utils import download_cosyvoice_repo
            download_cosyvoice_repo('CosyVoice')
        if not os.path.exists('CosyVoice2-0.5B'):
            from wan.utils.utils import download_cosyvoice_model
            download_cosyvoice_model('CosyVoice2-0.5B', 'CosyVoice2-0.5B')
        sys.path.append('CosyVoice')
        sys.path.append('CosyVoice/third_party/Matcha-TTS')
        from cosyvoice.cli.cosyvoice import CosyVoice2
        self.cosyvoice = CosyVoice2('CosyVoice2-0.5B')