# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import warnings
import gc
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Brand: Lynk & Co Jordan
# Edit these to re-skin the UI. Colors follow the Lynk & Co identity:
# black background, white text, vivid green accent.
# ---------------------------------------------------------------------------
BRAND_NAME = "LYNK & CO"
BRAND_SUB = "JORDAN"
BRAND_PRODUCT = "Avatar Studio"
BRAND_TAGLINE = "Audio-driven avatar generation — a connected guest experience."
BRAND_GREEN = "#00E676"
BRAND_BLACK = "#0A0A0A"
BRAND_FOOTER = "Lynk & Co Jordan · Avatar Studio"

import random
import gradio as gr
import torch
import torch.distributed as dist
from PIL import Image

from liveavatar.models.wan.wan_2_2.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from liveavatar.models.wan.wan_2_2.distributed.util import init_distributed_group
from liveavatar.models.wan.wan_2_2.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from liveavatar.models.wan.wan_2_2.utils.utils import merge_video_audio, save_video, str2bool
from liveavatar.utils.args_config import parse_args_for_training_config as training_config_parser

# Global variables for pipeline and config
wan_s2v_pipeline = None
global_args = None
global_cfg = None
global_training_settings = None
global_rank = 0
global_world_size = 1
global_save_rank = 0

EXAMPLE_PROMPT = {
    "s2v-14B": {
        "prompt":
            "A scene in which the anchorwoman is interacting with the audience, with a clean interior in the background.",
        "image":
            "examples/anchor.jpg",
        "audio":
            "examples/fashion_blogger.wav",
    },
}

# Example images for gallery
EXAMPLE_IMAGES = [
    ("examples/anchor.jpg", "Anchor"),
    ("examples/fashion_blogger.jpg", "Fashion Blogger"),
    ("examples/kitchen_grandmother.jpg", "Kitchen Grandmother"),
    ("examples/music_producer.jpg", "Music Producer"),
    ("examples/dwarven_blacksmith.jpg", "Dwarven Blacksmith"),
    ("examples/cyclops_baker.jpg", "Cyclops Baker"),
    ("examples/livestream_1.png", "Livestream 1"),
    ("examples/livestream_2.jpg", "Livestream 2"),
]

# Example audios
EXAMPLE_AUDIOS = [
    ("examples/fashion_blogger.wav", "Fashion Blogger Audio"),
    ("examples/kitchen_grandmother.wav", "Kitchen Grandmother Audio"),
    ("examples/music_producer.wav", "Music Producer Audio"),
    ("examples/dwarven_blacksmith.wav", "Dwarven Blacksmith Audio"),
    ("examples/cyclops_baker.wav", "Cyclops Baker Audio"),
]


def _validate_args(args):
    """Validate command line arguments"""
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"

    if args.size is None:
        if args.single_gpu or args.num_gpus_dit == 1:
            args.size = "704*384"
        else:
            args.size = "720*400"

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)


def _parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Gradio Web UI for LiveAvatar generation"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="s2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default=None,
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video.")
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames of video are generated. The number should be 4n+1")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="ckpt/Wan2.2-S2V-14B/",
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage.")
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./output/gradio/",
        help="The directory to save the generated video to.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='euler',
        choices=['euler', 'unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", 
        type=int, 
        default=4, 
        help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=0.0,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=420,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--infer_frames",
        type=int,
        default=48,
        help="Number of frames per clip, 48 or 80 or others (must be multiple of 4) for 14B s2v")
    parser.add_argument(
        "--load_lora",
        action="store_true",
        default=False,
        help="Whether to load the LoRA weights.")
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="The path to the LoRA weights.")
    parser.add_argument(
        "--lora_path_dmd",
        type=str,
        default=None,
        help="The path to the LoRA weights for DMD.")
    parser.add_argument(
        "--training_config",
        type=str,
        default="liveavatar/configs/s2v_causal_sft.yaml",
        help="The path to the training config file.")
    parser.add_argument(
        "--num_clip",
        type=int,
        default=10,
        help="Number of video clips to generate.")
    parser.add_argument(
        "--single_gpu",
        action="store_true",
        default=False,
        help="Whether to use a single GPU.")
    parser.add_argument(
        "--using_merged_ckpt",
        action="store_true",
        default=False,
        help="Whether to use the merged ckpt.")
    parser.add_argument(
        "--num_gpus_dit",
        type=int,
        default=4,
        help="The number of GPUs to use for DiT.")
    parser.add_argument(
        "--enable_vae_parallel",
        action="store_true",
        default=False,
        help="Whether to enable VAE parallel decoding on a separate GPU.")
    parser.add_argument(
        "--offload_kv_cache",
        action="store_true",
        default=False,
        help="Whether to offload the KV cache to CPU.")
    parser.add_argument(
        "--enable_tts",
        action="store_true",
        default=False,
        help="Use CosyVoice to synthesis audio")
    parser.add_argument(
        "--pose_video",
        type=str,
        default=None,
        help="Provide Dw-pose sequence to do Pose Driven")
    parser.add_argument(
        "--start_from_ref",
        action="store_true",
        default=False,
        help="whether set the reference image as the starting point for generation")
    parser.add_argument(
        "--drop_motion_noisy",
        action="store_true",
        default=False,
        help="Whether to drop the motion noisy.")
    parser.add_argument(
        "--server_port",
        type=int,
        default=7860,
        help="Port to run the Gradio server on.")
    parser.add_argument(
        "--server_name",
        type=str,
        default="0.0.0.0",
        help="Server name for Gradio (0.0.0.0 for public access).")
    parser.add_argument(
        "--fp8",
        action="store_true",
        default=False,
        help="Whether to enable fp8 quantization.")
    
    args = parser.parse_args()
    _validate_args(args)
    return args


def _init_logging(rank):
    """Initialize logging configuration"""
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def initialize_pipeline(args, training_settings):
    """
    Initialize the LiveAvatar pipeline (single or multi-GPU)
    This is called once at startup
    """
    global wan_s2v_pipeline, global_args, global_cfg, global_training_settings
    global global_rank, global_world_size, global_save_rank
    
    # Cleanup existing pipeline if it exists
    if wan_s2v_pipeline is not None:
        logging.info("Cleaning up existing pipeline to free GPU memory...")
        # Clear the pipeline reference
        wan_s2v_pipeline = None
        # Force garbage collection
        gc.collect()
        # Empty CUDA cache
        torch.cuda.empty_cache()
    
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    
    if world_size == 1:
        rank = 0
    
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    
    global_rank = rank
    global_world_size = world_size
    
    _init_logging(rank)
    
    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(f"offload_model is not specified, set to {args.offload_model}.")
    
    torch.cuda.set_device(local_rank)
    
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    
    if world_size > 1:
        # assert world_size >= 5, "At least 5 GPUs are supported for distributed inference."
        # assert args.num_gpus_dit == 4, "Only 4 GPUs are supported for distributed inference."
        assert args.enable_vae_parallel is True, "VAE parallel is required for distributed inference."
        args.single_gpu = False
        from liveavatar.models.wan.causal_s2v_pipeline_tpp import WanS2V
        logging.info("Using TPP distributed inference.")
    else:
        assert not (args.t5_fsdp or args.dit_fsdp), \
            "t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (args.ulysses_size > 1), \
            "sequence parallel are not supported in non-distributed environments."
        args.enable_vae_parallel = False
        args.num_gpus_dit = 1
        args.single_gpu = True
        from liveavatar.models.wan.causal_s2v_pipeline import WanS2V
        logging.info(f"Using single GPU inference with offload mode: {args.offload_model}")
    
    if args.ulysses_size > 1:
        assert False, "Sequence parallel is not supported."
        init_distributed_group()
    
    cfg = WAN_CONFIGS[args.task]
    
    logging.info(f"Pipeline initialization args: {args}")
    logging.info(f"Model config: {cfg}")
    
    # Broadcast base seed in distributed mode
    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]
    
    # Create WanS2V pipeline
    if "s2v" in args.task:
        logging.info("Creating WanS2V pipeline...")
        wan_s2v = WanS2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            sp_size=args.ulysses_size,
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
            single_gpu=args.single_gpu,
            offload_kv_cache=args.offload_kv_cache,
        )
        
        # Load LoRA if specified
        if args.load_lora and args.lora_path_dmd is not None:
            logging.info(f'Loading LoRA: path={args.lora_path_dmd}, rank={training_settings["lora_rank"]}, alpha={training_settings["lora_alpha"]}')
            

                
            if args.lora_path_dmd is not None:
                wan_s2v.noise_model = wan_s2v.add_lora_to_model(
                    wan_s2v.noise_model,
                    lora_rank=training_settings['lora_rank'],
                    lora_alpha=training_settings['lora_alpha'],
                    lora_target_modules=training_settings['lora_target_modules'],
                    init_lora_weights=training_settings['init_lora_weights'],
                    pretrained_lora_path=args.lora_path_dmd,
                    load_lora_weight_only=False,
                )
        
        if args.fp8:
            if hasattr(torch, "_scaled_mm"):
                from liveavatar.utils.fp8_linear import replace_linear_with_scaled_fp8
                replace_linear_with_scaled_fp8(
                    wan_s2v.noise_model,
                    ignore_keys=[
                        'text_embedding', 'time_embedding',
                        'time_projection', 'head.head',
                        'casual_audio_encoder.encoder.final_linear',
                    ]
                )

            else:
                logging.info(f"skip fp8_linear, Please update torch vision ")

        
        wan_s2v_pipeline = wan_s2v
        global_args = args
        global_cfg = cfg
        global_training_settings = training_settings
        
        # Determine which rank saves the video
        if args.enable_vae_parallel:
            global_save_rank = args.num_gpus_dit
        else:
            global_save_rank = 0 if world_size == 1 else args.num_gpus_dit - 1
        
        logging.info("Pipeline initialized successfully!")
    else:
        raise ValueError("Only s2v tasks are supported for Gradio interface.")


def run_single_sample(prompt, image_path, audio_path, num_clip, 
                     sample_steps, sample_guide_scale, infer_frames,
                     size, base_seed, sample_solver):
    """
    Run inference for a single sample (called from Gradio on rank 0).
    Broadcasts inputs to other ranks, then all ranks compute together.
    After computation, sends an empty broadcast to let workers continue waiting.
    """
    global global_rank
    
    # In multi-GPU mode, rank 0 broadcasts inputs to other ranks waiting in worker_loop
    if dist.is_initialized() and global_rank == 0:
        # Format: [command, prompt, image_path, audio_path, num_clip, 
        #          sample_steps, sample_guide_scale, infer_frames,
        #          size, base_seed, sample_solver, lora_path, load_lora]
        inputs = ["infer", prompt, image_path, audio_path, num_clip, 
                 sample_steps, sample_guide_scale, infer_frames,
                 size, base_seed, sample_solver, None, None]
        dist.broadcast_object_list(inputs, src=0)
        logging.info(f"[Rank 0] Broadcast inference inputs to other ranks")
    
    # Now rank 0 also participates in the actual computation
    video_path = _run_inference_computation(
        prompt, image_path, audio_path, num_clip,
        sample_steps, sample_guide_scale, infer_frames,
        size, base_seed, sample_solver
    )
    
    # CRITICAL: After computation, other ranks have returned to worker_loop
    # and are waiting at the next broadcast_object_list().
    # Rank 0 must send an "idle" signal to unblock them before returning to Gradio.
    if dist.is_initialized() and global_rank == 0:
        idle_signal = ["idle"] + [None] * 12
        dist.broadcast_object_list(idle_signal, src=0)
        logging.info(f"[Rank 0] Sent idle signal, workers can continue waiting for next request")
    
    return video_path


def reload_pipeline_command(use_lora, lora_path):
    """
    Command to reload the pipeline with new LoRA settings.
    Broadcasts to workers if in distributed mode.
    """
    global global_args, global_training_settings, global_rank
    
    logging.info(f"Reloading pipeline with use_lora={use_lora}, lora_path={lora_path}")
    
    # Update global args
    global_args.load_lora = use_lora
    global_args.lora_path_dmd = lora_path if lora_path else None
    
    if dist.is_initialized() and global_rank == 0:
        # Broadcast reload signal
        # command, prompt, image_path, audio_path, num_clip, sample_steps, sample_guide_scale, 
        # infer_frames, size, base_seed, sample_solver, lora_path, load_lora
        reload_signal = ["reload", None, None, None, None, None, None, None, None, None, None, global_args.lora_path_dmd, global_args.load_lora]
        dist.broadcast_object_list(reload_signal, src=0)
        logging.info(f"[Rank 0] Broadcast reload signal to other ranks")

    # Re-initialize on rank 0
    initialize_pipeline(global_args, global_training_settings)
    
    if dist.is_initialized() and global_rank == 0:
        # Send idle signal to workers after reloading
        idle_signal = ["idle"] + [None] * 12
        dist.broadcast_object_list(idle_signal, src=0)
        logging.info(f"[Rank 0] Sent idle signal after reload")
        
    return "Pipeline reloaded successfully!"


def _run_inference_computation(prompt, image_path, audio_path, num_clip,
                               sample_steps, sample_guide_scale, infer_frames,
                               size, base_seed, sample_solver):
    """
    The actual inference computation, called by both rank 0 and worker ranks.
    This function does NOT do any broadcasting - that's handled by the caller.
    """
    global wan_s2v_pipeline, global_args, global_cfg
    global global_rank, global_world_size, global_save_rank
    
    try:
        logging.info(f"[Rank {global_rank}] Generating video...")
        logging.info(f"  Prompt: {prompt}")
        logging.info(f"  Image: {image_path}")
        logging.info(f"  Audio: {audio_path}")
        logging.info(f"  Num_clip: {num_clip}")
        logging.info(f"  Sample_steps: {sample_steps}")
        logging.info(f"  Guide_scale: {sample_guide_scale}")
        logging.info(f"  Infer_frames: {infer_frames}")
        logging.info(f"  Size: {size}")
        logging.info(f"  Seed: {base_seed}")
        logging.info(f"  Solver: {sample_solver}")
        
        # Generate video
        video, dataset_info = wan_s2v_pipeline.generate(
            input_prompt=prompt,
            ref_image_path=image_path,
            audio_path=audio_path,
            enable_tts=global_args.enable_tts,
            tts_prompt_audio=None,
            tts_prompt_text=None,
            tts_text=None,
            num_repeat=num_clip,
            pose_video=global_args.pose_video,
            generate_size=size,
            max_area=MAX_AREA_CONFIGS[size],
            infer_frames=infer_frames,
            shift=global_args.sample_shift,
            sample_solver=sample_solver,
            sampling_steps=sample_steps,
            guide_scale=sample_guide_scale,
            seed=base_seed,
            offload_model=global_args.offload_model,
            init_first_frame=global_args.start_from_ref,
            use_dataset=False,
            dataset_sample_idx=0,
            drop_motion_noisy=global_args.drop_motion_noisy,
            num_gpus_dit=global_args.num_gpus_dit,
            enable_vae_parallel=global_args.enable_vae_parallel,
            input_video_for_sam2=None,
        )
        
        logging.info(f"[Rank {global_rank}] Denoising video done")
        
        # Save video (only on designated rank)
        video_path = None
        if global_rank == global_save_rank:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = prompt.replace(" ", "_").replace("/", "_")[:50]
            suffix = '.mp4'
            
            save_file = f"{formatted_time}_{sample_steps}step_{formatted_prompt}"
            
            if global_args.lora_path_dmd is not None:
                # Only add lora suffix for .pt files (local paths with sufficient depth)
                path_parts = global_args.lora_path_dmd.split("/")
                if global_args.lora_path_dmd.endswith(".pt") and len(path_parts) >= 3:
                    save_file = save_file + "_" + path_parts[-3] + "_" + path_parts[-1].split(".")[0]
            
            save_dir = global_args.save_dir
            os.makedirs(save_dir, exist_ok=True)
            save_file = os.path.join(save_dir, save_file + suffix)
            
            logging.info(f"Saving generated video to {save_file}")
            save_video(
                tensor=video[None],
                save_file=save_file,
                fps=global_cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
            
            # Merge audio
            if "s2v" in global_args.task:
                if global_args.enable_tts is False:
                    merge_video_audio(video_path=save_file, audio_path=audio_path)
                else:
                    merge_video_audio(video_path=save_file, audio_path="tts.wav")
            
            video_path = save_file
            logging.info(f"Video saved successfully: {video_path}")
        
        # Clean up
        del video
        torch.cuda.empty_cache()
        
        # Synchronize in distributed mode
        if dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()
            # Broadcast video path to all ranks (for consistency)
            video_path_list = [video_path] if global_rank == global_save_rank else [None]
            dist.broadcast_object_list(video_path_list, src=global_save_rank)
            video_path = video_path_list[0]
            
            logging.info(f"[Rank {global_rank}] Inference completed, synchronized with all ranks")
        
        return video_path
        
    except Exception as e:
        error_msg = f"Error during generation: {str(e)}"
        logging.error(error_msg)
        import traceback
        traceback.print_exc()
        return None


def create_gradio_interface():
    """
    Create Gradio web interface
    Only called on rank 0
    """
    brand_theme = gr.themes.Base(
        primary_hue=gr.themes.colors.green,
        neutral_hue=gr.themes.colors.gray,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ).set(
        body_background_fill=BRAND_BLACK,
        body_background_fill_dark=BRAND_BLACK,
        body_text_color="#FFFFFF",
        background_fill_primary="#141414",
        background_fill_secondary="#1C1C1C",
        block_background_fill="#141414",
        block_border_color="#2A2A2A",
        block_label_text_color=BRAND_GREEN,
        border_color_primary="#2A2A2A",
        button_primary_background_fill=BRAND_GREEN,
        button_primary_background_fill_hover="#00C853",
        button_primary_text_color="#000000",
        input_background_fill="#1C1C1C",
    )

    brand_css = f"""
    .lc-header {{ text-align:center; padding: 28px 0 8px; }}
    .lc-logo {{ font-weight:800; letter-spacing:.18em; font-size:2.4rem; color:#FFFFFF; line-height:1; }}
    .lc-logo .amp {{ color:{BRAND_GREEN}; }}
    .lc-sub {{ font-weight:600; letter-spacing:.42em; font-size:.95rem; color:{BRAND_GREEN}; margin-top:6px; }}
    .lc-product {{ font-weight:700; letter-spacing:.06em; font-size:1.15rem; color:#FFFFFF; margin-top:14px; }}
    .lc-tagline {{ color:#A0A0A0; font-size:.95rem; margin-top:4px; }}
    .lc-footer {{ text-align:center; color:#6E6E6E; font-size:.8rem; padding:24px 0 12px; border-top:1px solid #2A2A2A; margin-top:24px; }}
    .lc-footer .amp {{ color:{BRAND_GREEN}; }}
    """

    with gr.Blocks(title=f"{BRAND_NAME} {BRAND_SUB} — {BRAND_PRODUCT}", theme=brand_theme, css=brand_css) as demo:
        gr.HTML(f"""
        <div class="lc-header">
          <div class="lc-logo">LYNK <span class="amp">&amp;</span> CO</div>
          <div class="lc-sub">{BRAND_SUB}</div>
          <div class="lc-product">{BRAND_PRODUCT}</div>
          <div class="lc-tagline">{BRAND_TAGLINE}</div>
        </div>
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Basic Input")
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the video content you want to generate...",
                    value=EXAMPLE_PROMPT["s2v-14B"]["prompt"],
                    lines=5
                )

                # Image input with gallery
                image_input = gr.Image(
                    label="Reference Image",
                    type="filepath"
                )

                gr.Markdown("**Example Images (click to select):**")
                example_gallery = gr.Gallery(
                    value=[img_path for img_path, label in EXAMPLE_IMAGES if os.path.exists(img_path)],
                    label="",
                    show_label=False,
                    columns=4,
                    rows=2,
                    height=200,
                    object_fit="cover"
                )
                
                # Audio input with examples
                audio_input = gr.Audio(
                    label="Audio File",
                    type="filepath"
                )

                example_audio_dropdown = gr.Dropdown(
                    choices=[(label, audio_path) for audio_path, label in EXAMPLE_AUDIOS if os.path.exists(audio_path)],
                    label="Example Audio (auto-fill on selection)",
                    show_label=True,
                    value=None
                )

                with gr.Accordion("Advanced Parameters", open=False):
                    gr.Markdown("### Generation Parameters")
                    with gr.Row():
                        num_clip_input = gr.Slider(
                            minimum=1,
                            maximum=10000,
                            value=global_args.num_clip,
                            step=1,
                            label="Number of Clips"
                        )
                        sample_steps_input = gr.Slider(
                            minimum=1,
                            maximum=50,
                            value=global_args.sample_steps,
                            step=1,
                            label="Sampling Steps"
                        )

                    with gr.Row():
                        sample_guide_scale_input = gr.Slider(
                            minimum=0.0,
                            maximum=10.0,
                            value=global_args.sample_guide_scale,
                            step=0.1,
                            label="Guidance Scale"
                        )
                        infer_frames_input = gr.Slider(
                            minimum=16,
                            maximum=160,
                            value=global_args.infer_frames,
                            step=4,
                            label="Frames per Clip"
                        )

                    with gr.Row():
                        size_input = gr.Dropdown(
                            choices=list(SIZE_CONFIGS.keys()),
                            value=global_args.size,
                            label="Video Size"
                        )
                        base_seed_input = gr.Number(
                            value=global_args.base_seed,
                            label="Random Seed",
                            precision=0
                        )

                    sample_solver_input = gr.Dropdown(
                        choices=['euler', 'unipc', 'dpm++'],
                        value=global_args.sample_solver,
                        label="Sampler"
                    )

                with gr.Accordion("LoRA Settings", open=False):
                    gr.Markdown("### LoRA Weights")
                    use_lora_checkbox = gr.Checkbox(
                        label="Enable LoRA",
                        value=global_args.load_lora
                    )
                    lora_path_input = gr.Textbox(
                        label="LoRA Path",
                        placeholder="Enter local path to LoRA weights",
                        value=global_args.lora_path_dmd or ""
                    )
                    reload_pipe_btn = gr.Button("Reload Pipeline (Apply LoRA)", variant="secondary")

                generate_btn = gr.Button("Generate", variant="primary", size="lg")

            with gr.Column():
                gr.Markdown("### Generation Result")
                video_output = gr.Video(label="Generated Video")
                status_output = gr.Textbox(label="Status", lines=3)
        
        # Add example combinations
        gr.Markdown("### Quick Examples")
        gr.Examples(
            examples=[
                [
                    "A young boy with curly brown hair and a white polo shirt, wearing a backpack, standing outdoors in a sunny park. He is speaking animatedly with expressive hand gestures, his hands clearly visible and moving naturally. His facial expressions are lively and engaging, conveying enthusiasm and curiosity. The camera captures him from the chest up with a steady, clear shot, natural lighting and a soft green background. Realistic 3D animation style, lifelike motion and expression.",
                    "examples/boy.jpg",
                    "examples/boy.wav",
                    10000,
                    4,
                    0.0,
                    48,
                    "704*384",
                    420,
                    "euler"
                ],
                [
                    "A stout, cheerful dwarf with a magnificent braided beard adorned with metal rings, wearing a heavy leather apron. He's standing in his fiery, cluttered forge, laughing heartily as he explains the mastery of his craft, holding up a glowing hammer. Style of Blizzard Entertainment cinematics (like World of Warcraft), warm, dynamic lighting from the forge.",
                    "examples/dwarven_blacksmith.jpg",
                    "examples/dwarven_blacksmith.wav",
                    10,
                    4,
                    0.0,
                    48,
                    "704*384",
                    420,
                    "euler"
                ],
            ],
            inputs=[
                prompt_input, image_input, audio_input, num_clip_input,
                sample_steps_input, sample_guide_scale_input, infer_frames_input,
                size_input, base_seed_input, sample_solver_input
            ],
            examples_per_page=2
        )
        
        gr.HTML(f"""
        <div class="lc-footer">LYNK <span class="amp">&amp;</span> CO {BRAND_SUB} · {BRAND_PRODUCT}</div>
        """)

        def generate_wrapper(prompt, image, audio, num_clip, sample_steps,
                           sample_guide_scale, infer_frames, size, base_seed, sample_solver):
            """Wrapper function for Gradio interface"""
            if not prompt or not image or not audio:
                return None, "Error: Please provide all required inputs (prompt, image, audio)"

            try:
                status = f"Generating video...\nParameters: steps={sample_steps}, clips={num_clip}, frames={infer_frames}"
                video_path = run_single_sample(
                    prompt, image, audio, num_clip,
                    sample_steps, sample_guide_scale, infer_frames,
                    size, int(base_seed), sample_solver
                )

                if video_path and os.path.exists(video_path):
                    status = f"Generation successful!\nVideo saved at: {video_path}"
                    return video_path, status
                else:
                    status = "Generation failed, please check logs"
                    return None, status
            except Exception as e:
                status = f"Error: {str(e)}"
                return None, status
        
        def reload_wrapper(use_lora, lora_path):
            """Wrapper for pipeline reloading"""
            return reload_pipeline_command(use_lora, lora_path)
        
        def select_example_image(evt: gr.SelectData):
            """Handle example image selection"""
            selected_index = evt.index
            if selected_index < len(EXAMPLE_IMAGES):
                img_path = EXAMPLE_IMAGES[selected_index][0]
                if os.path.exists(img_path):
                    return img_path
            return None
        
        def select_example_audio(audio_path):
            """Handle example audio selection"""
            if audio_path and os.path.exists(audio_path):
                return audio_path
            return None
        
        # Connect event handlers
        example_gallery.select(
            fn=select_example_image,
            outputs=image_input
        )
        
        example_audio_dropdown.change(
            fn=select_example_audio,
            inputs=example_audio_dropdown,
            outputs=audio_input
        )
        
        generate_btn.click(
            fn=generate_wrapper,
            inputs=[
                prompt_input, image_input, audio_input, num_clip_input,
                sample_steps_input, sample_guide_scale_input, infer_frames_input,
                size_input, base_seed_input, sample_solver_input
            ],
            outputs=[video_output, status_output]
        )

        reload_pipe_btn.click(
            fn=reload_wrapper,
            inputs=[use_lora_checkbox, lora_path_input],
            outputs=[status_output]
        )
    
    return demo


def worker_loop():
    """
    Worker loop for non-rank-0 processes in distributed mode.
    They continuously wait for broadcast signals from rank 0 and participate in computation.
    NCCL timeout is set to infinite via environment variable.
    """
    logging.info(f"Rank {global_rank} entering worker loop, waiting for inference requests...")
    
    while True:
        try:
            # Wait for broadcast from rank 0
            # [command, prompt, image_path, audio_path, num_clip, sample_steps, sample_guide_scale, 
            #  infer_frames, size, base_seed, sample_solver, lora_path, load_lora]
            inputs = [None] * 13
            dist.broadcast_object_list(inputs, src=0)
            
            command = inputs[0]
            
            if command == "infer":
                _, prompt, image_path, audio_path, num_clip, \
                    sample_steps, sample_guide_scale, infer_frames, \
                    size, base_seed, sample_solver, _, _ = inputs
                
                # If we receive a valid prompt, participate in computation
                if prompt is not None and prompt != "":
                    logging.info(f"[Rank {global_rank}] Received valid inference request")
                    # Now do the actual computation (skip the broadcast part since we already received it)
                    _run_inference_computation(
                        prompt, image_path, audio_path, num_clip,
                        sample_steps, sample_guide_scale, infer_frames,
                        size, base_seed, sample_solver
                    )
                    logging.info(f"[Rank {global_rank}] Generation completed")
            elif command == "reload":
                _, _, _, _, _, _, _, _, _, _, _, lora_path, load_lora = inputs
                logging.info(f"[Rank {global_rank}] Received reload request: load_lora={load_lora}, lora_path={lora_path}")
                global_args.load_lora = load_lora
                global_args.lora_path_dmd = lora_path
                initialize_pipeline(global_args, global_training_settings)
            elif command == "idle":
                logging.debug(f"[Rank {global_rank}] Received idle signal, continuing to wait...")
            else:
                if command is not None:
                    logging.warning(f"[Rank {global_rank}] Received unknown command: {command}")
                
        except Exception as e:
            logging.error(f"[Rank {global_rank}] Error in worker loop: {e}")
            import traceback
            traceback.print_exc()
            import time
            time.sleep(1)


def main():
    """Main entry point"""
    args = _parse_args()
    training_settings = training_config_parser(args.training_config)
    
    # Initialize pipeline
    logging.info("Initializing pipeline...")
    initialize_pipeline(args, training_settings)
    
    # Launch Gradio interface (only on rank 0)
    if global_rank == 0:
        logging.info(f"Launching LiveAvatar Gradio interface on {args.server_name}:{args.server_port}")
        logging.info("Note: In multi-GPU mode, other ranks are waiting in background")
        demo = create_gradio_interface()
        demo.launch(
            server_name=args.server_name,
            server_port=args.server_port,
            share=False
        )
    else:
        # Other ranks enter worker loop to wait for and process requests
        worker_loop()


if __name__ == "__main__":
    main()

