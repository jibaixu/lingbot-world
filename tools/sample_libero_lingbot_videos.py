#!/usr/bin/env python3
"""Sample LIBERO frames and save decoded LingBot-World-Fast videos."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import wan  # noqa: E402
from tools.extract_lingbot_latents import (  # noqa: E402
    DEFAULT_INTRINSICS,
    FAST_TIMESTEP_INDEX,
    TEACHER_PROMPT,
    decode_anchor_frames,
    get_text_context,
    model_config,
    prepare_geometry,
    read_jsonl,
    rotation_matrix_xyz,
    video_path_for,
)
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS  # noqa: E402
from wan.distributed.util import init_distributed_group  # noqa: E402
from wan.utils.cam_utils import (  # noqa: E402
    compute_relative_poses,
    get_Ks_transformed,
    get_plucker_embeddings,
    interpolate_camera_poses,
)
from wan.utils.utils import save_video  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample LIBERO frames and save LingBot decoded videos."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data/jibaixu/Datasets/LIBERO/libero_lerobot_v2.1"),
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path("/data/jibaixu/Models/lingbot-world/lingbot-world-base-cam"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/jbx/Codes/lingbot-world/output/extract_style_teacher_shift10_midcam"),
    )
    parser.add_argument("--task", default="i2v-A14B", choices=list(WAN_CONFIGS.keys()))
    parser.add_argument("--size", default="480*832", choices=list(MAX_AREA_CONFIGS.keys()))
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--num-denoise-steps", type=int, default=4)
    parser.add_argument("--frame-num", type=int, default=9)
    parser.add_argument("--chunk-size", type=int, default=3)
    parser.add_argument("--sample-shift", type=float, default=10.0)
    parser.add_argument("--video-key", default="observation.images.image")
    parser.add_argument("--prompt", default=TEACHER_PROMPT)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--ulysses-size", type=int, default=8)
    parser.add_argument("--t5-cpu", action="store_true")
    parser.add_argument("--no-prewarm", action="store_true")
    parser.add_argument("--max-attention-size", type=int, default=None)
    parser.add_argument("--yaw-range", type=float, default=0.20)
    parser.add_argument("--pitch-range", type=float, default=0.10)
    parser.add_argument("--roll-range", type=float, default=0.04)
    parser.add_argument("--translation-norm-min", type=float, default=0.28)
    parser.add_argument("--translation-norm-max", type=float, default=0.36)
    parser.add_argument(
        "--intrinsics",
        type=float,
        nargs=4,
        default=DEFAULT_INTRINSICS,
        metavar=("FX", "FY", "CX", "CY"),
    )
    return parser.parse_args()


def init_distributed() -> Tuple[int, int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LingBot-World inference.")

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
        init_distributed_group()

    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)

    return rank, local_rank, world_size


def broadcast_object(value: Any, src: int = 0) -> Any:
    if not dist.is_initialized():
        return value
    payload = [value]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def validate_args(args: argparse.Namespace) -> None:
    cfg = WAN_CONFIGS[args.task]
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if not 1 <= args.num_denoise_steps <= len(FAST_TIMESTEP_INDEX):
        raise ValueError("--num-denoise-steps must be in [1, 4]")
    if args.frame_num <= 0 or (args.frame_num - 1) % cfg.vae_stride[0] != 0:
        raise ValueError(f"--frame-num must be 4n+1; got {args.frame_num}")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.translation_norm_min <= 0:
        raise ValueError("--translation-norm-min must be positive")
    if args.translation_norm_max < args.translation_norm_min:
        raise ValueError("--translation-norm-max must be >= --translation-norm-min")

    lat_f = (args.frame_num - 1) // cfg.vae_stride[0] + 1
    adjusted_lat_f = lat_f - (lat_f % args.chunk_size)
    if adjusted_lat_f <= 0:
        raise ValueError(
            f"--frame-num {args.frame_num} is too short for --chunk-size {args.chunk_size}"
        )


def sample_episode_frames(
    episodes: List[Mapping[str, Any]],
    *,
    count: int,
    seed: int,
) -> List[Dict[str, int]]:
    eligible = [episode for episode in episodes if int(episode["length"]) > 0]
    if len(eligible) < count:
        raise ValueError(f"Requested {count} samples, but only {len(eligible)} episodes are eligible")

    rng = random.Random(seed)
    chosen = rng.sample(eligible, count)
    samples: List[Dict[str, int]] = []
    for sample_index, episode in enumerate(chosen):
        episode_index = int(episode["episode_index"])
        episode_length = int(episode["length"])
        frame_index = rng.randrange(episode_length)
        samples.append(
            {
                "sample_index": sample_index,
                "episode_index": episode_index,
                "episode_length": episode_length,
                "frame_index": frame_index,
            }
        )
    return samples


def sample_seed(base_seed: int, episode_index: int, frame_index: int) -> int:
    return (base_seed + episode_index * 1_000_003 + frame_index * 9_176) % (2**31)


def deterministic_midnorm_camera_trajectory(
    *,
    episode_index: int,
    frame_index: int,
    frame_num: int,
    base_seed: int,
    yaw_range: float,
    pitch_range: float,
    roll_range: float,
    translation_norm_min: float,
    translation_norm_max: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    seed = (base_seed + episode_index * 1_000_003 + frame_index * 9_176) % (2**32)
    rng = np.random.default_rng(seed)
    yaw = float(rng.uniform(-yaw_range, yaw_range))
    pitch = float(rng.uniform(-pitch_range, pitch_range))
    roll = float(rng.uniform(-roll_range, roll_range))

    direction = rng.normal(size=3).astype(np.float32)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm < 1e-6:
        direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        direction = direction / direction_norm
    translation_norm = float(rng.uniform(translation_norm_min, translation_norm_max))
    translation = direction * translation_norm

    poses = np.zeros((frame_num, 4, 4), dtype=np.float32)
    for i in range(frame_num):
        t = i / max(frame_num - 1, 1)
        alpha = t * t * (3.0 - 2.0 * t)
        poses[i, :3, :3] = rotation_matrix_xyz(roll * alpha, pitch * alpha, yaw * alpha)
        poses[i, :3, 3] = translation * alpha
        poses[i, 3, 3] = 1.0

    camera_meta = {
        "seed": int(seed),
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "translation": [float(x) for x in translation.tolist()],
        "translation_norm": translation_norm,
        "translation_norm_range": [float(translation_norm_min), float(translation_norm_max)],
    }
    return poses, camera_meta


def build_camera_condition(
    *,
    pipe: wan.WanI2VFast,
    geom: Mapping[str, int],
    episode_index: int,
    frame_index: int,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    poses, camera_meta = deterministic_midnorm_camera_trajectory(
        episode_index=episode_index,
        frame_index=frame_index,
        frame_num=geom["frame_num"],
        base_seed=args.base_seed,
        yaw_range=args.yaw_range,
        pitch_range=args.pitch_range,
        roll_range=args.roll_range,
        translation_norm_min=args.translation_norm_min,
        translation_norm_max=args.translation_norm_max,
    )

    c2ws = poses[: geom["frame_num"]]
    ks = torch.tensor([args.intrinsics], dtype=torch.float32).repeat(geom["frame_num"], 1)
    ks = get_Ks_transformed(
        ks,
        height_org=480,
        width_org=832,
        height_resize=geom["height"],
        width_resize=geom["width"],
        height_final=geom["height"],
        width_final=geom["width"],
    )[0]

    len_c2ws = len(c2ws)
    len_c2ws_ = int((len_c2ws - 1) // 4) + 1
    len_c2ws_ = int(len_c2ws_ - (len_c2ws_ % args.chunk_size))
    c2ws_infer = interpolate_camera_poses(
        src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
        src_rot_mat=c2ws[:, :3, :3],
        src_trans_vec=c2ws[:, :3, 3],
        tgt_indices=np.linspace(0, len_c2ws - 1, len_c2ws_),
    )
    c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
    ks = ks.repeat(len(c2ws_infer), 1)

    c2ws_infer = c2ws_infer.to(pipe.device)
    ks = ks.to(pipe.device)
    c2ws_plucker_emb = get_plucker_embeddings(
        c2ws_infer,
        ks,
        geom["height"],
        geom["width"],
        only_rays_d=False,
    )
    c2ws_plucker_emb = rearrange(
        c2ws_plucker_emb,
        "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
        c1=int(geom["height"] // geom["lat_h"]),
        c2=int(geom["width"] // geom["lat_w"]),
    )
    c2ws_plucker_emb = c2ws_plucker_emb[None, ...]
    camera_condition = rearrange(
        c2ws_plucker_emb,
        "b (f h w) c -> b c f h w",
        f=geom["lat_f"],
        h=geom["lat_h"],
        w=geom["lat_w"],
    ).to(pipe.param_dtype)
    return camera_condition, camera_meta


def generate_video_for_frame(
    *,
    pipe: wan.WanI2VFast,
    context: List[torch.Tensor],
    img: Image.Image,
    episode_index: int,
    frame_index: int,
    args: argparse.Namespace,
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    geom = prepare_geometry(pipe, img, args)
    frame_num = geom["frame_num"]
    height, width = geom["height"], geom["width"]
    lat_h, lat_w, lat_f = geom["lat_h"], geom["lat_w"], geom["lat_f"]

    img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).to(pipe.device)
    seed = sample_seed(args.base_seed, episode_index, frame_index)
    generator = torch.Generator(device=pipe.device)
    generator.manual_seed(seed)
    noise = torch.randn(
        16,
        lat_f,
        lat_h,
        lat_w,
        dtype=torch.float32,
        generator=generator,
        device=pipe.device,
    )

    msk = torch.ones(1, frame_num, lat_h, lat_w, device=pipe.device)
    msk[:, 1:] = 0
    msk = torch.concat(
        [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
        dim=1,
    )
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]

    c2ws_plucker_emb, camera_meta = build_camera_condition(
        pipe=pipe,
        geom=geom,
        episode_index=episode_index,
        frame_index=frame_index,
        args=args,
    )

    y = pipe.vae.encode(
        [
            torch.concat(
                [
                    torch.nn.functional.interpolate(
                        img_tensor[None].cpu(),
                        size=(height, width),
                        mode="bicubic",
                    ).transpose(0, 1),
                    torch.zeros(3, frame_num - 1, height, width),
                ],
                dim=1,
            ).to(pipe.device)
        ]
    )[0]
    y = torch.concat([msk, y])

    pipe.scheduler.set_timesteps(pipe.num_train_timesteps, shift=args.sample_shift)
    timesteps_index = FAST_TIMESTEP_INDEX[: args.num_denoise_steps]
    timesteps = pipe.scheduler.timesteps[timesteps_index]

    args_model = model_config(pipe.model)
    transformer_dtype = pipe.pipe_dtype
    frame_seqlen = geom["frame_seqlen"]
    if pipe.local_attn_size > -1:
        kv_size = frame_seqlen * pipe.local_attn_size
    else:
        kv_size = frame_seqlen * lat_f
    head_dim = args_model.dim // args_model.num_heads
    local_num_heads = args_model.num_heads // pipe.sp_size
    self_kv_cache = pipe._initialize_self_kv_cache(
        num_layers=args_model.num_layers,
        shape=[1, kv_size, local_num_heads, head_dim],
        dtype=transformer_dtype,
        device=pipe.device,
    )
    cross_kv_cache = pipe._initialize_crossattn_cache(
        num_layers=args_model.num_layers,
        shape=[1, args_model.text_len, args_model.num_heads, head_dim],
        dtype=transformer_dtype,
        device=pipe.device,
    )

    @contextmanager
    def noop_no_sync() -> Iterable[None]:
        yield

    no_sync_model = getattr(pipe.model, "no_sync", noop_no_sync)
    pipe._cross_attn_initialized = False
    videos: Optional[torch.Tensor] = None

    with torch.amp.autocast("cuda", dtype=pipe.param_dtype), torch.no_grad(), no_sync_model():
        latents_chunk = noise.split(args.chunk_size, dim=1)
        condition_chunk = y.split(args.chunk_size, dim=1)
        c2ws_plucker_emb_chunk = c2ws_plucker_emb.split(args.chunk_size, dim=2)
        pred_latent_chunks = []

        for chunk_id in range(len(latents_chunk)):
            current_latent = latents_chunk[chunk_id]
            current_condition = condition_chunk[chunk_id]
            current_c2ws_plucker_emb = c2ws_plucker_emb_chunk[chunk_id]
            dit_cond_dict = {
                "c2ws_plucker_emb": current_c2ws_plucker_emb.chunk(1, dim=0),
            }
            kwargs = {
                "context": [context[0]],
                "seq_len": geom["max_seq_len"],
                "y": [current_condition],
                "dit_cond_dict": dit_cond_dict,
                "kv_cache": self_kv_cache,
                "crossattn_cache": cross_kv_cache,
                "current_start": chunk_id * args.chunk_size * frame_seqlen,
                "max_attention_size": kv_size
                if args.max_attention_size is None
                else args.max_attention_size,
                "frame_seqlen": frame_seqlen,
            }

            x0 = None
            for timestep_idx, timestep_value in enumerate(timesteps):
                timestep = torch.stack([timestep_value]).to(pipe.device)
                noise_pred = pipe.model(
                    x=[current_latent.to(pipe.device)],
                    t=timestep,
                    cross_attn_first_call=not pipe._cross_attn_initialized,
                    **kwargs,
                )[0]
                pipe._cross_attn_initialized = True
                x0 = pipe._convert_flow_pred_to_x0(
                    flow_pred=noise_pred,
                    xt=current_latent,
                    timestep=timestep_value,
                    scheduler=pipe.scheduler,
                )
                if timestep_idx < len(timesteps) - 1:
                    next_timestep = timesteps[timestep_idx + 1]
                    current_latent = pipe.scheduler.add_noise(
                        x0,
                        torch.randn(
                            x0.shape,
                            generator=generator,
                            device=x0.device,
                            dtype=x0.dtype,
                        ),
                        next_timestep,
                    )

            if x0 is None:
                raise RuntimeError("No denoising step was executed")
            pred_latent_chunks.append(x0)

            context_timestep = torch.stack([timesteps[-1] * 0.0]).to(pipe.device)
            pipe.model(
                x=[x0],
                t=context_timestep,
                cross_attn_first_call=False,
                **kwargs,
            )

        pred_latents = torch.cat(pred_latent_chunks, dim=1)
        if pipe.rank == 0:
            videos = pipe.vae.decode([pred_latents])[0]

    meta = {
        "seed": int(seed),
        "timesteps": [int(t.item()) for t in timesteps],
        "timesteps_index": [int(i) for i in timesteps_index],
        "height": int(height),
        "width": int(width),
        "frame_num": int(frame_num),
        "lat_f": int(lat_f),
        "lat_h": int(lat_h),
        "lat_w": int(lat_w),
        "camera": camera_meta,
    }
    del noise, y, c2ws_plucker_emb, self_kv_cache, cross_kv_cache
    torch.cuda.empty_cache()
    return videos, meta


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    rank, local_rank, world_size = init_distributed()
    if args.ulysses_size > 1 and args.ulysses_size != world_size:
        raise ValueError(
            f"--ulysses-size must equal WORLD_SIZE when enabled; "
            f"got ulysses_size={args.ulysses_size}, world_size={world_size}"
        )
    if world_size > 1 and WAN_CONFIGS[args.task].num_heads % world_size != 0:
        raise ValueError(
            f"num_heads={WAN_CONFIGS[args.task].num_heads} is not divisible by world_size={world_size}"
        )

    info = json.load((args.data_root / "meta" / "info.json").open("r", encoding="utf-8"))
    episodes = read_jsonl(args.data_root / "meta" / "episodes.jsonl")
    if not episodes:
        raise RuntimeError("No LIBERO episodes found")

    samples = None
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        samples = sample_episode_frames(
            episodes,
            count=args.num_samples,
            seed=args.base_seed,
        )
        logging.info(
            "Generating %d LIBERO LingBot samples into %s with world_size=%d",
            len(samples),
            args.output_dir,
            world_size,
        )
    samples = broadcast_object(samples)

    cfg = WAN_CONFIGS[args.task]
    pipe = wan.WanI2VFast(
        config=cfg,
        checkpoint_dir=str(args.ckpt_dir),
        device_id=local_rank,
        rank=rank,
        t5_fsdp=world_size > 1 and not args.t5_cpu,
        dit_fsdp=world_size > 1,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
    )
    context = get_text_context(pipe, args.prompt)

    if not args.no_prewarm:
        first = samples[0]
        first_video = video_path_for(
            args.data_root,
            info,
            int(first["episode_index"]),
            args.video_key,
        )
        first_frame = decode_anchor_frames(first_video, [int(first["frame_index"])])[
            int(first["frame_index"])
        ]
        pipe.prewarm(
            first_frame,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            chunk_size=args.chunk_size,
        )

    manifest_samples: List[Dict[str, Any]] = []
    iterator = tqdm(samples, disable=(rank != 0), desc="libero_lingbot_samples")
    for sample in iterator:
        episode_index = int(sample["episode_index"])
        frame_index = int(sample["frame_index"])
        video_path = video_path_for(args.data_root, info, episode_index, args.video_key)
        frame = decode_anchor_frames(video_path, [frame_index])[frame_index]

        video, meta = generate_video_for_frame(
            pipe=pipe,
            context=context,
            img=frame,
            episode_index=episode_index,
            frame_index=frame_index,
            args=args,
        )

        if rank == 0:
            output_name = (
                f"sample_{int(sample['sample_index']):02d}_"
                f"ep{episode_index:06d}_frame{frame_index:06d}_"
                f"seed{meta['seed']}.mp4"
            )
            output_path = args.output_dir / output_name
            if video is None:
                raise RuntimeError("Rank 0 did not receive decoded video")
            save_video(
                tensor=video[None],
                save_file=str(output_path),
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise RuntimeError(f"Failed to save video: {output_path}")

            record = {
                **sample,
                **meta,
                "video_key": args.video_key,
                "input_video": str(video_path),
                "output_video": str(output_path),
            }
            manifest_samples.append(record)
            logging.info(
                "saved sample=%02d episode=%06d frame=%06d -> %s",
                int(sample["sample_index"]),
                episode_index,
                frame_index,
                output_path,
            )

        del frame, video
        if dist.is_initialized():
            dist.barrier()

    if rank == 0:
        manifest = {
            "data_root": str(args.data_root),
            "ckpt_dir": str(args.ckpt_dir),
            "output_dir": str(args.output_dir),
            "task": args.task,
            "size": args.size,
            "prompt": args.prompt,
            "base_seed": args.base_seed,
            "num_denoise_steps": args.num_denoise_steps,
            "frame_num": args.frame_num,
            "chunk_size": args.chunk_size,
            "sample_shift": args.sample_shift,
            "yaw_range": args.yaw_range,
            "pitch_range": args.pitch_range,
            "roll_range": args.roll_range,
            "translation_norm_min": args.translation_norm_min,
            "translation_norm_max": args.translation_norm_max,
            "video_key": args.video_key,
            "samples": manifest_samples,
        }
        write_manifest(args.output_dir / "manifest.json", manifest)
        logging.info("wrote manifest: %s", args.output_dir / "manifest.json")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
