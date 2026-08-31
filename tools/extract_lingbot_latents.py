#!/usr/bin/env python3
"""Extract stride-shared LingBot-World-Fast teacher latents for LIBERO.

The script stores one safetensors file per episode. Each file contains
temporally pooled block features for anchor frames spaced by --share-stride.
Run with torchrun when using FSDP / sequence parallelism.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import imageio.v3 as iio
import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import wan  # noqa: E402
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS  # noqa: E402
from wan.distributed.util import init_distributed_group  # noqa: E402
from wan.utils.cam_utils import (  # noqa: E402
    compute_relative_poses,
    get_Ks_transformed,
    get_plucker_embeddings,
    interpolate_camera_poses,
)


TEACHER_PROMPT = "A slight camera motion with stable object layout and unchanged spatial relations."
FAST_TIMESTEP_INDEX = [0, 179, 358, 679]
DEFAULT_INTRINSICS = [415.5298, 415.6922, 415.77786, 239.77779]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract LingBot-World-Fast block features for LIBERO episodes."
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
    parser.add_argument("--task", default="i2v-A14B", choices=list(WAN_CONFIGS.keys()))
    parser.add_argument("--size", default="480*832", choices=list(MAX_AREA_CONFIGS.keys()))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--share-stride", type=int, default=16)
    parser.add_argument("--num-denoise-steps", type=int, default=2)
    parser.add_argument("--layer", type=int, default=24, help="1-based DiT block index.")
    parser.add_argument("--frame-num", type=int, default=9)
    parser.add_argument("--chunk-size", type=int, default=3)
    parser.add_argument("--sample-shift", type=float, default=5.0)
    parser.add_argument("--video-key", default="observation.images.image")
    parser.add_argument("--prompt", default=TEACHER_PROMPT)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument(
        "--num-teacher-samples",
        type=int,
        default=1,
        help="Number of diffusion realizations per anchor. Use 4 with --confidence-aware.",
    )
    parser.add_argument(
        "--confidence-aware",
        action="store_true",
        help="Average multiple realizations and store confidence-aware metadata/tensors.",
    )
    parser.add_argument("--ulysses-size", type=int, default=8)
    parser.add_argument("--episode-start", type=int, default=None)
    parser.add_argument("--episode-end", type=int, default=None)
    parser.add_argument(
        "--episode-indices",
        default=None,
        help="Comma-separated episode ids or a text file containing one id per line.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-prewarm", action="store_true")
    parser.add_argument("--t5-cpu", action="store_true")
    parser.add_argument("--max-attention-size", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--yaw-range", type=float, default=0.15)
    parser.add_argument("--pitch-range", type=float, default=0.08)
    parser.add_argument("--roll-range", type=float, default=0.03)
    parser.add_argument("--translation-range", type=float, default=0.12)
    parser.add_argument(
        "--intrinsics",
        type=float,
        nargs=4,
        default=DEFAULT_INTRINSICS,
        metavar=("FX", "FY", "CX", "CY"),
    )
    return parser.parse_args()


def init_distributed() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
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


def validate_args(args: argparse.Namespace, cfg: Any) -> None:
    if args.share_stride <= 0:
        raise ValueError("--share-stride must be positive")
    if not 1 <= args.num_denoise_steps <= len(FAST_TIMESTEP_INDEX):
        raise ValueError("--num-denoise-steps must be in [1, 4]")
    if not 1 <= args.layer <= cfg.num_layers:
        raise ValueError(f"--layer must be in [1, {cfg.num_layers}]")
    if args.num_teacher_samples <= 0:
        raise ValueError("--num-teacher-samples must be positive")
    if args.confidence_aware and args.num_teacher_samples < 2:
        raise ValueError("--confidence-aware requires --num-teacher-samples >= 2")
    lat_f = (args.frame_num - 1) // cfg.vae_stride[0] + 1
    adjusted_lat_f = lat_f - (lat_f % args.chunk_size)
    if adjusted_lat_f != args.chunk_size:
        raise ValueError(
            "This extractor expects one latent chunk. Use the paper-style "
            f"--frame-num 9 --chunk-size 3. Got frame_num={args.frame_num}, "
            f"chunk_size={args.chunk_size}, lat_f={lat_f}, adjusted_lat_f={adjusted_lat_f}."
        )


def default_output_dir(args: argparse.Namespace) -> Path:
    name = (
        f"lingbot_latent_step{args.num_denoise_steps}"
        f"_numlayer{args.layer}"
        f"_stride{args.share_stride}"
    )
    if args.confidence_aware:
        name += f"_samples{args.num_teacher_samples}_confv1"
    return args.data_root / name


def validate_output_dir_name(output_dir: Path, share_stride: int) -> None:
    required = f"_stride{share_stride}"
    if required not in output_dir.name:
        raise ValueError(
            f"--output-dir must include {required!r} in the directory name; "
            f"got {output_dir}"
        )


def read_jsonl(path: Path) -> List[Mapping[str, Any]]:
    rows: List[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_episode_indices(value: Optional[str]) -> Optional[set[int]]:
    if value is None:
        return None
    path = Path(value)
    if path.exists():
        return {int(line.strip()) for line in path.read_text().splitlines() if line.strip()}
    return {int(part) for part in value.split(",") if part.strip()}


def selected_episodes(args: argparse.Namespace) -> List[Mapping[str, Any]]:
    episodes = read_jsonl(args.data_root / "meta" / "episodes.jsonl")
    wanted = parse_episode_indices(args.episode_indices)
    out: List[Mapping[str, Any]] = []
    for episode in episodes:
        idx = int(episode["episode_index"])
        if wanted is not None and idx not in wanted:
            continue
        if args.episode_start is not None and idx < args.episode_start:
            continue
        if args.episode_end is not None and idx >= args.episode_end:
            continue
        out.append(episode)
    return out


def episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def video_path_for(data_root: Path, info: Mapping[str, Any], episode_index: int, video_key: str) -> Path:
    chunk = episode_chunk(episode_index, int(info["chunks_size"]))
    rel = str(info["video_path"]).format(
        episode_chunk=chunk,
        video_key=video_key,
        episode_index=episode_index,
    )
    return data_root / rel


def latent_path_for(output_dir: Path, info: Mapping[str, Any], episode_index: int) -> Path:
    chunk = episode_chunk(episode_index, int(info["chunks_size"]))
    return output_dir / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.safetensors"


def anchor_frames_for(length: int, share_stride: int) -> np.ndarray:
    return np.arange(0, length, share_stride, dtype=np.int64)


def safe_shape(path: Path, key: str) -> Optional[Tuple[int, ...]]:
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            return tuple(f.get_slice(key).get_shape())
    except Exception:
        return None


def existing_episode_ok(
    path: Path,
    *,
    expected_anchors: int,
    args: argparse.Namespace,
) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
            if metadata.get("complete") != "true":
                return False
            checks = {
                "share_stride": str(args.share_stride),
                "num_denoise_steps": str(args.num_denoise_steps),
                "layer": str(args.layer),
                "frame_num": str(args.frame_num),
                "pooling": "temporal_mean",
            }
            if args.confidence_aware:
                checks.update(
                    {
                        "confidence_aware": "true",
                        "confidence_version": "v1",
                        "num_teacher_samples": str(args.num_teacher_samples),
                        "target_aggregation": "mean",
                    }
                )
            for key, value in checks.items():
                if metadata.get(key) != value:
                    return False
            shape = tuple(f.get_slice("features").get_shape())
            anchors_shape = tuple(f.get_slice("anchor_frames").get_shape())
            if args.confidence_aware:
                confidence_shape = tuple(f.get_slice("confidence").get_shape())
                disagreement_shape = tuple(f.get_slice("disagreement").get_shape())
                if confidence_shape != anchors_shape or disagreement_shape != anchors_shape:
                    return False
    except Exception:
        return False
    return len(shape) == 3 and shape[0] == expected_anchors and anchors_shape == (expected_anchors,)


def decode_anchor_frames(video_path: Path, anchor_frames: Sequence[int]) -> Dict[int, Image.Image]:
    wanted = set(int(i) for i in anchor_frames)
    decoded: Dict[int, Image.Image] = {}
    try:
        for frame_index, frame_rgb in enumerate(iio.imiter(video_path)):
            if frame_index in wanted:
                decoded[frame_index] = Image.fromarray(np.asarray(frame_rgb, dtype=np.uint8))
                wanted.remove(frame_index)
                if not wanted:
                    return decoded
            if wanted and frame_index > max(wanted):
                break
    except Exception as imageio_error:
        logging.warning("imageio failed to decode %s: %s; falling back to OpenCV", video_path, imageio_error)

    if not wanted:
        return decoded

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    frame_index = 0
    try:
        while wanted and frame_index <= max(wanted):
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if frame_index in wanted:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                decoded[frame_index] = Image.fromarray(frame_rgb)
                wanted.remove(frame_index)
            frame_index += 1
    finally:
        capture.release()
    if wanted:
        missing = sorted(wanted)
        raise IndexError(f"Missing frames {missing[:5]} in {video_path}")
    return decoded


def rotation_matrix_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return rz_m @ ry_m @ rx_m


def deterministic_camera_trajectory(
    *,
    episode_index: int,
    frame_index: int,
    frame_num: int,
    base_seed: int,
    yaw_range: float,
    pitch_range: float,
    roll_range: float,
    translation_range: float,
) -> np.ndarray:
    seed = (base_seed + episode_index * 1_000_003 + frame_index * 9_176) % (2**32)
    rng = np.random.default_rng(seed)
    yaw = float(rng.uniform(-yaw_range, yaw_range))
    pitch = float(rng.uniform(-pitch_range, pitch_range))
    roll = float(rng.uniform(-roll_range, roll_range))
    translation = rng.uniform(-translation_range, translation_range, size=3).astype(np.float32)
    if float(np.linalg.norm(translation)) < 1e-6:
        translation[2] = translation_range

    poses = np.zeros((frame_num, 4, 4), dtype=np.float32)
    for i in range(frame_num):
        t = i / max(frame_num - 1, 1)
        alpha = t * t * (3.0 - 2.0 * t)
        poses[i, :3, :3] = rotation_matrix_xyz(roll * alpha, pitch * alpha, yaw * alpha)
        poses[i, :3, 3] = translation * alpha
        poses[i, 3, 3] = 1.0
    return poses


def get_text_context(pipe: wan.WanI2VFast, prompt: str) -> List[torch.Tensor]:
    cache_key = f"extract:{prompt}"
    if cache_key in pipe._t5_cache:
        return pipe._t5_cache[cache_key]
    if not pipe.t5_cpu:
        pipe.text_encoder.model.to(pipe.device)
        context = pipe.text_encoder([prompt], pipe.device)
    else:
        context = pipe.text_encoder([prompt], torch.device("cpu"))
        context = [t.to(pipe.device) for t in context]
    pipe._t5_cache[cache_key] = context
    return context


def model_config(model: torch.nn.Module) -> Any:
    return model.module.config if hasattr(model, "module") else model.config


def target_block(model: torch.nn.Module, layer: int) -> torch.nn.Module:
    module = model.module if hasattr(model, "module") else model
    return module.blocks[layer - 1]


def prepare_geometry(pipe: wan.WanI2VFast, img: Image.Image, args: argparse.Namespace) -> Dict[str, int]:
    cfg = pipe.config
    h, w = img.size[1], img.size[0]
    aspect_ratio = h / w
    max_area = MAX_AREA_CONFIGS[args.size]
    lat_h = round(
        np.sqrt(max_area * aspect_ratio)
        // cfg.vae_stride[1]
        // cfg.patch_size[1]
        * cfg.patch_size[1]
    )
    lat_w = round(
        np.sqrt(max_area / aspect_ratio)
        // cfg.vae_stride[2]
        // cfg.patch_size[2]
        * cfg.patch_size[2]
    )
    height = lat_h * cfg.vae_stride[1]
    width = lat_w * cfg.vae_stride[2]
    lat_f = (args.frame_num - 1) // cfg.vae_stride[0] + 1
    lat_f = int(lat_f - (lat_f % args.chunk_size))
    frame_num = (lat_f - 1) * cfg.vae_stride[0] + 1
    frame_seqlen = int(lat_h * lat_w // (cfg.patch_size[1] * cfg.patch_size[2]))
    max_seq_len = int(math.ceil((args.chunk_size * frame_seqlen) / pipe.sp_size) * pipe.sp_size)
    return {
        "height": height,
        "width": width,
        "lat_h": lat_h,
        "lat_w": lat_w,
        "lat_f": lat_f,
        "frame_num": frame_num,
        "frame_seqlen": frame_seqlen,
        "max_seq_len": max_seq_len,
        "spatial_tokens": int(frame_seqlen),
        "token_h": int(lat_h // cfg.patch_size[1]),
        "token_w": int(lat_w // cfg.patch_size[2]),
    }


class BlockFeatureCapture:
    def __init__(self, *, use_sp: bool, seq_len: int) -> None:
        self.use_sp = use_sp
        self.seq_len = seq_len
        self.enabled = False
        self.tensor: Optional[torch.Tensor] = None

    def hook(self, _module: torch.nn.Module, _inputs: Tuple[Any, ...], output: torch.Tensor) -> None:
        if not self.enabled:
            return
        x = output.detach()
        if self.use_sp and dist.is_initialized():
            gathered = [torch.empty_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, x.contiguous())
            x = torch.cat(gathered, dim=1)
        self.tensor = x[:, : self.seq_len].to(torch.bfloat16).cpu()


def _extract_anchor_feature_once(
    *,
    pipe: wan.WanI2VFast,
    context: List[torch.Tensor],
    img: Image.Image,
    episode_index: int,
    frame_index: int,
    args: argparse.Namespace,
    use_sp: bool,
    sample_index: int = 0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    geom = prepare_geometry(pipe, img, args)
    if geom["frame_num"] != args.frame_num:
        raise ValueError(f"frame_num adjusted from {args.frame_num} to {geom['frame_num']}")

    img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).to(pipe.device)
    height, width = geom["height"], geom["width"]
    lat_h, lat_w, lat_f = geom["lat_h"], geom["lat_w"], geom["lat_f"]

    seed = (
        args.base_seed
        + episode_index * 1_000_003
        + frame_index * 9_176
        + sample_index * 97_531
    ) % (2**31)
    generator = torch.Generator(device=pipe.device)
    generator.manual_seed(seed)
    noise = torch.randn(16, lat_f, lat_h, lat_w, dtype=torch.float32, generator=generator, device=pipe.device)

    msk = torch.ones(1, args.frame_num, lat_h, lat_w, device=pipe.device)
    msk[:, 1:] = 0
    msk = torch.concat(
        [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
        dim=1,
    )
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]

    poses = deterministic_camera_trajectory(
        episode_index=episode_index,
        frame_index=frame_index,
        frame_num=args.frame_num,
        base_seed=args.base_seed,
        yaw_range=args.yaw_range,
        pitch_range=args.pitch_range,
        roll_range=args.roll_range,
        translation_range=args.translation_range,
    )
    c2ws = poses[: args.frame_num]
    ks = torch.tensor([args.intrinsics], dtype=torch.float32).repeat(args.frame_num, 1)
    ks = get_Ks_transformed(
        ks,
        height_org=480,
        width_org=832,
        height_resize=height,
        width_resize=width,
        height_final=height,
        width_final=width,
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
    c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, ks, height, width, only_rays_d=False)
    c2ws_plucker_emb = rearrange(
        c2ws_plucker_emb,
        "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
        c1=int(height // lat_h),
        c2=int(width // lat_w),
    )
    c2ws_plucker_emb = c2ws_plucker_emb[None, ...]
    c2ws_plucker_emb = rearrange(
        c2ws_plucker_emb,
        "b (f h w) c -> b c f h w",
        f=lat_f,
        h=lat_h,
        w=lat_w,
    ).to(pipe.param_dtype)

    y = pipe.vae.encode(
        [
            torch.concat(
                [
                    torch.nn.functional.interpolate(
                        img_tensor[None].cpu(), size=(height, width), mode="bicubic"
                    ).transpose(0, 1),
                    torch.zeros(3, args.frame_num - 1, height, width),
                ],
                dim=1,
            ).to(pipe.device)
        ]
    )[0]
    y = torch.concat([msk, y])

    pipe.scheduler.set_timesteps(pipe.num_train_timesteps, shift=args.sample_shift)
    timesteps_index = FAST_TIMESTEP_INDEX[: args.num_denoise_steps]
    timesteps = pipe.scheduler.timesteps[timesteps_index]

    model_args = model_config(pipe.model)
    transformer_dtype = pipe.pipe_dtype
    frame_seqlen = int(noise.shape[-2] * noise.shape[-1] // 4)
    if pipe.local_attn_size > -1:
        kv_size = frame_seqlen * pipe.local_attn_size
    else:
        kv_size = frame_seqlen * lat_f
    head_dim = model_args.dim // model_args.num_heads
    local_num_heads = model_args.num_heads // pipe.sp_size
    self_kv_cache = pipe._initialize_self_kv_cache(
        num_layers=model_args.num_layers,
        shape=[1, kv_size, local_num_heads, head_dim],
        dtype=transformer_dtype,
        device=pipe.device,
    )
    cross_kv_cache = pipe._initialize_crossattn_cache(
        num_layers=model_args.num_layers,
        shape=[1, model_args.text_len, model_args.num_heads, head_dim],
        dtype=transformer_dtype,
        device=pipe.device,
    )

    capture = BlockFeatureCapture(use_sp=use_sp, seq_len=geom["frame_seqlen"] * args.chunk_size)
    handle = target_block(pipe.model, args.layer).register_forward_hook(capture.hook)

    @contextmanager
    def noop_no_sync() -> Iterable[None]:
        yield

    no_sync_model = getattr(pipe.model, "no_sync", noop_no_sync)
    pipe._cross_attn_initialized = False
    try:
        with torch.amp.autocast("cuda", dtype=pipe.param_dtype), torch.no_grad(), no_sync_model():
            current_latent = noise
            current_condition = y
            current_c2ws_plucker_emb = c2ws_plucker_emb
            dit_cond_dict = {"c2ws_plucker_emb": current_c2ws_plucker_emb.chunk(1, dim=0)}
            kwargs = {
                "context": [context[0]],
                "seq_len": geom["max_seq_len"],
                "y": [current_condition],
                "dit_cond_dict": dit_cond_dict,
                "kv_cache": self_kv_cache,
                "crossattn_cache": cross_kv_cache,
                "current_start": 0,
                "max_attention_size": kv_size if args.max_attention_size is None else args.max_attention_size,
                "frame_seqlen": geom["frame_seqlen"],
            }
            x0 = None
            for timestep_idx, timestep_value in enumerate(timesteps):
                timestep = torch.stack([timestep_value]).to(pipe.device)
                capture.enabled = timestep_idx == len(timesteps) - 1
                noise_pred = pipe.model(
                    x=[current_latent.to(pipe.device)],
                    t=timestep,
                    cross_attn_first_call=not pipe._cross_attn_initialized,
                    **kwargs,
                )[0]
                capture.enabled = False
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
                        torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype),
                        next_timestep,
                    )
            del x0
    finally:
        handle.remove()

    if capture.tensor is None:
        raise RuntimeError("Feature hook did not capture a tensor")

    hidden = capture.tensor[0]
    hidden = hidden.reshape(
        args.chunk_size,
        geom["token_h"],
        geom["token_w"],
        hidden.shape[-1],
    )
    pooled = hidden.mean(dim=0).reshape(geom["spatial_tokens"], hidden.shape[-1])
    meta = {
        "timesteps": [int(t.item()) for t in timesteps],
        "timesteps_index": timesteps_index,
        "sigmas": [float(pipe.scheduler.sigmas[i].item()) for i in timesteps_index],
        "height": geom["height"],
        "width": geom["width"],
        "lat_f": geom["lat_f"],
        "lat_h": geom["lat_h"],
        "lat_w": geom["lat_w"],
        "token_h": geom["token_h"],
        "token_w": geom["token_w"],
        "spatial_tokens": geom["spatial_tokens"],
    }
    del capture.tensor, noise, y, c2ws_plucker_emb
    torch.cuda.empty_cache()
    return pooled.contiguous(), meta


def _pairwise_disagreement(samples: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return one raw disagreement scalar per anchor from CPU feature samples."""
    if len(samples) < 2:
        return torch.zeros((), dtype=torch.float32)
    normalized = [
        torch.nn.functional.normalize(
            torch.nn.functional.layer_norm(sample.float(), (sample.shape[-1],)), dim=-1
        )
        for sample in samples
    ]
    stacked = torch.stack(normalized, dim=0)
    cosine_sum = torch.zeros(stacked.shape[1], dtype=torch.float32)
    pair_count = 0
    for i in range(stacked.shape[0]):
        for j in range(i + 1, stacked.shape[0]):
            cosine_sum += (stacked[i] * stacked[j]).sum(dim=-1)
            pair_count += 1
    disagreement = (1.0 - cosine_sum / float(pair_count)).clamp(0.0, 2.0)
    return disagreement.median()


def extract_anchor_feature(
    *,
    pipe: wan.WanI2VFast,
    context: List[torch.Tensor],
    img: Image.Image,
    episode_index: int,
    frame_index: int,
    args: argparse.Namespace,
    use_sp: bool,
) -> Tuple[torch.Tensor, Dict[str, Any], float]:
    """Extract one anchor, optionally averaging multiple fixed-trajectory samples."""
    count = args.num_teacher_samples if args.confidence_aware else 1
    samples: List[torch.Tensor] = []
    meta: Optional[Dict[str, Any]] = None
    for sample_index in range(count):
        feature, current_meta = _extract_anchor_feature_once(
            pipe=pipe,
            context=context,
            img=img,
            episode_index=episode_index,
            frame_index=frame_index,
            args=args,
            use_sp=use_sp,
            sample_index=sample_index,
        )
        if meta is None:
            meta = current_meta
        if samples and tuple(feature.shape) != tuple(samples[0].shape):
            raise ValueError("Teacher samples have inconsistent feature shapes")
        if not torch.isfinite(feature.float()).all():
            raise ValueError("Teacher feature contains non-finite values")
        samples.append(feature.float().cpu())
    assert meta is not None
    disagreement = float(_pairwise_disagreement(samples).item())
    mean_feature = torch.stack(samples, dim=0).mean(dim=0).to(torch.bfloat16)
    meta = dict(meta)
    meta["sample_seeds"] = [
        int(
            (
                args.base_seed
                + episode_index * 1_000_003
                + frame_index * 9_176
                + sample_index * 97_531
            )
            % (2**31)
        )
        for sample_index in range(count)
    ]
    del samples
    return mean_feature.contiguous(), meta, disagreement


def save_episode(
    *,
    path: Path,
    features: torch.Tensor,
    anchor_frames: np.ndarray,
    episode: Mapping[str, Any],
    args: argparse.Namespace,
    extract_meta: Mapping[str, Any],
    disagreement: Optional[torch.Tensor] = None,
    confidence: Optional[torch.Tensor] = None,
    complete: bool = True,
    calibration: Optional[Mapping[str, float]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    tensors = {
        "features": features.contiguous(),
        "anchor_frames": torch.from_numpy(anchor_frames.astype(np.int64)),
    }
    if args.confidence_aware:
        if disagreement is None or confidence is None:
            raise ValueError("Confidence-aware cache requires disagreement and confidence tensors")
        expected_shape = (features.shape[0],)
        if tuple(disagreement.shape) != expected_shape or tuple(confidence.shape) != expected_shape:
            raise ValueError("Confidence/disagreement must contain one scalar per anchor")
        if not torch.isfinite(disagreement).all() or not torch.isfinite(confidence).all():
            raise ValueError("Confidence-aware statistics contain non-finite values")
        tensors["confidence"] = confidence.float().contiguous()
        tensors["disagreement"] = disagreement.float().contiguous()
    metadata = {
        "complete": str(complete).lower(),
        "episode_index": str(int(episode["episode_index"])),
        "episode_length": str(int(episode["length"])),
        "share_stride": str(args.share_stride),
        "num_denoise_steps": str(args.num_denoise_steps),
        "layer": str(args.layer),
        "frame_num": str(args.frame_num),
        "chunk_size": str(args.chunk_size),
        "sample_shift": str(args.sample_shift),
        "video_key": args.video_key,
        "pooling": "temporal_mean",
        "prompt": args.prompt,
        "timesteps": json.dumps(extract_meta["timesteps"]),
        "timesteps_index": json.dumps(extract_meta["timesteps_index"]),
        "sigmas": json.dumps(extract_meta["sigmas"]),
        "height": str(extract_meta["height"]),
        "width": str(extract_meta["width"]),
        "lat_f": str(extract_meta["lat_f"]),
        "lat_h": str(extract_meta["lat_h"]),
        "lat_w": str(extract_meta["lat_w"]),
        "token_h": str(extract_meta["token_h"]),
        "token_w": str(extract_meta["token_w"]),
        "spatial_tokens": str(extract_meta["spatial_tokens"]),
        "feature_dim": str(features.shape[-1]),
        "dtype": str(features.dtype).replace("torch.", ""),
    }
    if args.confidence_aware:
        metadata.update(
            {
                "confidence_aware": "true",
                "confidence_version": "v1",
                "num_teacher_samples": str(args.num_teacher_samples),
                "target_aggregation": "mean",
                "uncertainty_metric": "pairwise_cosine_disagreement",
                "feature_normalization": "layernorm_last_dim",
                "uncertainty_aggregation": "median_token",
                "seed_policy": "explicit_per_anchor_fixed_trajectory",
                "sample_seeds": json.dumps(extract_meta.get("sample_seeds", [])),
                "calibration": "pending" if calibration is None else "train_quantile_05_95",
            }
        )
        if calibration is not None:
            metadata.update(
                {
                    "calibration_q_low": f"{calibration['q_low']:.9g}",
                    "calibration_q_high": f"{calibration['q_high']:.9g}",
                    "calibration_eps": f"{calibration['eps']:.9g}",
                }
            )
    save_file(tensors, str(tmp_path), metadata=metadata)
    os.replace(tmp_path, path)


def finalize_confidence_cache(
    *,
    paths: Sequence[Path],
    args: argparse.Namespace,
    calibration_manifest: Path,
) -> None:
    """Calibrate raw per-anchor disagreement and atomically finalize episode files."""
    disagreements: List[torch.Tensor] = []
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as reader:
            disagreements.append(reader.get_tensor("disagreement").float())
    raw = torch.cat(disagreements)
    if raw.numel() == 0 or not torch.isfinite(raw).all():
        raise ValueError("Cannot calibrate empty or non-finite disagreement statistics")
    q_low = float(torch.quantile(raw, 0.05).item())
    q_high = float(torch.quantile(raw, 0.95).item())
    calibration = {"q_low": q_low, "q_high": q_high, "eps": 1e-8}

    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as reader:
            features = reader.get_tensor("features")
            anchors = reader.get_tensor("anchor_frames")
            disagreement = reader.get_tensor("disagreement").float()
            metadata = dict(reader.metadata() or {})
        confidence = 1.0 - ((disagreement - q_low) / (q_high - q_low + calibration["eps"]))
        confidence = confidence.clamp_(0.0, 1.0)
        tensors = {
            "features": features.contiguous(),
            "anchor_frames": anchors.contiguous(),
            "confidence": confidence.float().contiguous(),
            "disagreement": disagreement.contiguous(),
        }
        metadata.update(
            {
                "complete": "true",
                "calibration": "train_quantile_05_95",
                "calibration_q_low": f"{q_low:.9g}",
                "calibration_q_high": f"{q_high:.9g}",
                "calibration_eps": f"{calibration['eps']:.9g}",
            }
        )
        tmp_path = path.with_name(path.name + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        save_file(tensors, str(tmp_path), metadata=metadata)
        os.replace(tmp_path, path)

    calibration_manifest.write_text(
        json.dumps(
            {
                "confidence_aware": True,
                "confidence_version": "v1",
                "num_teacher_samples": args.num_teacher_samples,
                "calibration": "train_quantile_05_95",
                "q_low": q_low,
                "q_high": q_high,
                "eps": calibration["eps"],
                "episode_count": len(paths),
                "anchor_count": int(raw.numel()),
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = init_distributed()
    cfg = WAN_CONFIGS[args.task]
    validate_args(args, cfg)
    output_dir = args.output_dir if args.output_dir is not None else default_output_dir(args)
    validate_output_dir_name(output_dir, args.share_stride)

    info = json.load((args.data_root / "meta" / "info.json").open("r", encoding="utf-8"))
    episodes = selected_episodes(args)
    if not episodes:
        raise RuntimeError("No episodes selected")

    if rank == 0:
        logging.info(
            "Extracting LingBot latents: episodes=%d output=%s step=%d layer=%d stride=%d world_size=%d",
            len(episodes),
            output_dir,
            args.num_denoise_steps,
            args.layer,
            args.share_stride,
            world_size,
        )

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
    use_sp = args.ulysses_size > 1 and world_size > 1
    context = get_text_context(pipe, args.prompt)

    if not args.no_prewarm:
        first_episode = episodes[0]
        first_video = video_path_for(args.data_root, info, int(first_episode["episode_index"]), args.video_key)
        first_frame = decode_anchor_frames(first_video, [0])[0]
        pipe.prewarm(
            first_frame,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            chunk_size=args.chunk_size,
        )

    for episode_number, episode in enumerate(episodes, start=1):
        episode_index = int(episode["episode_index"])
        episode_length = int(episode["length"])
        anchor_frames = anchor_frames_for(episode_length, args.share_stride)
        out_path = latent_path_for(output_dir, info, episode_index)

        should_skip = False
        if rank == 0 and not args.overwrite:
            should_skip = existing_episode_ok(
                out_path,
                expected_anchors=len(anchor_frames),
                args=args,
            )
            if should_skip and (episode_number % args.log_every == 0):
                logging.info("skip episode=%06d existing=%s", episode_index, out_path)
        should_skip = bool(broadcast_object(should_skip))
        if should_skip:
            continue

        video_path = video_path_for(args.data_root, info, episode_index, args.video_key)
        if rank == 0:
            logging.info(
                "episode %d/%d index=%06d length=%d anchors=%d -> %s",
                episode_number,
                len(episodes),
                episode_index,
                episode_length,
                len(anchor_frames),
                out_path,
            )

        frames = decode_anchor_frames(video_path, anchor_frames)
        features: List[torch.Tensor] = []
        disagreements: List[float] = []
        extract_meta: Optional[Dict[str, Any]] = None
        iterator = tqdm(anchor_frames, disable=(rank != 0), desc=f"episode_{episode_index:06d}")
        t0 = time.perf_counter()
        for frame_index in iterator:
            feature, meta, raw_disagreement = extract_anchor_feature(
                pipe=pipe,
                context=context,
                img=frames[int(frame_index)],
                episode_index=episode_index,
                frame_index=int(frame_index),
                args=args,
                use_sp=use_sp,
            )
            if rank == 0:
                features.append(feature)
                disagreements.append(raw_disagreement)
                if extract_meta is None:
                    extract_meta = meta

        if rank == 0:
            if not features or extract_meta is None:
                raise RuntimeError(f"No features extracted for episode {episode_index}")
            stacked = torch.stack(features, dim=0).contiguous()
            disagreement_tensor = torch.tensor(disagreements, dtype=torch.float32)
            confidence_tensor = torch.ones_like(disagreement_tensor)
            save_episode(
                path=out_path,
                features=stacked,
                anchor_frames=anchor_frames,
                episode=episode,
                args=args,
                extract_meta=extract_meta,
                disagreement=disagreement_tensor if args.confidence_aware else None,
                confidence=confidence_tensor if args.confidence_aware else None,
                complete=not args.confidence_aware,
            )
            elapsed = time.perf_counter() - t0
            logging.info(
                "episode=%06d saved shape=%s elapsed=%.1fs",
                episode_index,
                tuple(stacked.shape),
                elapsed,
            )
            del stacked, disagreement_tensor, confidence_tensor
        del frames, features
        if dist.is_initialized():
            dist.barrier()

    if args.confidence_aware and rank == 0:
        paths = [latent_path_for(output_dir, info, int(episode["episode_index"])) for episode in episodes]
        missing = [path for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Cannot finalize confidence cache; missing files: {missing[:3]}")
        finalize_confidence_cache(
            paths=paths,
            args=args,
            calibration_manifest=output_dir / "confidence_manifest.json",
        )
        logging.info("Finalized confidence-aware cache: q05/q95 computed over %d anchors", sum(
            int(safe_shape(path, "confidence")[0]) for path in paths
        ))

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
