#!/usr/bin/env python3
"""Add LingBot latent references to LIBERO episodes_train.jsonl."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create meta/episodes_train_lingbot.jsonl with LingBot latent indices."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data/jibaixu/Datasets/LIBERO/libero_lerobot_v2.1"),
    )
    parser.add_argument(
        "--latent-dir",
        required=True,
        help="Latent directory name relative to data root, or an absolute path.",
    )
    parser.add_argument("--input", type=Path, default=Path("meta/episodes_train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("meta/episodes_train_lingbot.jsonl"))
    parser.add_argument("--share-stride", type=int, default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--num-denoise-steps", type=int, default=None)
    parser.add_argument("--no-strict", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--require-confidence",
        action="store_true",
        help="Require confidence-aware teacher tensors and finalized calibration metadata.",
    )
    return parser.parse_args()


def resolve_under_root(data_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else data_root / path


def relative_to_root(data_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(data_root))
    except ValueError:
        return str(path)


def read_jsonl(path: Path) -> List[Mapping[str, Any]]:
    rows: List[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def latent_path_for(
    *,
    data_root: Path,
    latent_dir: Path,
    chunks_size: int,
    episode_index: int,
) -> Path:
    chunk = episode_chunk(episode_index, chunks_size)
    return latent_dir / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.safetensors"


def infer_from_name(latent_dir: Path) -> Dict[str, int]:
    name = latent_dir.name
    out: Dict[str, int] = {}
    match = re.search(r"_step(\d+)", name)
    if match:
        out["num_denoise_steps"] = int(match.group(1))
    match = re.search(r"_numlayer(\d+)", name)
    if match:
        out["layer"] = int(match.group(1))
    match = re.search(r"_stride(\d+)", name)
    if match:
        out["share_stride"] = int(match.group(1))
    return out


def tensor_shape(path: Path, key: str) -> Tuple[int, ...]:
    with safe_open(path, framework="pt", device="cpu") as f:
        return tuple(f.get_slice(key).get_shape())


def metadata_for(path: Path) -> Dict[str, str]:
    with safe_open(path, framework="pt", device="cpu") as f:
        return dict(f.metadata() or {})


def expected_anchor_count(length: int, share_stride: int) -> int:
    return (length + share_stride - 1) // share_stride


def validate_episode_file(
    *,
    path: Path,
    episode_index: int,
    episode_length: int,
    share_stride: int,
    layer: Optional[int],
    num_denoise_steps: Optional[int],
    strict: bool,
    require_confidence: bool = False,
) -> Dict[str, Any]:
    if not path.exists():
        if strict:
            raise FileNotFoundError(f"Missing latent file for episode {episode_index}: {path}")
        return {}
    metadata = metadata_for(path)
    shape = tensor_shape(path, "features")
    anchors_shape = tensor_shape(path, "anchor_frames")
    expected_anchors = expected_anchor_count(episode_length, share_stride)
    if len(shape) != 3:
        raise ValueError(f"{path} features must be rank 3, got shape={shape}")
    if shape[0] != expected_anchors:
        raise ValueError(
            f"{path} anchor count mismatch: got {shape[0]}, expected {expected_anchors}"
        )
    if anchors_shape != (expected_anchors,):
        raise ValueError(
            f"{path} anchor_frames mismatch: got {anchors_shape}, expected {(expected_anchors,)}"
        )
    checks = {
        "episode_index": str(episode_index),
        "episode_length": str(episode_length),
        "share_stride": str(share_stride),
    }
    if layer is not None:
        checks["layer"] = str(layer)
    if num_denoise_steps is not None:
        checks["num_denoise_steps"] = str(num_denoise_steps)
    for key, expected in checks.items():
        actual = metadata.get(key)
        if actual != expected:
            raise ValueError(f"{path} metadata {key}={actual!r}, expected {expected!r}")
    if require_confidence:
        metadata_checks = {
            "confidence_aware": "true",
            "confidence_version": "v1",
            "calibration": "train_quantile_05_95",
        }
        for key, expected in metadata_checks.items():
            if metadata.get(key) != expected:
                raise ValueError(f"{path} metadata {key}={metadata.get(key)!r}, expected {expected!r}")
        confidence_shape = tensor_shape(path, "confidence")
        disagreement_shape = tensor_shape(path, "disagreement")
        expected_shape = (expected_anchors,)
        if confidence_shape != expected_shape or disagreement_shape != expected_shape:
            raise ValueError(
                f"{path} confidence/disagreement shape mismatch: "
                f"confidence={confidence_shape}, disagreement={disagreement_shape}, expected={expected_shape}"
            )
    return {"metadata": metadata, "shape": shape}


def infer_missing_settings(
    *,
    args: argparse.Namespace,
    latent_dir: Path,
) -> Tuple[int, Optional[int], Optional[int]]:
    inferred = infer_from_name(latent_dir)
    share_stride = args.share_stride or inferred.get("share_stride")
    layer = args.layer or inferred.get("layer")
    num_denoise_steps = args.num_denoise_steps or inferred.get("num_denoise_steps")
    if share_stride is None:
        raise ValueError(
            "Could not infer share stride from --share-stride or latent directory "
            "name like lingbot_latent_step2_numlayer24_stride16."
        )
    return share_stride, layer, num_denoise_steps


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    latent_dir = Path(args.latent_dir)
    latent_dir = resolve_under_root(data_root, latent_dir)
    input_path = resolve_under_root(data_root, args.input)
    output_path = resolve_under_root(data_root, args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Pass --overwrite to replace it.")

    info = json.load((data_root / "meta" / "info.json").open("r", encoding="utf-8"))
    chunks_size = int(info["chunks_size"])
    episodes = {
        int(row["episode_index"]): int(row["length"])
        for row in read_jsonl(data_root / "meta" / "episodes.jsonl")
    }
    train_rows = read_jsonl(input_path)

    share_stride, layer, num_denoise_steps = infer_missing_settings(
        args=args,
        latent_dir=latent_dir,
    )
    strict = not args.no_strict

    validated: Dict[int, Path] = {}
    if strict:
        for episode_index, episode_length in episodes.items():
            path = latent_path_for(
                data_root=data_root,
                latent_dir=latent_dir,
                chunks_size=chunks_size,
                episode_index=episode_index,
            )
            validate_episode_file(
                path=path,
                episode_index=episode_index,
                episode_length=episode_length,
                share_stride=share_stride,
                layer=layer,
                num_denoise_steps=num_denoise_steps,
                strict=True,
                require_confidence=args.require_confidence,
            )
            validated[episode_index] = path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with tmp_path.open("w", encoding="utf-8") as out:
        for row in train_rows:
            episode_index = int(row["episode_index"])
            start_frame = int(row["start_frame"])
            latent_path = validated.get(episode_index)
            if latent_path is None:
                latent_path = latent_path_for(
                    data_root=data_root,
                    latent_dir=latent_dir,
                    chunks_size=chunks_size,
                    episode_index=episode_index,
                )
                if strict:
                    raise FileNotFoundError(f"Latent file not validated: {latent_path}")
            latent_index = start_frame // share_stride
            enriched = dict(row)
            enriched.update(
                {
                    "lingbot_latent": relative_to_root(data_root, latent_path),
                    "lingbot_latent_index": latent_index,
                }
            )
            out.write(json.dumps(enriched, ensure_ascii=False) + "\n")
    os.replace(tmp_path, output_path)
    print(
        f"wrote {output_path} rows={len(train_rows)} "
        f"latent_dir={relative_to_root(data_root, latent_dir)} "
        f"share_stride={share_stride} layer={layer} steps={num_denoise_steps}"
    )


if __name__ == "__main__":
    main()
