#!/usr/bin/env python3
"""Finalize RoboTwin confidence tensors without initializing the teacher model.

The extractor writes raw disagreement and a confidence=1 placeholder.  This
utility reads those CPU tensors, calibrates each (domain, task_id) group, and
atomically rewrites the episode safetensors files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file


EPS = 1e-8
CALIBRATION = "train_quantile_05_95"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-teacher-samples", type=int, default=4)
    return parser.parse_args()


def read_manifest(path: Path) -> List[Mapping[str, Any]]:
    rows: List[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Manifest row {line_number} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def group_key(episode: Mapping[str, Any]) -> Tuple[str, str]:
    domain = str(episode.get("domain", "")).strip().lower()
    task_id = str(episode.get("task_id", "")).strip()
    if not domain or not task_id:
        raise ValueError("Every episode needs non-empty domain and task_id")
    return domain, task_id


def cache_path(episode: Mapping[str, Any], output_dir: Path) -> Path:
    try:
        dataset_root = Path(str(episode["dataset_root"]))
        index = int(episode["episode_index"])
        chunk_size = int(episode.get("chunks_size", 1000))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid cache location fields: {episode}") from exc
    if chunk_size <= 0:
        raise ValueError(f"chunks_size must be positive for episode {index}")
    return (
        dataset_root
        / output_dir.name
        / f"chunk-{index // chunk_size:03d}"
        / f"episode_{index:06d}.safetensors"
    )


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def safe_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip(".")
    return component or "_"


def group_manifest_payload(
    *,
    domain: str,
    task_id: str,
    episode_count: int,
    anchor_count: int,
    q_low: float,
    q_high: float,
    num_teacher_samples: int,
) -> Dict[str, Any]:
    key = f"{domain}/{task_id}"
    return {
        "confidence_aware": True,
        "confidence_version": "v1",
        "num_teacher_samples": num_teacher_samples,
        "calibration": CALIBRATION,
        "calibration_scope": "task_training_episodes",
        "groups": {
            key: {
                "domain": domain,
                "task_id": task_id,
                "episode_count": episode_count,
                "anchor_count": anchor_count,
                "q_low": q_low,
                "q_high": q_high,
                "eps": EPS,
            }
        },
    }


def finalize_group(
    *,
    domain: str,
    task_id: str,
    entries: Sequence[Tuple[Mapping[str, Any], Path]],
    num_teacher_samples: int,
) -> Dict[str, Any]:
    disagreements: List[torch.Tensor] = []
    for _episode, path in entries:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing cache for {domain}/{task_id}: {path}")
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            if "disagreement" not in reader.keys():
                raise ValueError(f"Missing disagreement tensor: {path}")
            value = reader.get_tensor("disagreement").float()
        if value.ndim != 1 or value.numel() == 0 or not torch.isfinite(value).all():
            raise ValueError(f"Invalid disagreement tensor: {path}")
        disagreements.append(value)

    raw = torch.cat(disagreements)
    q_low = float(torch.quantile(raw, 0.05).item())
    q_high = float(torch.quantile(raw, 0.95).item())
    if not q_high > q_low:
        raise ValueError(f"Degenerate disagreement quantiles for {domain}/{task_id}: {q_low} >= {q_high}")

    for _episode, path in entries:
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            tensors = {key: reader.get_tensor(key).contiguous() for key in reader.keys()}
            metadata = dict(reader.metadata() or {})
        disagreement = tensors["disagreement"].float()
        tensors["confidence"] = (
            1.0 - (disagreement - q_low) / (q_high - q_low + EPS)
        ).clamp(0.0, 1.0).float().contiguous()
        metadata.update(
            {
                "extraction_complete": "true",
                "complete": "true",
                "calibration": CALIBRATION,
                "calibration_scope": "task_training_episodes",
                "calibration_domain": domain,
                "calibration_task_id": task_id,
                "calibration_q_low": f"{q_low:.9g}",
                "calibration_q_high": f"{q_high:.9g}",
                "calibration_eps": f"{EPS:.9g}",
            }
        )
        tmp = path.with_name(path.name + ".tmp")
        if tmp.exists():
            tmp.unlink()
        save_file(tensors, str(tmp), metadata=metadata)
        os.replace(tmp, path)

    return group_manifest_payload(
        domain=domain,
        task_id=task_id,
        episode_count=len(entries),
        anchor_count=int(raw.numel()),
        q_low=q_low,
        q_high=q_high,
        num_teacher_samples=num_teacher_samples,
    )


def main() -> None:
    args = parse_args()
    if args.num_teacher_samples < 2:
        raise ValueError("--num-teacher-samples must be at least 2")
    episodes = read_manifest(args.manifest)
    groups: Dict[Tuple[str, str], List[Tuple[Mapping[str, Any], Path]]] = {}
    for episode in episodes:
        groups.setdefault(group_key(episode), []).append((episode, cache_path(episode, args.output_dir)))

    all_groups: Dict[str, Dict[str, Any]] = {}
    for (domain, task_id), entries in sorted(groups.items()):
        payload = finalize_group(
            domain=domain,
            task_id=task_id,
            entries=entries,
            num_teacher_samples=args.num_teacher_samples,
        )
        key = f"{domain}/{task_id}"
        all_groups[key] = payload["groups"][key]

        # Keep a discoverable per-group manifest without colliding when the
        # dataset stores multiple tasks under one shared cache root.
        cache_roots = sorted({path.parent.parent for _episode, path in entries})
        for cache_root in cache_roots:
            group_path = (
                cache_root
                / "confidence_groups"
                / safe_component(domain)
                / safe_component(task_id)
                / "confidence_manifest.json"
            )
            atomic_json(group_path, payload)

    total_payload = {
        "confidence_aware": True,
        "confidence_version": "v1",
        "num_teacher_samples": args.num_teacher_samples,
        "calibration": CALIBRATION,
        "calibration_scope": "task_training_episodes",
        "groups": all_groups,
    }
    atomic_json(args.output_dir / "confidence_manifest.json", total_payload)
    print(
        f"Finalized {len(episodes)} episodes, {len(groups)} domain/task groups; "
        f"wrote {args.output_dir / 'confidence_manifest.json'}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
