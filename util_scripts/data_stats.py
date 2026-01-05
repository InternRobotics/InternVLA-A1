#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional
import csv

def is_lerobot_dataset_dir(d: Path) -> bool:
    """A lerobot dataset dir is identified by having meta/info.json."""
    return (d / "meta" / "info.json").is_file()

def iter_dataset_dirs(root: Path):
    """Walk recursively under root, yielding all dirs that look like lerobot datasets."""
    # Fast path: many datasets are 3~5 levels deep; os.walk handles all
    for p in root.rglob("meta/info.json"):
        yield p.parent.parent  # dataset dir is meta's parent

def load_info(info_path: Path) -> Tuple[int, int, Optional[str]]:
    """Return (total_episodes, total_frames, robot_type) from meta/info.json."""
    try:
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
        total_episodes = int(info.get("total_episodes", 0))
        total_frames = int(info.get("total_frames", 0))
        robot_type = info.get("robot_type")
        return total_episodes, total_frames, robot_type
    except Exception as e:
        print(f"[WARN] Failed to read {info_path}: {e}", file=sys.stderr)
        return 0, 0, None

def load_tasks(task_jsonl_path: Path) -> Set[str]:
    """Read tasks from meta/task.jsonl; each line is a JSON object with 'task'."""
    tasks: Set[str] = set()
    if not task_jsonl_path.is_file():
        return tasks
    try:
        with task_jsonl_path.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    t = obj.get("task")
                    if isinstance(t, str) and t.strip():
                        tasks.add(t.strip())
                except Exception as ie:
                    print(f"[WARN] Bad JSON at {task_jsonl_path}:{ln}: {ie}", file=sys.stderr)
    except Exception as e:
        print(f"[WARN] Failed to read {task_jsonl_path}: {e}", file=sys.stderr)
    return tasks

def make_row(dataset_dir: Path,
             total_episodes: int,
             total_frames: int,
             robot_type: Optional[str],
             tasks: Set[str]) -> Dict[str, Any]:
    # Derive some handy columns for reporting
    # e.g., robot = the first directory under root (if present)
    parts = dataset_dir.parts
    row = {
        "dataset_path": str(dataset_dir),
        "robot": None,
        "dataset_name": dataset_dir.name,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "robot_type": robot_type or "",
        "num_tasks_in_file": len(tasks),
        "tasks_in_file": "|".join(sorted(tasks)) if tasks else "",
    }
    # Try to infer 'robot' as the first child of root by walking up until we see root later in main
    # We'll fill 'robot' later once we know the root prefix length.
    return row

def write_csv(rows: List[Dict[str, Any]], out_csv: Path):
    if not rows:
        print("[INFO] No rows to write for CSV.")
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "robot", "dataset_name", "dataset_path",
        "robot_type", "total_episodes", "total_frames",
        "num_tasks_in_file", "tasks_in_file",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"[OK] CSV written -> {out_csv}")

def write_json(summary: Dict[str, Any], out_json: Path):
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] JSON written -> {out_json}")

def main():
    ap = argparse.ArgumentParser(description="Aggregate stats for lerobot datasets under a root directory.")
    ap.add_argument("--root", type=str, required=True, help="Root directory containing robot folders (e.g., franka, etc.)")
    ap.add_argument("--out-csv", type=str, default=None, help="Optional path to save per-dataset stats as CSV")
    ap.add_argument("--out-json", type=str, default=None, help="Optional path to save overall summary JSON")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"[ERR] root does not exist or is not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    rows: List[Dict[str, Any]] = []
    unique_tasks: Set[str] = set()
    total_episodes_all = 0
    total_frames_all = 0

    # Precompute root depth to infer 'robot' column as child of root
    root_depth = len(root.parts)

    dataset_count = 0
    for ds_dir in iter_dataset_dirs(root):
        dataset_count += 1
        info_path = ds_dir / "meta" / "info.json"
        task_path = ds_dir / "meta" / "tasks.jsonl"

        total_episodes, total_frames, robot_type = load_info(info_path)
        tasks = load_tasks(task_path)
        unique_tasks.update(tasks)

        row = make_row(ds_dir, total_episodes, total_frames, robot_type, tasks)

        # Fill 'robot' as the first-level subfolder under root if present
        # e.g., /root/franka/Close_v_... -> robot=franka
        if len(ds_dir.parts) > root_depth:
            row["robot"] = ds_dir.parts[root_depth]
        else:
            row["robot"] = ""

        rows.append(row)
        total_episodes_all += total_episodes
        total_frames_all += total_frames

    # Print concise summary to stdout
    print("========== lerobot dataset summary ==========")
    print(f"Root: {root}")
    print(f"Datasets found: {dataset_count}")
    print(f"Total episodes: {total_episodes_all}")
    print(f"Total frames:   {total_frames_all}")
    print(f"Unique tasks:   {len(unique_tasks)}")
    # if unique_tasks:
    #     for t in sorted(unique_tasks):
    #         print(f"  - {t}")

    # Optional outputs
    if args.out_csv:
        write_csv(rows, Path(args.out_csv))

    if args.out_json:
        summary = {
            "root": str(root),
            "datasets_found": dataset_count,
            "total_episodes": total_episodes_all,
            "total_frames": total_frames_all,
            "num_unique_tasks": len(unique_tasks),
            "unique_tasks": sorted(unique_tasks),
            "per_dataset": rows,
        }
        write_json(summary, Path(args.out_json))

if __name__ == "__main__":
    main()
