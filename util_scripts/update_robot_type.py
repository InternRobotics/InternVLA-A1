import os
from pathlib import Path
from lerobot.datasets.utils import load_json, write_json

def update_robot_type(base_dir: str, new_robot_type: str = "arx_lift2"):
    """
    Recursively scan all subdirectories under base_dir.
    For each directory containing meta/info.json, load it,
    set info["robot_type"] = new_robot_type, and save it back.

    Args:
        base_dir (str): The root directory containing lift2_sim_* folders.
        new_robot_type (str): The new robot type to assign.
    """
    base_dir = Path(base_dir)

    # Walk through all subfolders
    for root, dirs, files in os.walk(base_dir):
        root_path = Path(root)

        # Check if this folder contains meta/info.json
        info_path = root_path / "meta" / "info.json"
        if info_path.exists():
            print(f"[INFO] Updating robot_type in: {info_path}")

            try:
                info = load_json(info_path)
            except Exception as e:
                print(f"[ERROR] Failed to load JSON: {info_path} ({e})")
                continue

            # Update value
            old = info.get("robot_type", None)
            info["robot_type"] = new_robot_type

            try:
                write_json(info, info_path)
                print(f"       robot_type: {old} → {new_robot_type}")
            except Exception as e:
                print(f"[ERROR] Failed to write JSON: {info_path} ({e})")


if __name__ == "__main__":
    # Example: assuming the current working directory contains lift2_sim_*
    update_robot_type("data/a1/basic_tasks/genie1", new_robot_type="genie1")
    update_robot_type("data/a1/pick_and_place_tasks/genie1", new_robot_type="genie1")
    # update_robot_type("data/a1/pick_and_place_tasks/split_aloha", new_robot_type="piper")
    # update_robot_type("data/a1/long_horizon_tasks/split_aloha", new_robot_type="piper")
