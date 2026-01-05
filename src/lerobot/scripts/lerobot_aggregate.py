import argparse
import shutil

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.utils.constants import HF_LEROBOT_HOME

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-ids",
        type=str,
        nargs='+',
        required=True,
        help="List of repository IDs for the datasets to aggregate.",
    )
    parser.add_argument(
        "--aggr-repo-id",
        type=str,
        required=True,
        help="Repository ID for the aggregated output dataset.",
    )

    args = parser.parse_args()
    if (HF_LEROBOT_HOME / args.aggr_repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / args.aggr_repo_id)
    aggregate_datasets(
        repo_ids=args.repo_ids, 
        aggr_repo_id=args.aggr_repo_id, 
        # video_files_size_in_mb=50000, 
    )

if __name__ == "__main__":
    main()
