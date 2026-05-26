import os
from typing import Any, Dict, Optional


def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def resolve_ground_truth_path(image_path: str, ground_truth_arg: Optional[str]) -> Optional[str]:
    if not ground_truth_arg:
        return None
    if os.path.isfile(ground_truth_arg):
        return ground_truth_arg
    if os.path.isdir(ground_truth_arg):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        candidate = os.path.join(ground_truth_arg, f"{base_name}.txt")
        if os.path.isfile(candidate):
            return candidate
    return None