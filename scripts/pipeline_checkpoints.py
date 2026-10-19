"""Helpers for SmartLink pipeline checkpoints."""

import json
from pathlib import Path
from typing import Any


def write_checkpoint(
    checkpoint_path: Path,
    metadata: dict[str, Any],
) -> None:
    """Write a successful pipeline checkpoint."""

    checkpoint_path = Path(checkpoint_path)

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def checkpoint_exists(
    checkpoint_path: Path,
) -> bool:
    """Return True when a checkpoint exists."""

    return Path(checkpoint_path).is_file()