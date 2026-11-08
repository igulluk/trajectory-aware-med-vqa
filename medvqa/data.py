"""Loading of the reasoning-augmented medical VQA datasets.

Each JSON file is a list of records::

    {
      "question": "Is there a skull fracture?",
      "gt_answer": "no",
      "reasoning_answer": "<step 1>.\\n<step 2>.\\n...\\nThe final answer is: ...",
      "image": "./VQA-RAD/VQA_RAD Image Folder/synpic57237.jpg",
      "image_path": "./VQA-RAD/VQA_RAD Image Folder/synpic57237.jpg",
      "split": "train" | "test"
    }
"""

from __future__ import annotations

import os
from typing import Optional

from datasets import DatasetDict, load_dataset


def load_splits(data_json_path: str, image_root: Optional[str] = None) -> DatasetDict:
    """Load a dataset JSON file and split it by its ``split`` column.

    Args:
        data_json_path: Path to one of the ``data/*_all.json`` files.
        image_root: Optional directory that the relative ``image`` paths are
            resolved against. If ``None``, paths are used as stored, i.e.
            relative to the current working directory.
    """
    raw = load_dataset("json", data_files=data_json_path)["train"]

    if image_root is not None:
        def _resolve(example):
            for key in ("image", "image_path"):
                if example.get(key) and not os.path.isabs(example[key]):
                    example[key] = os.path.normpath(os.path.join(image_root, example[key]))
            return example

        raw = raw.map(_resolve)

    return DatasetDict(
        {
            "train": raw.filter(lambda x: x["split"] == "train"),
            "test": raw.filter(lambda x: x["split"] == "test"),
        }
    )
