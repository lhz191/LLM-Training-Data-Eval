#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path


REQUIRED_KEYS = ["id", "image_id", "image_path", "caption"]


def convert(
    input_json: Path,
    output_jsonl: Path,
    image_root: Path | None = None,
    strict: bool = True,
):
    """
    Convert a standard JSON (list of dicts) to JSONL format.

    Args:
        input_json (Path): input .json file
        output_jsonl (Path): output .jsonl file
        image_root (Path | None): optional root to prefix image_path
        strict (bool): whether to enforce required keys
    """
    with input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON top-level must be a list")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    total = len(data)
    written = 0
    skipped = 0

    with output_jsonl.open("w", encoding="utf-8") as fout:
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                skipped += 1
                continue

            # -------- key validation --------
            missing = [k for k in REQUIRED_KEYS if k not in item]
            if missing:
                msg = f"[skip] item {idx} missing keys: {missing}"
                if strict:
                    raise ValueError(msg)
                else:
                    print(msg)
                    skipped += 1
                    continue

            # -------- normalize image path --------
            if image_root is not None:
                item["image_path"] = str(image_root / item["image_path"])

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1

    print("=== JSON → JSONL Conversion Done ===")
    print(f"Input items : {total}")
    print(f"Written     : {written}")
    print(f"Skipped     : {skipped}")
    print(f"Output file : {output_jsonl}")


def main():
    parser = argparse.ArgumentParser("JSON → JSONL converter")
    parser.add_argument("--input", required=True, type=Path, help="input .json file")
    parser.add_argument("--output", required=True, type=Path, help="output .jsonl file")
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="optional root dir to prefix image_path",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="skip invalid items instead of failing",
    )

    args = parser.parse_args()

    convert(
        input_json=args.input,
        output_jsonl=args.output,
        image_root=args.image_root,
        strict=not args.no_strict,
    )


if __name__ == "__main__":
    main()
