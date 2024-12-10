''' Build upon: https://github.com/dcharatan/real_estate_10k_tools
                https://github.com/donydchen/matchnerf/blob/main/datasets/dtu.py 
    DTU Acquired instruction: https://github.com/donydchen/matchnerf?tab=readme-ov-file#dtu-for-both-training-and-testing'''

import subprocess
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch
from jaxtyping import Float, Int, UInt8
from PIL import Image
from torch import Tensor
import argparse
from tqdm import tqdm
import json

import os

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, help="input raw data directory")
parser.add_argument("--output_dir", type=str, help="output directory")
args = parser.parse_args()

INPUT_IMAGE_DIR = Path(args.input_dir)
OUTPUT_DIR = Path(args.output_dir)

# number of cameras
NUM_CAMS = 10

# Target 100 MB per chunk.
TARGET_BYTES_PER_CHUNK = int(1e8)

# scale_factor = 1.0 / 200
scale_factor = 1.0

def build_camera_info(id_list, root_dir):
    """Return the camera information for the given id_list"""
    intrinsics, world2cams, cam2worlds, near_fars = {}, {}, {}, {}
    downSample = 1.0
    for vid in id_list:
        proj_mat_filename = os.path.join(
            root_dir, f"Cameras/train/{vid:08d}_cam.txt")
        intrinsic, extrinsic, near_far = read_cam_file(proj_mat_filename)

        # intrinsic[:2] *= 1000.0
        intrinsic[:2] = intrinsic[:2] * downSample
        intrinsics[vid] = intrinsic

        extrinsic[:3, 3] *= scale_factor
        world2cams[vid] = extrinsic
        cam2worlds[vid] = np.linalg.inv(extrinsic)

        near_fars[vid] = near_far

    return intrinsics, world2cams, cam2worlds, near_fars


def read_cam_file(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsic = np.fromstring(" ".join(lines[1:5]), dtype=np.float32, sep=" ")
    extrinsic = extrinsic.reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsic = np.fromstring(" ".join(lines[7:10]), dtype=np.float32, sep=" ")
    intrinsic = intrinsic.reshape((3, 3))
    # depth_min & depth_interval: line 11
    depth_min = float(lines[11].split()[0]) * scale_factor
    depth_max = depth_min + float(lines[11].split()[1]) * 192 * scale_factor
    near_far = [depth_min, depth_max]
    return intrinsic, extrinsic, near_far


def get_example_keys(stage: Literal["test", "train"]) -> list[str]:
    # id_list = [
    #     34, 38, 40, 41, 45, 55, 63, 82, 103, 110, 
    #     114, 138, 150, 159, 173, 226, 255, 286, 321, 437, 602
    # ]
    # id_list = list(range(34, 72, 2))

    id_list = [34, 35, 36]
    keys = [f"scan{id}_train" for id in id_list]
    print(f"Found {len(keys)} keys.")
    return keys


def get_size(path: Path) -> int:
    """Get file or folder size in bytes."""
    return int(subprocess.check_output(["du", "-b", path]).split()[0].decode("utf-8"))


def load_raw(path: Path) -> UInt8[Tensor, " length"]:
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))


def load_images(example_path: Path) -> dict[int, UInt8[Tensor, "..."]]:
    """Load JPG images as raw bytes (do not decode)."""
    images_dict = {}
    width, height = None, None
    for cur_id in range(1, NUM_CAMS+1):
        cur_image_name = f"rect_{cur_id:03d}_3_r5000.png"
        image_path = example_path / cur_image_name
        img_bin = load_raw(image_path)
        images_dict[cur_id - 1] = img_bin
        with Image.open(image_path) as img:
            img_width, img_height = img.size
        if width is None and height is None:
            width, height = img_width, img_height
        else:
            if img_width != width or img_height != height:
                raise ValueError(
                    f"Image {cur_image_name} has dimensions {img_width}x{img_height}, "
                    f"which do not match the expected {width}x{height} pixels."
                )
        print(f"Image dimensions: {width}x{height} pixels.")

    return images_dict, width, height


class Metadata(TypedDict):
    url: str
    timestamps: Int[Tensor, " camera"]
    cameras: Float[Tensor, "camera entry"]


class Example(Metadata):
    key: str
    images: list[UInt8[Tensor, "..."]]


def load_metadata(intrinsics, world2cams, w, h) -> Metadata:
    timestamps = []
    cameras = []
    url = ""

    for vid, intr in intrinsics.items():
        timestamps.append(int(vid))

        fx = intr[0, 0]
        fy = intr[1, 1]
        cx = intr[0, 2]
        cy = intr[1, 2]
        w = 2.0 * cx
        h = 2.0 * cy
        saved_fx = fx / w
        saved_fy = fy / h
        saved_cx = 0.5
        saved_cy = 0.5
        # saved_fx = fx / w
        # saved_fy = fy / h
        # saved_cx = cx / w
        # saved_cy = cy / h
        camera = [saved_fx, saved_fy, saved_cx, saved_cy, 0.0, 0.0]

        w2c = world2cams[vid]
        camera.extend(w2c[:3].flatten().tolist())
        cameras.append(np.array(camera))

    timestamps = torch.tensor(timestamps, dtype=torch.int64)
    cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)

    return {
        "url": url,
        "timestamps": timestamps,
        "cameras": cameras,
    }


if __name__ == "__main__":
    # The data is used only for testing, not training
    for stage in ("test",):
        intrinsics, world2cams, cam2worlds, near_fars = build_camera_info(
            list(range(NUM_CAMS)), INPUT_IMAGE_DIR
        )

        keys = get_example_keys(stage)

        chunk_size = 0
        chunk_index = 0
        chunk: list[Example] = []

        def save_chunk():
            global chunk_size
            global chunk_index
            global chunk

            chunk_key = f"{chunk_index:0>6}"
            print(
                f"Saving chunk {chunk_key} of {len(keys)} ({chunk_size / 1e6:.2f} MB)."
            )
            dir = OUTPUT_DIR / stage
            dir.mkdir(exist_ok=True, parents=True)
            torch.save(chunk, dir / f"{chunk_key}.torch")

            # Reset the chunk.
            chunk_size = 0
            chunk_index += 1
            chunk = []

        for key in keys:
            image_dir = INPUT_IMAGE_DIR / "Rectified" / key
            num_bytes = get_size(image_dir) // 7

            # Read images and metadata.
            images, width, height = load_images(image_dir)
            example = load_metadata(intrinsics, world2cams, width, height)

            # Merge the images into the example.
            example["images"] = [
                images[timestamp.item()] for timestamp in example["timestamps"]
            ]
            assert len(images) == len(example["timestamps"])

            # Add the key to the example.
            example["key"] = key

            print(f"    Added {key} to chunk ({num_bytes / 1e6:.2f} MB).")
            chunk.append(example)
            chunk_size += num_bytes

            if chunk_size >= TARGET_BYTES_PER_CHUNK:
                save_chunk()

        if chunk_size > 0:
            save_chunk()

        # generate index
        print("Generate key:torch index...")
        index = {}
        stage_path = OUTPUT_DIR / stage
        for chunk_path in tqdm(list(stage_path.iterdir()), desc=f"Indexing {stage_path.name}"):
            if chunk_path.suffix == ".torch":
                chunk = torch.load(chunk_path)
                for example in chunk:
                    index[example["key"]] = str(chunk_path.relative_to(stage_path))
        with (stage_path / "index.json").open("w") as f:
            json.dump(index, f)
