import os
from PIL import Image
import re
import numpy as np
import argparse
import shutil

def clean_directory(directory_path):
    """
    Removes the directory if it exists and then recreates it.
    """
    if os.path.exists(directory_path):
        print(f"Cleaning existing directory: {directory_path}")
        shutil.rmtree(directory_path)
    else:
        print(f"Directory does not exist and will be created: {directory_path}")
    
    os.makedirs(directory_path)
    print(f"Directory is ready: {directory_path}")

def parse_matrix_arg(arg):
    """
    Parses a string argument into a NumPy matrix.
    The input can be either 3 comma-separated values for a diagonal matrix
    or 9 values for a full 3x3 matrix.
    """
    values = [float(x) for x in arg.split(',')]
    if len(values) == 3:
        # Diagonal matrix
        return np.diag(values)
    elif len(values) == 9:
        # Full 3x3 matrix
        return np.array(values).reshape(3, 3)
    else:
        raise ValueError(f"Invalid number of elements in matrix argument: expected 3 or 9, got {len(values)}")

def check_rotation_matrix(R):
    """
    Checks if a matrix is a valid rotation matrix.
    """
    # Check orthonormality: R^T * R = I
    identity = np.dot(R.T, R)
    is_orthonormal = np.allclose(identity, np.eye(3), atol=1e-6)

    # Check determinant: det(R) = 1
    determinant = np.linalg.det(R)
    is_valid_determinant = np.isclose(determinant, 1.0, atol=1e-6)

    return is_orthonormal, is_valid_determinant

def parse_extrinsics(file_path, flip_R, flip_T):
    """Parse extrinsics file into a 4x4 matrix."""
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    if len(lines) < 4:
        raise ValueError(f"Insufficient lines in {file_path}")
    
    T = np.array(list(map(float, lines[0].split())))
    R = np.array([list(map(float, lines[i].split())) for i in range(1, 4)])
    
    is_orthonormal, _ = check_rotation_matrix(R)
    
    # original conventions: -X, +Y, +Z, target: +X, -Y, +Z
    # Blender: +X, +Y, -Z
    # flip_R = np.array([[-1, 0, 0], 
    #                     [0, -1, 0], 
    #                     [0, 0, 1]])
    # R = np.dot(np.array(R), flip_R)
    # flip_R = np.diag([-1, 1, 1])
    # flip_T = np.diag([-1, 1, 1])
    R = R @ flip_R
    T = T @ flip_T
    
    # input: c2w, target: w2c
    # R_c2w = R.T
    # t_c2w = -R_c2w @ T
    # C2W = np.eye(4)
    # C2W[:3, :3] = R_c2w
    # C2W[:3, 3] = t_c2w.flatten()
    
    C2W = np.eye(4)
    C2W [:3, :3] = R
    C2W [:3, 3] = T
    
    W2C = np.linalg.inv(C2W)
    return W2C

def parse_intrinsics(file_path):
    """Parse intrinsics file into a 3x3 matrix and additional parameters."""
    with open(file_path, 'r') as f:
        lines = [line.strip().split() for line in f.readlines()]
    
    intrinsics = {int(parts[0]): float(parts[1]) for parts in lines if len(parts) == 2}
    fx, fy = intrinsics.get(4, 0), intrinsics.get(5, 0)
    cx, cy = intrinsics.get(2, 0), intrinsics.get(3, 0)
    
    intrinsic_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    additional_params = "425 2.5"
    return intrinsic_matrix, additional_params

def format_matrix(matrix, header):
    """Format matrix with a header for the output file."""
    formatted = f"{header}\n" + "\n".join(" ".join(f"{val:.6f}" for val in row) for row in matrix) + "\n\n"
    return formatted

def process_cameras(metadata_dir, cameras_dir, flip_R, flip_T):
    """Convert and save camera files."""
    extrinsics_files = [f for f in os.listdir(metadata_dir) if f.startswith('extrinsics_') and f.endswith('.txt')]
    
    for extr_file in extrinsics_files:
        match = re.match(r'extrinsics_(\d+)\.txt', extr_file)
        if not match:
            continue
        cam_id = int(match.group(1))
        intr_file = f'intrinsics_{cam_id}.txt'
        extr_path = os.path.join(metadata_dir, extr_file)
        intr_path = os.path.join(metadata_dir, intr_file)
        
        if not os.path.exists(intr_path):
            print(f"Missing {intr_file} for camera {cam_id}. Skipping.")
            continue
        
        try:
            extrinsic = parse_extrinsics(extr_path, flip_R, flip_T)
            intrinsic, params = parse_intrinsics(intr_path)
        except ValueError as e:
            print(e)
            continue
        
        extr_str = format_matrix(extrinsic, "extrinsic")
        intr_str = format_matrix(intrinsic, "intrinsic") + f"{params}\n"
        camera_content = extr_str + intr_str
        
        new_cam_filename = f"{cam_id:08d}_cam.txt"
        new_cam_path = os.path.join(cameras_dir, new_cam_filename)
        
        with open(new_cam_path, 'w') as f:
            f.write(camera_content)
        
        print(f"Processed camera {cam_id} -> {new_cam_filename}")

def process_images(color_dir, rectified_dir):
    """Rename, check format, and copy images to the Rectified directory."""
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp')
    color_files = [f for f in os.listdir(color_dir) if f.lower().endswith(supported_formats)]
    
    for color_file in color_files:
        match = re.match(r'(\d+)_color_(\d+)\.(png|jpg|jpeg|bmp)', color_file, re.IGNORECASE)
        if not match:
            continue
        scan_id, cam_id = map(int, match.groups()[:2])
        
        src = os.path.join(color_dir, color_file)
        rect_dir = os.path.join(rectified_dir, f"scan{scan_id}_train")
        os.makedirs(rect_dir, exist_ok=True)
        
        new_filename = f"rect_{cam_id+1:03d}_3_r5000.png"
        dest = os.path.join(rect_dir, new_filename)
        
        try:
            with Image.open(src) as img:
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                img.save(dest, format='PNG')
            print(f"Processed {color_file} -> scan{scan_id}_train/{new_filename}")
        except Exception as e:
            print(f"Failed to process {color_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Dataset Conversion to DTU")
    parser.add_argument('root_dir', type=str, help='Path to the dataset root directory')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Path to the output directory (default: same as root_dir)')
    parser.add_argument('--flip_R', type=str, default="-1,1,1",
                        help='Diagonal entries of flip_R matrix or full 3x3 matrix (comma-separated). Default: "-1,1,1"')
    parser.add_argument('--flip_T', type=str, default="-1,1,1",
                        help='Diagonal entries of flip_T matrix or full 3x3 matrix (comma-separated). Default: "-1,1,1"')
    args = parser.parse_args()
    
    root_dir = args.root_dir
    output_dir = args.output_dir if args.output_dir else root_dir
    metadata_dir = os.path.join(root_dir, 'metadata')
    color_dir = os.path.join(root_dir, 'color')
    cameras_dir = os.path.join(output_dir, 'Cameras/train')
    rectified_dir = os.path.join(output_dir, 'Rectified')
    
    # Clean and create output directories
    clean_directory(cameras_dir)
    clean_directory(rectified_dir)
    
    print("Starting dataset conversion...")
    flip_R = parse_matrix_arg(args.flip_R)
    flip_T = parse_matrix_arg(args.flip_T)
    process_cameras(metadata_dir, cameras_dir, flip_R, flip_T)
    process_images(color_dir, rectified_dir)
    print("Dataset conversion completed.")

if __name__ == "__main__":
    main()