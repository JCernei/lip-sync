"""
Visualization script to debug masks and SAM3 overlays before diffusion.
Generates a video showing:
- Original video frames
- Landmark-based masks
- SAM3 segmentation masks
- Combined masks (union with occlusions removed)
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
import cv2
from torchvision.io import read_video, write_video
from safetensors.torch import load_file as load_safetensors

# Add the parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from sgm.util import save_audio_video
from sgm.data.data_utils import (
    create_masks_from_landmarks_box,
    scale_landmarks,
)

try:
    from transformers import Sam3VideoModel, Sam3VideoProcessor
    from transformers.video_utils import load_video
    from accelerate import Accelerator
    SAM3_AVAILABLE = True
except ImportError:
    print("WARNING: SAM3 is not installed. Install with: pip install transformers accelerate")
    SAM3_AVAILABLE = False


def get_segmentation_mask_arms_sam3(
    video_path: str,
    ann_frame_idx: int = 0,
    position: Optional[List[float]] = None,
    text_prompts: Optional[List[str]] = None,
    video_len: Optional[int] = None,
    video_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Generate segmentation masks using SAM3 from transformers.

    Args:
        video_path: Path to the video file
        ann_frame_idx: Frame index for annotation
        position: Position to place the annotation point [x, y] (optional if text_prompts provided)
        text_prompts: List of text descriptions to segment (e.g., ["microphone", "hands"])
        video_len: Length of the video in frames
        video_size: Size of the video frames (height, width)

    Returns:
        Segmentation masks, shape (n_prompts, T, H, W) at full video resolution
    """
    if not SAM3_AVAILABLE:
        raise ImportError(
            "SAM3 is not installed. Install with: pip install transformers accelerate"
        )

    if position is None and (text_prompts is None or len(text_prompts) == 0):
        raise ValueError("Either 'position' (coordinates) or 'text_prompts' must be provided")

    device = Accelerator().device
    print(f"Using device: {device}")

    # Load SAM3 model and processor
    print("Loading SAM3 model...")
    model = Sam3VideoModel.from_pretrained("facebook/sam3").to(
        device, dtype=torch.bfloat16
    )
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

    # Load video frames
    print(f"Loading video from: {video_path}")
    video_frames, _ = load_video(video_path)
    video_frames = torch.from_numpy(video_frames).to(device)

    # Ensure correct shape: (T, C, H, W)
    if video_frames.shape[1] not in [3, 4]:
        video_frames = video_frames.permute(0, 3, 1, 2)

    print(f"Video shape: {video_frames.shape}")

    # Initialize video inference session
    print("Initializing video inference session...")
    inference_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16,
    )

    # Add prompt - either text or point
    if text_prompts is not None and len(text_prompts) > 0:
        print(f"Adding text prompts: {text_prompts}")
        for text in text_prompts:
            inference_session = processor.add_text_prompt(
                inference_session=inference_session,
                text=text,
            )
    else:
        print(f"Adding point prompt at position: {position}")
        inference_session = processor.add_point_prompt(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            point_coords=np.array([[position]], dtype=np.float32),
            point_labels=np.array([[1]], dtype=np.int32),
        )

    # Process all frames in the video
    print("Propagating through video frames...")
    n_prompts = len(text_prompts) if text_prompts else 1
    mask = np.zeros((n_prompts, video_len, video_size[0], video_size[1]), dtype=np.float32)
    
    outputs_per_frame = {}
    frame_count = 0
    max_masks = 0

    # First pass: determine max masks
    for model_outputs in model.propagate_in_video_iterator(
        inference_session=inference_session, max_frame_num_to_track=video_len
    ):
        frame_idx = model_outputs.frame_idx
        frame_count += 1
        
        if frame_count % 10 == 0:
            print(f"  Processed {frame_count} frames (first pass)...")
        
        processed_outputs = processor.postprocess_outputs(
            inference_session, model_outputs
        )
        
        if processed_outputs["masks"] is not None:
            obj_masks = processed_outputs["masks"]
            max_masks = max(max_masks, len(obj_masks))

    # Expand mask array if needed
    if max_masks > n_prompts:
        print(f"Expanding mask array from {n_prompts} to {max_masks}")
        new_mask = np.zeros((max_masks, video_len, video_size[0], video_size[1]), dtype=np.float32)
        new_mask[:n_prompts] = mask
        mask = new_mask
        n_prompts = max_masks

    # Second pass: fill in the masks
    frame_count = 0
    for model_outputs in model.propagate_in_video_iterator(
        inference_session=inference_session, max_frame_num_to_track=video_len
    ):
        frame_idx = model_outputs.frame_idx
        frame_count += 1
        
        if frame_count % 10 == 0:
            print(f"  Processed {frame_count} frames (second pass)...")
        
        processed_outputs = processor.postprocess_outputs(
            inference_session, model_outputs
        )
        
        if processed_outputs["masks"] is not None:
            obj_masks = processed_outputs["masks"]
            
            for mask_idx, obj_mask in enumerate(obj_masks):
                if obj_mask.shape != video_size:
                    obj_mask = torch.from_numpy(obj_mask).float().unsqueeze(0).unsqueeze(0)
                    obj_mask = F.interpolate(
                        obj_mask, size=video_size, mode="nearest"
                    ).squeeze(0).squeeze(0).numpy()
                else:
                    if isinstance(obj_mask, torch.Tensor):
                        obj_mask = obj_mask.cpu().numpy()
                
                mask[mask_idx, frame_idx] = obj_mask

    print(f"Processed {frame_count} total frames")

    # Fill backward if needed
    for mask_idx in range(n_prompts):
        if ann_frame_idx > 0 and np.sum(mask[mask_idx, ann_frame_idx]) > 0:
            for frame_idx in range(ann_frame_idx - 1, -1, -1):
                if np.sum(mask[mask_idx, frame_idx]) == 0:
                    mask[mask_idx, frame_idx] = mask[mask_idx, frame_idx + 1]

    return mask


def load_landmarks_mask(
    landmarks: np.ndarray,
    original_size: Tuple[int, int],
    target_size: Tuple[int, int] = (512, 512),
    what_mask: str = "box",
    nose_index: int = 28,
) -> torch.Tensor:
    """Load and process facial landmarks to create masks."""
    if len(landmarks.shape) == 2:
        landmarks = landmarks[None, ...]
    
    if what_mask == "box":
        mask = create_masks_from_landmarks_box(
            landmarks,
            original_size,
            box_expand=0.0,
            nose_index=nose_index,
        )
    else:
        mask = create_masks_from_landmarks_box(
            landmarks,
            original_size,
            box_expand=0.0,
            nose_index=nose_index,
        )
    
    mask = F.interpolate(
        mask.unsqueeze(1).float(), size=target_size, mode="nearest"
    )
    return mask.squeeze(1).numpy()


def create_visualization_frame(
    original_frame: np.ndarray,
    landmark_mask: np.ndarray,
    sam3_mask: Optional[np.ndarray] = None,
    combined_mask: Optional[np.ndarray] = None,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Create a visualization frame showing all masks overlaid on the original.
    
    Args:
        original_frame: Original video frame (H, W, 3) in [0, 255]
        landmark_mask: Landmark-based mask (H, W) in [0, 1]
        sam3_mask: SAM3 segmentation mask (H, W) in [0, 1]
        combined_mask: Combined mask (H, W) in [0, 1]
        alpha: Transparency for overlays
    
    Returns:
        Visualization frame (H, W, 3) in [0, 255]
    """
    frame = original_frame.copy().astype(np.float32)
    
    # Green overlay for landmark mask
    if landmark_mask is not None:
        landmark_mask = (landmark_mask * 255).astype(np.uint8)
        mask_colored = np.zeros_like(frame)
        mask_colored[..., 1] = landmark_mask  # Green channel
        frame = cv2.addWeighted(frame, 1 - alpha, mask_colored, alpha, 0)
    
    # Red overlay for SAM3 mask
    if sam3_mask is not None:
        sam3_mask_uint8 = (sam3_mask * 255).astype(np.uint8)
        mask_colored = np.zeros_like(frame)
        mask_colored[..., 2] = sam3_mask_uint8  # Red channel
        frame = cv2.addWeighted(frame, 1 - alpha, mask_colored, alpha, 0)
    
    # Blue overlay for combined mask (final mask used in model)
    if combined_mask is not None:
        combined_mask_uint8 = (combined_mask * 255).astype(np.uint8)
        mask_colored = np.zeros_like(frame)
        mask_colored[..., 0] = combined_mask_uint8  # Blue channel
        frame = cv2.addWeighted(frame, 1 - alpha, mask_colored, alpha, 0)
    
    return np.clip(frame, 0, 255).astype(np.uint8)


def main(
    video_path: str,
    landmark_path: str,
    output_path: str = "outputs/mask_visualization.mp4",
    text_prompts: Optional[List[str]] = None,
    position: Optional[List[float]] = None,
    start_frame: int = 0,
    max_frames: Optional[int] = None,
    what_mask: str = "box",
    nose_index: int = 28,
    frame_size: int = 512,
):
    """
    Main function to create mask visualization video.
    
    Args:
        video_path: Path to input video
        landmark_path: Path to landmarks numpy file
        output_path: Path to save output video
        text_prompts: Text prompts for SAM3 (e.g., ["hands", "microphone"])
        position: Point position for SAM3 [x, y]
        start_frame: Frame index to start annotation
        max_frames: Maximum frames to process
        what_mask: Type of landmark mask ("box", "full", "mouth", etc.)
        nose_index: Index of nose landmark
        frame_size: Size to resize frames to
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    print(f"Loading video from {video_path}")
    video, _, _ = read_video(video_path, output_format="TCHW")
    video = video.float() / 255.0
    original_h, original_w = video.shape[2:]
    
    # Resize to frame_size
    video = F.interpolate(video, (frame_size, frame_size), mode="bilinear")
    video_np = (video * 255).permute(0, 2, 3, 1).numpy().astype(np.uint8)
    
    if max_frames is not None:
        video = video[:max_frames]
        video_np = video_np[:max_frames]
    
    print(f"Video shape: {video.shape}, size: {frame_size}x{frame_size}")
    
    # Load landmarks
    print(f"Loading landmarks from {landmark_path}")
    landmarks = np.load(landmark_path, allow_pickle=True)
    landmarks = scale_landmarks(landmarks[:, :, :2], (original_h, original_w), (frame_size, frame_size))
    
    # Pad landmarks if needed
    if len(landmarks) < len(video):
        landmarks = np.concatenate([
            landmarks,
            landmarks[-1:].repeat(len(video) - len(landmarks), axis=0),
        ])
    landmarks = landmarks[:len(video)]
    
    print(f"Landmarks shape: {landmarks.shape}")
    
    # Generate SAM3 masks if prompts or position provided
    sam3_masks = None
    if (text_prompts is not None and len(text_prompts) > 0) or position is not None:
        if SAM3_AVAILABLE:
            print("Generating SAM3 masks...")
            sam3_masks = get_segmentation_mask_arms_sam3(
                video_path=video_path,
                ann_frame_idx=start_frame,
                position=position,
                text_prompts=text_prompts,
                video_len=len(video),
                video_size=(frame_size, frame_size),
            )
            print(f"SAM3 masks shape: {sam3_masks.shape}")
            # Combine all prompts into single mask
            sam3_masks = np.max(sam3_masks, axis=0)  # (T, H, W)
        else:
            print("WARNING: SAM3 not available, skipping SAM3 masks")
    
    # Generate landmark masks
    print("Generating landmark masks...")
    landmark_masks_list = []
    for i in range(len(landmarks)):
        mask = load_landmarks_mask(
            landmarks[i:i+1],
            (frame_size, frame_size),
            target_size=(frame_size, frame_size),
            what_mask=what_mask,
            nose_index=nose_index,
        )
        landmark_masks_list.append(mask[0])
    landmark_masks = np.stack(landmark_masks_list)
    
    print(f"Landmark masks shape: {landmark_masks.shape}")
    
    # Create visualization frames
    print("Creating visualization frames...")
    viz_frames = []
    
    for frame_idx in range(len(video)):
        if frame_idx % 10 == 0:
            print(f"  Frame {frame_idx}/{len(video)}")
        
        original = video_np[frame_idx]
        landmark_mask = landmark_masks[frame_idx]
        
        # Create combined mask (landmark mask excluding SAM3 occlusions)
        combined_mask = landmark_mask.copy()
        if sam3_masks is not None:
            # Remove SAM3-detected occlusions from landmark mask
            combined_mask = np.logical_and(
                landmark_mask,
                np.logical_not(sam3_masks[frame_idx])
            ).astype(np.float32)
        
        sam3_mask = sam3_masks[frame_idx] if sam3_masks is not None else None
        
        viz_frame = create_visualization_frame(
            original,
            landmark_mask=None,
            sam3_mask=None,
            combined_mask=combined_mask,
            alpha=0.3,
        )
        
        viz_frames.append(viz_frame)
    
    # Stack and convert to tensor
    viz_array = np.stack(viz_frames)  # Shape: (T, H, W, 3) in uint8
    
    print(f"Visualization array shape: {viz_array.shape}, dtype: {viz_array.dtype}")
    
    # Convert to torch tensor - write_video expects (T, H, W, C) format
    viz_tensor = torch.from_numpy(viz_array)  # (T, H, W, 3)
    
    print(f"Tensor shape: {viz_tensor.shape}")
    print(f"Saving visualization to {output_path}")
    
    write_video(output_path, viz_tensor, fps=25)
    
    print(f"Done! Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize masks and SAM3 overlays")
    parser.add_argument("--video_path", type=str, required=True, help="Path to video file")
    parser.add_argument("--landmark_path", type=str, required=True, help="Path to landmarks .npy file")
    parser.add_argument("--output_path", type=str, default="outputs/mask_visualization.mp4", help="Output video path")
    parser.add_argument("--text_prompts", nargs="+", default=None, help="Text prompts for SAM3 (e.g., hands microphone)")
    parser.add_argument("--position", nargs=2, type=float, default=None, help="Point position for SAM3 [x y]")
    parser.add_argument("--start_frame", type=int, default=0, help="Frame to start annotation")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--what_mask", type=str, default="box", help="Type of landmark mask")
    parser.add_argument("--nose_index", type=int, default=28, help="Nose landmark index")
    parser.add_argument("--frame_size", type=int, default=512, help="Frame size to resize to")
    
    args = parser.parse_args()
    
    main(
        video_path=args.video_path,
        landmark_path=args.landmark_path,
        output_path=args.output_path,
        text_prompts=args.text_prompts,
        position=args.position,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        what_mask=args.what_mask,
        nose_index=args.nose_index,
        frame_size=args.frame_size,
    )
