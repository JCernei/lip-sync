#!/usr/bin/env python3
"""
Test script for SAM3-based occlusion masking.

This script tests the SAM3 occlusion fixing functionality independently
without running the full inference pipeline.

Usage:
    python test_sam3_occlusion.py --video_path path/to/video.mp4 --position 450,450 --start_frame 0
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import cv2

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from transformers import Sam3VideoModel, Sam3VideoProcessor
    from transformers.video_utils import load_video
    from accelerate import Accelerator
    SAM3_AVAILABLE = True
except ImportError as e:
    print(f"SAM3 dependencies not installed: {e}")
    print("Install with: pip install transformers accelerate")
    SAM3_AVAILABLE = False


def overlay_masks(image, masks):
    """
    Overlay masks on an image with different colors.
    
    Args:
        image: PIL Image
        masks: numpy array of shape (n_masks, H, W) with values in [0, 1]
    
    Returns:
        PIL Image with overlaid masks
    """
    from PIL import Image
    import matplotlib
    
    image = image.convert("RGBA")
    masks = (255 * masks).astype(np.uint8)
    
    n_masks = masks.shape[0]
    if n_masks == 0:
        return image
    
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    for mask, color in zip(masks, colors):
        # Resize mask to match image size if needed
        mask_img = Image.fromarray(mask)
        if mask_img.size != image.size:
            mask_img = mask_img.resize(image.size, Image.NEAREST)
        
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask_img.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    
    return image


def save_masks_as_video(
    video_frames: torch.Tensor,
    masks: np.ndarray,
    output_path: str,
    fps: float = 25.0,
    text_prompts: Optional[List[str]] = None,
) -> None:
    """
    Save video with mask overlay.
    
    Args:
        video_frames: Original video frames with shape (n_frames, C, H, W) or (n_frames, H, W, C)
        masks: Segmentation masks with shape (n_masks, n_frames, H, W)
        output_path: Path to save the video file
        fps: Frames per second for the output video
        text_prompts: List of text prompts used for segmentation
    """
    n_masks, n_frames, h, w = masks.shape
    
    # Ensure video frames are in (n_frames, H, W, C) format
    if video_frames.shape[1] in [3, 4]:  # If channels dimension is second
        video_frames = video_frames.permute(0, 2, 3, 1)
    
    # Convert video frames to numpy if needed
    if isinstance(video_frames, torch.Tensor):
        video_frames = video_frames.cpu().numpy()
    
    # Ensure frames are uint8
    if video_frames.dtype != np.uint8:
        if video_frames.max() <= 1.0:
            video_frames = (video_frames * 255).astype(np.uint8)
        else:
            video_frames = video_frames.astype(np.uint8)
    
    # Create a colormap for different masks
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        (int(c[2] * 255), int(c[1] * 255), int(c[0] * 255))  # BGR format for OpenCV
        for c in [cmap(i)[:3] for i in range(n_masks)]
    ]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    print(f"Writing mask overlay video to {output_path}...")
    
    for frame_idx in range(n_frames):
        # Get original video frame
        orig_frame = video_frames[frame_idx].copy()  # (H, W, 3)
        
        # Convert to BGR if needed (OpenCV uses BGR)
        if orig_frame.shape[2] == 3:
            orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_RGB2BGR)
        elif orig_frame.shape[2] == 4:
            orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_RGBA2BGR)
        
        # Overlay masks on the original frame
        frame = orig_frame.astype(np.float32)
        
        for mask_idx in range(n_masks):
            mask = masks[mask_idx, frame_idx, :, :]  # (H, W)
            color = np.array(colors[mask_idx], dtype=np.float32)
            
            # Create a color overlay
            colored_mask = np.zeros((h, w, 3), dtype=np.float32)
            colored_mask[:, :] = color
            
            # Blend the mask with the frame using alpha compositing
            alpha = (mask * 0.5).astype(np.float32)
            frame = frame * (1.0 - alpha[:, :, np.newaxis]) + colored_mask * alpha[:, :, np.newaxis]
        
        # Convert back to uint8
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        out.write(frame)
        
        if (frame_idx + 1) % 30 == 0:
            print(f"  Written {frame_idx + 1}/{n_frames} frames...")
    
    out.release()
    print(f"Mask overlay video saved successfully!")
    print(f"  Resolution: {w}x{h}")
    print(f"  Frames: {n_frames}")
    print(f"  FPS: {fps}")
    if text_prompts:
        print(f"  Masks: {', '.join(text_prompts)}")



def get_segmentation_mask_arms_sam3(
    video_path: str,
    ann_frame_idx: int,
    position: Optional[List[float]] = None,
    text_prompts: Optional[List[str]] = None,
    video_len: int = None,
    video_size: Tuple[int, int] = None,
) -> np.ndarray:
    """
    Generate a segmentation mask for arms using SAM3 from transformers.

    Args:
        video_path: Path to the video file
        ann_frame_idx: Frame index for annotation
        position: Position to place the annotation point [x, y] (optional if text_prompt provided)
        text_prompt: Text description of object to segment (e.g., "arm", "hand", "person")
        video_len: Length of the video in frames
        video_size: Size of the video frames (height, width)

    Returns:
        Segmentation mask for arms region at full video resolution
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
    video_frames = torch.from_numpy(video_frames).to(device)  # Shape: (T, C, H, W) or (T, H, W, C)

    # Ensure correct shape: (T, C, H, W)
    if video_frames.shape[1] not in [3, 4]:  # If not channels dimension
        video_frames = video_frames.permute(0, 3, 1, 2)  # Convert (T, H, W, C) to (T, C, H, W)

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
            point_coords=np.array([[position]], dtype=np.float32),  # (1, 1, 2)
            point_labels=np.array([[1]], dtype=np.int32),  # Positive label
        )

    # Process all frames in the video
    print("Propagating through video frames...")
    n_prompts = len(text_prompts) if text_prompts else 1
    mask = np.zeros((n_prompts, video_len, video_size[0], video_size[1]), dtype=np.float32)
    
    outputs_per_frame = {}
    frame_count = 0
    max_masks = 0  # Track the maximum number of masks returned
    
    for model_outputs in model.propagate_in_video_iterator(
        inference_session=inference_session, max_frame_num_to_track=video_len
    ):
        frame_idx = model_outputs.frame_idx
        frame_count += 1
        
        if frame_count % 10 == 0:
            print(f"  Processed {frame_count} frames...")
        
        # Process the outputs
        processed_outputs = processor.postprocess_outputs(
            inference_session, model_outputs
        )
        
        if processed_outputs["masks"] is not None:
            # Get all object masks
            obj_masks = processed_outputs["masks"]  # (n_masks, H, W)
            max_masks = max(max_masks, len(obj_masks))
    
    # If SAM3 returned more masks than expected, expand the mask array
    if max_masks > n_prompts:
        print(f"Warning: SAM3 returned {max_masks} masks but only {n_prompts} prompts were provided")
        print(f"Expanding mask array to {max_masks}")
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
        
        # Process the outputs
        processed_outputs = processor.postprocess_outputs(
            inference_session, model_outputs
        )
        
        if processed_outputs["masks"] is not None:
            # Get all object masks
            obj_masks = processed_outputs["masks"]  # (n_masks, H, W)
            
            for mask_idx, obj_mask in enumerate(obj_masks):
                # Resize to video size if needed
                if obj_mask.shape != video_size:
                    obj_mask = torch.from_numpy(obj_mask).float().unsqueeze(0).unsqueeze(0)
                    obj_mask = F.interpolate(
                        obj_mask, size=video_size, mode="nearest"
                    ).squeeze(0).squeeze(0).numpy()
                else:
                    # Convert to numpy and ensure it's on CPU
                    if isinstance(obj_mask, torch.Tensor):
                        obj_mask = obj_mask.cpu().numpy()
                
                mask[mask_idx, frame_idx] = obj_mask
        
        outputs_per_frame[frame_idx] = processed_outputs

    print(f"Processed {frame_count} total frames")

    # Fill backward if needed
    for mask_idx in range(n_prompts):
        if ann_frame_idx > 0 and np.sum(mask[mask_idx, ann_frame_idx]) > 0:
            for frame_idx in range(ann_frame_idx - 1, -1, -1):
                if np.sum(mask[mask_idx, frame_idx]) == 0:
                    mask[mask_idx, frame_idx] = mask[mask_idx, frame_idx + 1]

    return mask


def main():
    parser = argparse.ArgumentParser(
        description="Test SAM3-based occlusion masking"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to the video file"
    )
    parser.add_argument(
        "--position",
        type=str,
        default=None,
        help="Position for annotation (format: x,y, e.g., '450,450')"
    )
    parser.add_argument(
        "--text_prompts",
        type=str,
        nargs="+",
        default=None,
        help="Text descriptions of objects to segment (e.g., 'ear', 'dial')"
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="Start frame index for annotation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/sam3_test",
        help="Output directory for masks"
    )
    
    args = parser.parse_args()

    # Validate inputs
    if args.position is None and (args.text_prompts is None or len(args.text_prompts) == 0):
        print("Error: Either --position or --text_prompts must be provided")
        sys.exit(1)
    
    # Parse position if provided
    position = None
    if args.position is not None:
        pos_x, pos_y = map(float, args.position.split(","))
        position = [pos_x, pos_y]
    
    # Check if video exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)
    
    if not SAM3_AVAILABLE:
        print("Error: SAM3 is not installed")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("SAM3 Occlusion Mask Generation Test")
    print(f"{'='*60}")
    print(f"Video: {args.video_path}")
    if position:
        print(f"Position: {position}")
    if args.text_prompts:
        print(f"Text Prompts: {args.text_prompts}")
    print(f"Start Frame: {args.start_frame}")
    print(f"Output Dir: {args.output_dir}")
    print(f"{'='*60}\n")

    try:
        from torchvision.io import read_video
        
        # Get video info
        print("Reading video metadata...")
        video, _, info = read_video(args.video_path, output_format="TCHW")
        video_len = video.shape[0]
        h, w = video.shape[2], video.shape[3]
        
        print(f"Video shape: {video.shape}")
        print(f"Total frames: {video_len}")
        print(f"Video resolution: {h}x{w}")
        
        # Generate mask
        print("\nGenerating segmentation masks...")
        mask = get_segmentation_mask_arms_sam3(
            video_path=args.video_path,
            ann_frame_idx=args.start_frame,
            position=position,
            text_prompts=args.text_prompts,
            video_len=video_len,
            video_size=(h, w),
        )
        
        # Save masks
        mask_path = os.path.join(args.output_dir, "masks_sam3.npy")
        np.save(mask_path, mask)
        print(f"\nMasks saved to: {mask_path}")
        print(f"Masks shape: {mask.shape}")
        print(f"Masks min: {mask.min():.4f}, max: {mask.max():.4f}")
        print(f"Masks mean: {mask.mean():.4f}")
        if args.text_prompts:
            for i, prompt in enumerate(args.text_prompts):
                print(f"  Mask {i} ('{prompt}'): mean = {mask[i].mean():.4f}")
        
        # Save visualization
        print("\nGenerating visualizations...")
        vis_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Save masks as video
        video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
        video_output_path = os.path.join(args.output_dir, f"{video_basename}_masked.mp4")
        save_masks_as_video(
            video_frames=video,
            masks=mask,
            output_path=video_output_path,
            fps=25.0,
            text_prompts=args.text_prompts
        )
        
        try:
            import matplotlib.pyplot as plt
            from PIL import Image
            
            # Save a few sample frames
            sample_frames = [
                args.start_frame,
                min(args.start_frame + video_len // 3, video_len - 1),
                min(args.start_frame + 2 * video_len // 3, video_len - 1),
            ]
            
            for frame_idx in sample_frames:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                frame = video[frame_idx].permute(1, 2, 0).numpy().astype(np.uint8)
                axes[0].imshow(frame)
                axes[0].set_title(f"Original Frame {frame_idx}")
                axes[0].axis("off")
                
                # Overlay all masks on this frame
                mask_frame = mask[:, frame_idx, :, :]  # Shape (n_prompts, H, W)
                pil_image = Image.fromarray(frame)
                pil_with_mask = overlay_masks(pil_image, mask_frame)
                
                axes[1].imshow(np.array(pil_with_mask))
                mask_labels = ", ".join(args.text_prompts) if args.text_prompts else "position"
                axes[1].set_title(f"Mask Overlay: {mask_labels}")
                axes[1].axis("off")
                
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f"frame_{frame_idx:03d}.png"), dpi=100)
                plt.close()
            
            print(f"Visualizations saved to: {vis_dir}")
            
        except ImportError:
            print("matplotlib not installed, skipping visualizations")
        
        print(f"\n{'='*60}")
        print("Test completed successfully!")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\nError during mask generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
