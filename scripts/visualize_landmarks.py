"""
Visualization script to visualize facial landmarks on video frames.
Draws landmarks as points and optionally connects them with lines.
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

# Add the parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from sgm.data.data_utils import scale_landmarks


def draw_landmarks_on_frame(
    frame: np.ndarray,
    landmarks: np.ndarray,
    point_radius: int = 4,
    point_color: Tuple[int, int, int] = (0, 255, 0),
    line_color: Tuple[int, int, int] = (255, 0, 0),
    line_width: int = 2,
    draw_connections: bool = True,
) -> np.ndarray:
    """
    Draw facial landmarks on a frame.
    
    Args:
        frame: Video frame (H, W, 3) in [0, 255]
        landmarks: Landmarks array (N, 2) with x, y coordinates
        point_radius: Radius of landmark points
        point_color: Color of landmark points (B, G, R)
        line_color: Color of connection lines (B, G, R)
        line_width: Width of connection lines
        draw_connections: Whether to draw lines connecting landmarks
    
    Returns:
        Frame with drawn landmarks (H, W, 3) in [0, 255]
    """
    frame = frame.copy().astype(np.uint8)
    
    if landmarks is None or len(landmarks) == 0:
        return frame
    
    # Standard MediaPipe face landmark connections
    # These connect the landmark points to form the face structure
    if draw_connections:
        connections = [
            # Face contour
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
            (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
            (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23), (23, 24),
            (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32),
            # Left eyebrow
            (33, 34), (34, 35), (35, 36), (36, 37), (37, 38),
            # Right eyebrow
            (39, 40), (40, 41), (41, 42), (42, 43), (43, 44),
            # Left eye
            (45, 46), (46, 47), (47, 48), (48, 49), (49, 50), (50, 45),
            # Right eye
            (52, 53), (53, 54), (54, 55), (55, 56), (56, 57), (57, 52),
            # Mouth
            (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 61),
            (68, 69), (69, 70), (70, 71), (71, 72), (72, 73), (73, 74), (74, 75), (75, 68),
            # Nose
            (51, 58), (51, 59), (51, 60),
        ]
        
        for start, end in connections:
            if start < len(landmarks) and end < len(landmarks):
                pt1 = tuple(landmarks[start].astype(int))
                pt2 = tuple(landmarks[end].astype(int))
                cv2.line(frame, pt1, pt2, line_color, line_width)
    
    # Draw landmark points
    for i, (x, y) in enumerate(landmarks):
        x, y = int(x), int(y)
        cv2.circle(frame, (x, y), point_radius, point_color, -1)
        # Optionally add index text
        # cv2.putText(frame, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return frame


def main(
    video_path: str,
    landmark_path: str,
    output_path: str = "outputs/landmarks_visualization.mp4",
    start_frame: int = 0,
    max_frames: Optional[int] = None,
    frame_size: int = 512,
    draw_connections: bool = True,
    point_radius: int = 4,
    line_width: int = 2,
):
    """
    Main function to create landmarks visualization video.
    
    Args:
        video_path: Path to input video
        landmark_path: Path to landmarks numpy file
        output_path: Path to save output video
        start_frame: Frame index to start from
        max_frames: Maximum frames to process
        frame_size: Size to resize frames to
        draw_connections: Whether to draw lines connecting landmarks
        point_radius: Radius of landmark points
        line_width: Width of connection lines
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    print(f"Loading video from {video_path}")
    video, _, _ = read_video(video_path, output_format="TCHW")
    video = video.float() / 255.0
    original_h, original_w = video.shape[2:]
    
    # Resize to frame_size
    video = F.interpolate(video, (frame_size, frame_size), mode="bilinear")
    video_np = (video * 255).permute(0, 2, 3, 1).numpy().astype(np.uint8)
    
    # Apply start_frame and max_frames
    if start_frame > 0:
        video_np = video_np[start_frame:]
    
    if max_frames is not None:
        video_np = video_np[:max_frames]
    
    print(f"Video shape: {video_np.shape}, size: {frame_size}x{frame_size}")
    
    # Load landmarks
    print(f"Loading landmarks from {landmark_path}")
    landmarks = np.load(landmark_path, allow_pickle=True)
    
    # Scale landmarks to match frame size
    landmarks = scale_landmarks(landmarks[:, :, :2], (original_h, original_w), (frame_size, frame_size))
    
    # Apply start_frame offset
    if start_frame > 0:
        landmarks = landmarks[start_frame:]
    
    # Trim to match video frames
    if len(landmarks) > len(video_np):
        landmarks = landmarks[:len(video_np)]
    
    # Pad landmarks if needed
    if len(landmarks) < len(video_np):
        landmarks = np.concatenate([
            landmarks,
            landmarks[-1:].repeat(len(video_np) - len(landmarks), axis=0),
        ])
    
    print(f"Landmarks shape: {landmarks.shape}")
    
    # Create visualization frames
    print("Creating visualization frames...")
    viz_frames = []
    
    for frame_idx in range(len(video_np)):
        if frame_idx % 10 == 0:
            print(f"  Frame {frame_idx}/{len(video_np)}")
        
        original = video_np[frame_idx]
        frame_landmarks = landmarks[frame_idx]
        
        # Draw landmarks on frame
        viz_frame = draw_landmarks_on_frame(
            original,
            frame_landmarks,
            point_radius=point_radius,
            point_color=(0, 255, 0),  # Green
            line_color=(255, 0, 0),   # Red
            line_width=line_width,
            draw_connections=draw_connections,
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
    parser = argparse.ArgumentParser(description="Visualize facial landmarks")
    parser.add_argument("--video_path", type=str, required=True, help="Path to video file")
    parser.add_argument("--landmark_path", type=str, required=True, help="Path to landmarks .npy file")
    parser.add_argument("--output_path", type=str, default="outputs/landmarks_visualization.mp4", help="Output video path")
    parser.add_argument("--start_frame", type=int, default=0, help="Frame index to start from")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--frame_size", type=int, default=512, help="Frame size to resize to")
    parser.add_argument("--draw_connections", type=bool, default=True, help="Whether to draw landmark connections")
    parser.add_argument("--point_radius", type=int, default=4, help="Radius of landmark points")
    parser.add_argument("--line_width", type=int, default=2, help="Width of connection lines")
    
    args = parser.parse_args()
    
    main(
        video_path=args.video_path,
        landmark_path=args.landmark_path,
        output_path=args.output_path,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        frame_size=args.frame_size,
        draw_connections=args.draw_connections,
        point_radius=args.point_radius,
        line_width=args.line_width,
    )
