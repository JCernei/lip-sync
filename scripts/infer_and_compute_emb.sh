#!/bin/bash

# Script to run the full inference pipeline:
# 1. Compute video embeddings
# 2. Compute audio embeddings
# 3. Create filelist
# 4. Run inference

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --video_dir)
            video_dir="${2:-data/videos}"
            shift 2
            ;;
        --audio_dir)
            audio_dir="${2:-data/audios}" 
            shift 2
            ;;
        --output_folder)
            output_folder="${2:-outputs}"
            shift 2
            ;;
        --keyframes_ckpt)
            keyframes_ckpt="${2:-None}"
            shift 2
            ;;
        --interpolation_ckpt)
            interpolation_ckpt="${2:-None}"
            shift 2
            ;;
        --compute_until)
            compute_until="${2:-45}"
            shift 2
            ;;
        --fix_occlusion)
            fix_occlusion="${2:-false}"
            shift 2
            ;;
        --fix_occlusion_pro)
            fix_occlusion_pro="${2:-false}"
            shift 2
            ;;
        --position)
            position="${2:-None}"
            shift 2
            ;;
        --text_prompts)
            text_prompts="${2:-None}"
            shift 2
            ;; 
        --start_frame)
            start_frame="${2:-0}"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done
echo "video_dir: $video_dir"
echo "audio_dir: $audio_dir"
echo "output_folder: $output_folder"
echo "keyframes_ckpt: $keyframes_ckpt"
echo "interpolation_ckpt: $interpolation_ckpt"
echo "compute_until: $compute_until"
echo "fix_occlusion: $fix_occlusion"
echo "fix_occlusion_pro: $fix_occlusion_pro"
echo "position: $position"
echo "text_prompts: ${text_prompts[@]:-None}"
echo "start_frame: $start_frame"

# Set defaults if not provided
video_dir=${video_dir:-"data/videos"}
audio_dir=${audio_dir:-"data/audios"}
output_folder=${output_folder:-"outputs"}
keyframes_ckpt=${keyframes_ckpt:-pretrained_models/checkpoints/keyframe_dub.pt}
interpolation_ckpt=${interpolation_ckpt:-pretrained_models/checkpoints/interpolation_dub.pt}
compute_until=${compute_until:-45}

# Define directories
script_dir="scripts"
util_dir="$script_dir/util"
video_latent_dir="video_crop_emb"
audio_emb_dir="audio_emb"
filelist="filelist_inference.txt"
filelist_audio="filelist_inference_audio.txt"

# Step 0: Pre-process video and audio
echo "Step 0: Pre-processing video and audio..."

# Create new directories for processed files
video_dir_25fps="${video_dir}_25fps"
audio_dir_16k="${audio_dir}_16k"
mkdir -p "$video_dir_25fps"
mkdir -p "$audio_dir_16k"

# Convert videos to 25 fps and audios to 16000 Hz using the Python script
echo "Converting videos to 25 fps and audios to 16000 Hz..."
python $util_dir/ffmpeg_converter.py \
    --video_dir "$video_dir" \
    --video_dir_25fps "$video_dir_25fps" \
    --audio_dir "$audio_dir" \
    --audio_dir_16k "$audio_dir_16k"

if [ $? -ne 0 ]; then
    echo "ffmpeg_converter.py failed. Exiting." >&2
    exit 1
fi

# Update variables to point to the new directories
video_dir="$video_dir_25fps"
audio_dir="$audio_dir_16k"

echo "Pre-processing complete. Using processed files from $video_dir and $audio_dir"


echo "Step 1: Computing landmarks..."
python $util_dir/gen_landmarks.py \
    $video_dir \
    --output_dir "landmarks_25fps" \
    --batch_size 10

video_dir_cropped="${video_dir}_cropped"

echo "Step 2: Cropping video..."
python $util_dir/crop_video.py \
    --video_dir "$video_dir" \
    --video_dir_cropped "$video_dir_cropped" \
    --landmarks_dir "landmarks_25fps" \
    --landmarks_dir_cropped "landmarks_25fps_cropped" \

video_dir="$video_dir_cropped"

echo "Step 3: Computing video embeddings..."
python $util_dir/video_to_latent.py \
    --filelist "$video_dir" \

echo "Step 4: Computing audio embeddings..."
for model_type in wavlm hubert; do
    python $util_dir/get_audio_embeddings.py \
        --audio_path "$audio_dir/*.wav" \
        --model_type $model_type \
        --skip_video
done

echo "Step 5: Creating filelist for inference..."
python $util_dir/create_filelist.py \
    --root_dir $video_dir \
    --dest_file $filelist \
    --ext ".mp4"

python $util_dir/create_filelist.py \
    --root_dir $audio_dir \
    --dest_file $filelist_audio \
    --ext ".wav"

echo "Step 6: Running inference..."
$script_dir/inference.sh \
    --output_folder "$output_folder" \
    --file_list "$filelist" \
    --keyframes_ckpt "$keyframes_ckpt" \
    --interpolation_ckpt "$interpolation_ckpt" \
    --compute_until "$compute_until" \
    --file_list_audio "$filelist_audio" \
    --fix_occlusion "$fix_occlusion" \
    --fix_occlusion_pro "$fix_occlusion_pro" \
    --position "$position" \
    --text_prompts "${text_prompts}" \
    --start_frame "$start_frame"

echo "Inference pipeline completed successfully!"
