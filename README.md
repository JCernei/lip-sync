## Installation

### Prerequisites
- CUDA-compatible GPU
- Python 3.11
- Conda package manager

### Setup Environment

```bash
# Create conda environment with necessary dependencies
conda create -n KeySync python=3.11 conda-forge::ffmpeg -y
conda activate KeySync

# Install requirements
python -m pip install -r requirements.txt --no-deps

# Install PyTorch with CUDA support
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# OPTIONAL
git clone https://github.com/facebookresearch/sam2.git && cd sam2

pip install -e . --no-deps
```

### Known Issues

- On some machines, you may need to install `nvidia::cuda-nvcc`. If this is the case, you can do so by running:

```bash
conda install nvidia::cuda-nvcc
```

- If you encounter synchronization issues between omegaconf and antlr4, you can fix them by running:


```bash
python -m pip uninstall omegaconf antlr4-python3-runtime -y
python -m pip install "omegaconf==2.3.0" "antlr4-python3-runtime==4.9.3"
```


### Download Pretrained Models

```bash
git lfs install
git clone https://huggingface.co/toninio19/keysync pretrained_models
```

## Quick Start Guide

### 1. Data Preparation

To use KeySync with your own data, for simplicity organize your files as follows:
- Place video files (`.mp4`) in the `data/videos/` directory
- Place audio files (`.wav`) in the `data/audios/` directory

Otherwise you need to specify a different video_dir and audio_dir.

### 2. Running Inference

For inference you need to have the audio and video embeddings precomputed.
The simplest way to run inference on your own data is using the `infer_raw.sh` script which will compute those embeddings for you:

```bash
bash scripts/infer_raw.sh \
  --file_list "data/videos" \
  --file_list_audio "data/audios" \
  --output_folder "my_animations" \
  --keyframes_ckpt "path/to/keyframe_dub.pt" \
  --interpolation_ckpt "path/to/interpolation_dub.pt" \
  --compute_until 45
```

This script handles the entire pipeline:
1. Extracts video embeddings
2. Computes landmarks
3. Computes audio embeddings (using WavLM, and Hubert)
4. Creates a filelist for inference
5. Runs the full animation pipeline

For more control over the inference process, you can directly use the `inference.sh` script:

```bash
bash scripts/inference.sh \
  --output_folder "output_folder_name" \
  --file_list "path/to/filelist.txt" \
  --keyframes_ckpt "path/to/keyframes_model.ckpt" \
  --interpolation_ckpt "path/to/interpolation_model.ckpt" \
  --compute_until "compute_until"
```

or if you need to also save intermediate embeddings for faster recompute

```bash
bash scripts/infer_and_compute_emb.sh \
  --filelist "data/videos" \
  --file_list_audio "data/audios" \
  --output_folder "my_animations" \
  --keyframes_ckpt "path/to/keyframes_model.ckpt" \
  --interpolation_ckpt "path/to/interpolation_model.ckpt" \
  --compute_until 45
```

### 3. Training Your Own Models

The dataloader needs the path to all the videos you want to train on. Then you need to separate the audio and video as follows:
- root_folder:
  - videos: raw videos
  - videos_emb: embedding for your videos
  - audios: raw audios
  - audios_emb: precomputed embeddigns for the audios
  - landmarks_folder: landmarks computed from raw video
  
You can have different folders but make sure to change them in the training scripts.

KeySYnc uses a two-stage model approach. You can train each component separately:

#### KeySync Model Training

```bash
bash train_keyframe.sh path/to/filelist.txt [num_workers] [batch_size] [num_devices]
```

#### Interpolation Model Training

```bash
bash train_interpolation.sh path/to/filelist.txt [num_workers] [batch_size] [num_devices]
```

## Advanced Usage

### Command Line Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `video_dir` | Directory with input videos | `data/videos` |
| `audio_dir` | Directory with input audio files | `data/audios` |
| `output_folder` | Where to save generated animations | - |
| `keyframes_ckpt` | Keyframe model checkpoint path | - |
| `interpolation_ckpt` | Interpolation model checkpoint path | - |
| `compute_until` | Animation length in seconds | 45 |
| `fix_occlusion` | Enable occlusion handling to mask objects that block the face | False |
| `position` | Coordinates of the object to mask in the occlusion pipeline (format: x,y, e.g., "450,450") | None |
| `start_frame` | Frame number where the specified position coordinates apply (using the first frame typically works best) | 0 |

### Advanced Configuration

For more fine-grained control, you can edit the configuration files in the `configs/` directory.

## LipScore Evaluation

KeySync can be evaluated using the LipScore metric available in the `evaluation/` folder. This metric measures the lip synchronization quality between generated and ground truth videos.

To use the LipScore evaluation, you'll need to install the following dependencies:

1. Face detection library: [https://github.com/hhj1897/face_detection](https://github.com/hhj1897/face_detection)
2. Face alignment library: [https://github.com/ibug-group/face_alignment](https://github.com/ibug-group/face_alignment)

Once installed, you can use the LipScore class in `evaluation/lipscore.py` to evaluate your generated animations:
