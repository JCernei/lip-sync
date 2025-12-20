import os
import ffmpeg
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_media_files(input_dir, output_dir, ffmpeg_options, output_format=None):
    """
    Processes all media files in a directory using specified FFmpeg options.

    Args:
        input_dir (str): The directory containing input media files.
        output_dir (str): The directory where processed files will be saved.
        ffmpeg_options (dict): A dictionary of FFmpeg options for the output.
        output_format (str): Optional output format (e.g., 'wav'). If provided, files will be converted to this format.
    """
    if not os.path.isdir(input_dir):
        logging.error(f"Input directory not found: {input_dir}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)

        if os.path.isfile(input_path):
            # Change file extension if output format is specified
            if output_format:
                base_name = os.path.splitext(filename)[0]
                output_filename = f"{base_name}.{output_format}"
            else:
                output_filename = filename
            
            output_path = os.path.join(output_dir, output_filename)

            try:
                logging.info(f"Processing {filename}...")
                (
                    ffmpeg
                    .input(input_path)
                    .output(output_path, **ffmpeg_options)
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
                logging.info(f"Successfully processed {filename} and saved to {output_path}")
            except ffmpeg.Error as e:
                logging.error(f"Error processing {filename}:")
                # Decode stderr for a readable error message
                error_message = e.stderr.decode('utf-8') if e.stderr else "No stderr output."
                logging.error(error_message)

def main():
    """
    Main function to parse arguments and run media conversions.
    """
    parser = argparse.ArgumentParser(description="Convert video and audio files using FFmpeg.")
    parser.add_argument('--video_dir', type=str, required=True, help='Directory with original videos.')
    parser.add_argument('--video_dir_25fps', type=str, required=True, help='Output directory for 25fps videos.')
    parser.add_argument('--audio_dir', type=str, required=True, help='Directory with original audios.')
    parser.add_argument('--audio_dir_16k', type=str, required=True, help='Output directory for 16kHz audios.')
    
    args = parser.parse_args()

    # Convert videos to 25 fps
    logging.info("--- Starting video conversion to 25 fps ---")
    video_options = {'r': 25}
    process_media_files(args.video_dir, args.video_dir_25fps, video_options)

    # Convert audios to 16000 Hz and wav format
    logging.info("--- Starting audio conversion to 16000 Hz and wav format ---")
    audio_options = {'ar': 16000}
    process_media_files(args.audio_dir, args.audio_dir_16k, audio_options, output_format='wav')
    
    logging.info("--- All processing finished. ---")

if __name__ == "__main__":
    # Example usage:
    # python ffmpeg_converter.py --video_dir ./videos --video_dir_25fps ./videos_25fps \
    # --audio_dir ./audios --audio_dir_16k ./audios_16k
    main()