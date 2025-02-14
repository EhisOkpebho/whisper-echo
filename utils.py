import os.path
import subprocess

import imageio_ffmpeg as ffmpeg

from config import WHISPER_PATH, WHISPER_MODEL_PATH

def is_model_available(model: str):
    return os.path.exists(f"{WHISPER_MODEL_PATH}/ggml-{model}.bin")

def convert_to_wav(input_path: str, output_path: str):
    ffmpeg_cmd = [
        ffmpeg.get_ffmpeg_exe(), "-i", input_path,
        "-ar", "16000", "-ac", "1", output_path, "-y"
    ]
    subprocess.run(ffmpeg_cmd, check=True)

def transcribe(file_path: str, model: str):
    if not is_model_available(model):
        raise FileNotFoundError(
            f"The model '{model}' does not exist or is not yet imported. Please try to pull it first.")

    return subprocess.run(
        [WHISPER_PATH, "-m", f"{WHISPER_MODEL_PATH}/ggml-{model}.bin", "-f", file_path],
        capture_output=True, text=True, check=True
    ).stdout