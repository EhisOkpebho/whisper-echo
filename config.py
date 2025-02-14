import os
from dotenv import load_dotenv

load_dotenv()

WHISPER_PATH = os.getenv("WHISPER_PATH")
WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH")
