import os
from contextlib import asynccontextmanager
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse

from utils import convert_to_wav, transcribe

temp_audio_path = None
temp_wav_path = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    for file_path in [temp_audio_path, temp_wav_path]:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted temporary file: {file_path}")

app = FastAPI(lifespan=lifespan)

@app.get("/")
def hello_world():
    return {"message": "Hello, World!"}

@app.post("/transcribe/file")
async def transcribe_file(file: UploadFile = File(...), model: str = Query(...)):
    global temp_audio_path, temp_wav_path

    with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_path = temp_audio_file.name
        temp_audio_file.write(file.file.read())

    with NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
        temp_wav_path = temp_wav_file.name

    convert_to_wav(temp_audio_path, temp_wav_path)

    try:
        transcription_result = transcribe(temp_wav_path, model)
        return JSONResponse(
            status_code=200,
            content={"transcription": transcription_result}
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
