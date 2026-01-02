import os
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# --------------------------------------------------
# ENV
# --------------------------------------------------
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# --------------------------------------------------
# APP
# --------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# MODELS
# --------------------------------------------------
class TranslateRequest(BaseModel):
    text: str
    target_lang: str

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def clean_translation(text: str) -> str:
    bad_phrases = [
        "the source language is",
        "source language",
        "translation:",
        "translated text:",
    ]

    cleaned = text.strip()
    for bad in bad_phrases:
        cleaned = cleaned.replace(bad, "")

    return cleaned.strip()

# --------------------------------------------------
# TEXT TRANSLATION
# --------------------------------------------------
@app.post("/translate")
def translate_text(req: TranslateRequest):
    prompt = f"""
You are a professional translation engine.

Task:
- Detect the source language automatically.
- Translate the text into {req.target_lang}.

Rules:
- Output ONLY the translated text.
- Do NOT mention the source language.
- Do NOT add explanations.
- Do NOT add labels or prefixes.
- Do NOT repeat the input.

Text:
{req.text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a translation engine."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    translation = response.choices[0].message.content
    translation = clean_translation(translation)

    return {
        "translation": translation
    }

# --------------------------------------------------
# AUDIO → TRANSCRIPT + TRANSLATION
# --------------------------------------------------
@app.post("/translate-audio")
async def translate_audio(
    file: UploadFile = File(...),
    target_lang: str = Query(...)
):
    # save temp audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name

    try:
        # 1️⃣ SPEECH → TEXT
        with open(temp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="gpt-4o-transcribe",
            )

        transcript_text = transcription.text.strip()

        # 2️⃣ TRANSLATION
        prompt = f"""
You are a professional translation engine.

Task:
- Detect the source language automatically.
- Translate the text into {target_lang}.

Rules:
- Output ONLY the translated text.
- Do NOT mention the source language.
- Do NOT add explanations.
- Do NOT add labels or prefixes.
- Do NOT repeat the input.

Text:
{transcript_text}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a translation engine."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        translation = response.choices[0].message.content
        translation = clean_translation(translation)

        # 🔑 ΕΠΙΣΤΡΕΦΟΥΜΕ ΚΑΙ ΤΑ ΔΥΟ
        return {
            "transcript": transcript_text,
            "translation": translation,
        }

    finally:
        os.remove(temp_path)
