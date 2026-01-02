import os
from openai import OpenAI
from dotenv import load_dotenv
from pydub import AudioSegment

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def translate_text(text: str, target_language: str) -> str:
    """
    Auto-detect source language and translate to target_language
    """
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
{text}
"""


    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professional translator."},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content.strip()


def normalize_audio_to_wav(input_path: str) -> str:
    """
    Convert any audio file to WAV (16kHz mono)
    """
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)

    wav_path = input_path.rsplit(".", 1)[0] + ".wav"
    audio.export(wav_path, format="wav")

    return wav_path


def speech_to_text(audio_file_path: str) -> str:
    wav_path = normalize_audio_to_wav(audio_file_path)

    with open(wav_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            file=audio_file,
            model="gpt-4o-transcribe"
        )

    return transcript.text
