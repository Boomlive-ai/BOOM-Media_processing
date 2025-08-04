import openai
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, openai_api_key: str):
        self.client = openai.OpenAI(api_key=openai_api_key)

    def transcribe_audio(self, audio_file_path: str, language: Optional[str] = None) -> Optional[str]:
        """
        Transcribes audio to text using OpenAI Whisper.

        Args:
            audio_file_path (str): Path to the audio file (e.g., mp3, wav, m4a).
            language (Optional[str]): Language code (e.g., 'en' for English), or None to auto-detect.

        Returns:
            Optional[str]: Transcribed text, or None if failed.
        """
        try:
            with open(audio_file_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1",
                    language=language  # None means auto-detect
                )
            return response.text
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None

# Usage example:
# import os
# processor = AudioProcessor(openai_api_key=os.getenv("OPENAI_API_KEY"))
# transcription = processor.transcribe_audio("/path/to/audio.mp3")
# print(transcription)
