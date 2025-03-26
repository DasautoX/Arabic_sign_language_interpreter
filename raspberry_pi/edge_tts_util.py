import re
import asyncio
import edge_tts
import io
from pydub import AudioSegment
import pyaudio

def is_arabic_text(text: str) -> bool:
    """
    Check if there's at least one Arabic character in the text.
    If found, we'll treat the text as Arabic.
    """
    return bool(re.search(r'[\u0600-\u06FF]', text))

async def speak_edge_tts(text: str):
    """
    Auto-detects if text is Arabic or English,
    then streams audio via edge-tts and plays it with PyAudio.
    """

    if not text.strip():
        print("Nothing to speak.")
        return

    # Choose voice based on language detection
    if is_arabic_text(text):
        voice = "ar-SA-HamedNeural"  # or "ar-SA-ZariyahNeural", etc.
    else:
        voice = "en-US-AriaNeural"   # choose any English neural voice

    # Rate and volume can be adjusted
    rate = "+0%"
    volume = "+0%"

    # 1) Start edge-tts streaming
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate, volume=volume)
    
    # 2) Accumulate MP3 data
    mp3_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_data += chunk["data"]

    # 3) Decode the MP3 data to raw PCM audio
    audio_segment = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")
    raw_data = audio_segment.raw_data
    sample_width = audio_segment.sample_width
    frame_rate = audio_segment.frame_rate
    channels = audio_segment.channels

    # 4) Play audio via PyAudio
    p = pyaudio.PyAudio()
    audio_stream = p.open(
        format=p.get_format_from_width(sample_width),
        channels=channels,
        rate=frame_rate,
        output=True
    )

    audio_stream.write(raw_data)
    audio_stream.stop_stream()
    audio_stream.close()
    p.terminate()

def speak(text: str):
    """
    A synchronous 'wrapper' to make it easier
    to call speak_edge_tts without manually doing asyncio.run().
    """
    asyncio.run(speak_edge_tts(text))

if __name__ == "__main__":
    # Quick test
    speak("Hello, this is a test in English.")
    speak("مرحباً، هذا اختبار باللغة العربية.")
