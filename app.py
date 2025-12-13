import streamlit as st
import os
import tempfile
import subprocess
import json
from pathlib import Path
import whisper
from whisper_timestamped import load_model, transcribe_timestamped
from gtts import gTTS
import edge_tts
import asyncio
from pydub import AudioSegment
import pandas as pd
from typing import List, Dict, Tuple, Optional
from deep_translator import GoogleTranslator
from io import BytesIO
import time

# Page configuration
st.set_page_config(
    page_title="Video Translation & Dubbing App",
    page_icon="🎬",
    layout="wide"
)

# Initialize session state
if 'transcript_data' not in st.session_state:
    st.session_state.transcript_data = None
if 'translated_script' not in st.session_state:
    st.session_state.translated_script = None
if 'original_video_path' not in st.session_state:
    st.session_state.original_video_path = None
if 'whisper_model' not in st.session_state:
    st.session_state.whisper_model = None
if 'detected_language' not in st.session_state:
    st.session_state.detected_language = None
if 'edge_tts_voices' not in st.session_state:
    st.session_state.edge_tts_voices = None

# Language codes for translation and TTS
LANGUAGE_CODES = {
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh",
    "Russian": "ru",
    "Arabic": "ar",
    "Hindi": "hi",
    "Dutch": "nl",
    "Polish": "pl",
    "Turkish": "tr"
}

# Voice options for different languages (gTTS language variants)
# Format: "Voice Name": {"lang": "language_code", "tld": "tld_code"}
# Different TLDs can provide slight voice variations
VOICE_OPTIONS = {
    "French": {
        "Default French (Standard)": {"lang": "fr", "tld": "com"},
        "French (Canada)": {"lang": "fr-ca", "tld": "ca"},
        "French (France - Standard)": {"lang": "fr", "tld": "fr"},
        "French (Belgium)": {"lang": "fr", "tld": "be"},
        "French (Switzerland)": {"lang": "fr", "tld": "ch"},
        "French (Canada - Alternative)": {"lang": "fr-ca", "tld": "com"},
        "French (European)": {"lang": "fr", "tld": "co.uk"}
    },
    "Spanish": {
        "Default Spanish": {"lang": "es", "tld": "com"},
        "Spanish (Mexico)": {"lang": "es-mx", "tld": "com.mx"},
        "Spanish (Spain)": {"lang": "es-es", "tld": "es"}
    },
    "Portuguese": {
        "Default Portuguese": {"lang": "pt", "tld": "com"},
        "Portuguese (Brazil)": {"lang": "pt-br", "tld": "com.br"},
        "Portuguese (Portugal)": {"lang": "pt-pt", "tld": "pt"}
    },
    "English": {
        "US English": {"lang": "en-us", "tld": "com"},
        "UK English": {"lang": "en-gb", "tld": "co.uk"},
        "Australian English": {"lang": "en-au", "tld": "com.au"},
        "Canadian English": {"lang": "en-ca", "tld": "ca"}
    },
    "Chinese": {
        "Mandarin (Simplified)": {"lang": "zh-cn", "tld": "com"},
        "Mandarin (Traditional)": {"lang": "zh-tw", "tld": "com"}
    }
}

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def extract_audio_from_video(video_path: str, output_audio_path: str) -> bool:
    """Extract audio from video using FFmpeg"""
    try:
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            '-y', output_audio_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"FFmpeg error: {e.stderr.decode()}")
        return False

def transcribe_audio_with_timestamps(audio_path: str, language: str = None) -> List[Dict]:
    """Transcribe audio using Whisper with timestamps"""
    try:
        # Load model if not already loaded
        if st.session_state.whisper_model is None:
            with st.spinner("Loading Whisper model (this may take a moment)..."):
                st.session_state.whisper_model = load_model("base", device="cpu")
        
        # Transcribe with timestamps
        with st.spinner("Transcribing audio..."):
            result = transcribe_timestamped(
                st.session_state.whisper_model,
                audio_path,
                language=language,
                verbose=False
            )
        
        # Detect and store the language
        detected_lang_code = result.get('language', 'en')
        # Map Whisper language codes to our language names
        whisper_to_lang_name = {
            'en': 'English', 'fr': 'French', 'es': 'Spanish', 'de': 'German',
            'it': 'Italian', 'pt': 'Portuguese', 'ja': 'Japanese', 'ko': 'Korean',
            'zh': 'Chinese', 'ru': 'Russian', 'ar': 'Arabic', 'hi': 'Hindi',
            'nl': 'Dutch', 'pl': 'Polish', 'tr': 'Turkish'
        }
        st.session_state.detected_language = whisper_to_lang_name.get(detected_lang_code, 'English')
        
        # Extract segments with timestamps
        segments = []
        for segment in result.get('segments', []):
            segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip()
            })
        
        return segments
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return []

def translate_text_with_llm(text: str, target_language: str, source_language: str = "English") -> str:
    """
    Translate text using Google Translate (free) via deep-translator
    This is a modular function that can be replaced with other translation services
    """
    try:
        # Normalize language names for comparison (case-insensitive)
        source_lang_normalized = source_language.strip().title() if source_language else "English"
        target_lang_normalized = target_language.strip().title() if target_language else "English"
        
        # If source and target languages are the same, return original text
        # This allows users to edit English text or add English subtitles
        if source_lang_normalized == target_lang_normalized:
            return text.strip()
        
        # Map language names to Google Translate language codes
        lang_code_map = {
            "French": "fr", "Spanish": "es", "German": "de", "Italian": "it",
            "Portuguese": "pt", "Japanese": "ja", "Korean": "ko", "Chinese": "zh-cn",
            "Russian": "ru", "Arabic": "ar", "Hindi": "hi", "Dutch": "nl",
            "Polish": "pl", "Turkish": "tr", "English": "en"
        }
        
        # Use normalized language names for lookup
        target_code = lang_code_map.get(target_lang_normalized, target_lang_normalized.lower()[:2])
        source_code = lang_code_map.get(source_lang_normalized, "en")
        
        # Translate with retry logic for rate limiting
        max_retries = 3
        for attempt in range(max_retries):
            try:
                translator = GoogleTranslator(source=source_code, target=target_code)
                result = translator.translate(text)
                return result.strip()
            except Exception as e:
                if attempt < max_retries - 1:
                    # Wait before retrying (exponential backoff)
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise e
                    
    except Exception as e:
        st.warning(f"Translation error: {str(e)}")
        # Return original text if translation fails
        return text

def translate_segments(segments: List[Dict], target_language: str, source_language: str = "English") -> List[Dict]:
    """Translate all segments"""
    translated = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, segment in enumerate(segments):
        status_text.text(f"Translating segment {i+1}/{len(segments)}...")
        translated_text = translate_text_with_llm(segment['text'], target_language, source_language)
        translated.append({
            'start': segment['start'],
            'end': segment['end'],
            'original_text': segment['text'],
            'translated_text': translated_text
        })
        progress_bar.progress((i + 1) / len(segments))
    
    status_text.empty()
    progress_bar.empty()
    
    return translated

async def get_edge_tts_voices():
    """Get all available Edge TTS voices (cached)"""
    if st.session_state.edge_tts_voices is None:
        try:
            voices = await edge_tts.list_voices()
            st.session_state.edge_tts_voices = voices
        except Exception as e:
            st.warning(f"Failed to load Edge TTS voices: {str(e)}")
            return []
    return st.session_state.edge_tts_voices

def get_edge_voices_for_language(language_code: str) -> List[Dict]:
    """Get Edge TTS voices for a specific language"""
    try:
        voices = asyncio.run(get_edge_tts_voices())
        if not voices:
            return []
        
        # Map language codes to Edge TTS language format
        lang_map = {
            "en": "en-", "fr": "fr-", "es": "es-", "de": "de-", "it": "it-",
            "pt": "pt-", "ja": "ja-", "ko": "ko-", "zh": "zh-", "ru": "ru-",
            "ar": "ar-", "hi": "hi-", "nl": "nl-", "pl": "pl-", "tr": "tr-"
        }
        
        lang_prefix = lang_map.get(language_code.lower(), language_code.lower() + "-")
        
        # Filter voices by language
        matching_voices = [
            v for v in voices 
            if v["Locale"].lower().startswith(lang_prefix.lower())
        ]
        
        # Sort by gender and name for consistency
        matching_voices.sort(key=lambda x: (x.get("Gender", ""), x.get("ShortName", "")))
        
        return matching_voices
    except Exception as e:
        st.warning(f"Error getting Edge TTS voices: {str(e)}")
        return []

async def synthesize_audio_edge_tts(text: str, voice: str, output_path: str) -> bool:
    """Generate TTS audio using Edge TTS"""
    try:
        communicate = edge_tts.Communicate(text, voice)
        # Edge TTS outputs webm by default, but we need MP3 for pydub
        # Save to temporary webm first, then convert to MP3
        temp_webm = output_path.replace('.mp3', '.webm')
        await communicate.save(temp_webm)
        
        # Convert webm to mp3 using pydub (requires ffmpeg)
        audio = AudioSegment.from_file(temp_webm, format="webm")
        audio.export(output_path, format="mp3")
        
        # Clean up temp file
        if os.path.exists(temp_webm):
            os.remove(temp_webm)
        
        return True
    except Exception as e:
        st.error(f"Edge TTS error: {str(e)}")
        # Clean up on error
        if os.path.exists(output_path.replace('.mp3', '.webm')):
            os.remove(output_path.replace('.mp3', '.webm'))
        return False

def synthesize_audio(text: str, language_code: str, output_path: str, tld: str = "com", 
                     tts_provider: str = "edge", voice: Optional[str] = None) -> bool:
    """Generate TTS audio using specified provider"""
    if tts_provider == "edge":
        if voice is None:
            # Get default voice for language
            voices = get_edge_voices_for_language(language_code)
            if not voices:
                st.warning(f"No Edge TTS voices found for {language_code}, falling back to gTTS")
                tts_provider = "gtts"
            else:
                voice = voices[0]["ShortName"]
        
        if voice:
            try:
                # Run async function
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(synthesize_audio_edge_tts(text, voice, output_path))
                loop.close()
                return result
            except Exception as e:
                st.warning(f"Edge TTS failed: {str(e)}, falling back to gTTS")
                tts_provider = "gtts"
    
    # Fallback to gTTS
    if tts_provider == "gtts":
        try:
            tts = gTTS(text=text, lang=language_code, slow=False, tld=tld)
            tts.save(output_path)
            return True
        except Exception as e:
            st.error(f"gTTS error: {str(e)}")
            return False
    
    return False

async def preview_voice_edge_tts(text: str, voice: str) -> bytes:
    """Generate a preview audio sample using Edge TTS"""
    try:
        communicate = edge_tts.Communicate(text, voice)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        # Edge TTS outputs webm/opus, convert to MP3 for compatibility
        if audio_data:
            # Save to temp file, convert, then return
            with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_webm:
                temp_webm.write(audio_data)
                temp_webm_path = temp_webm.name
            
            # Convert to MP3
            audio = AudioSegment.from_file(temp_webm_path, format="webm")
            mp3_buffer = BytesIO()
            audio.export(mp3_buffer, format="mp3")
            mp3_buffer.seek(0)
            result = mp3_buffer.getvalue()
            
            # Cleanup
            os.remove(temp_webm_path)
            return result
        return None
    except Exception as e:
        st.error(f"Edge TTS preview error: {str(e)}")
        return None

def preview_voice(text: str, language_code: str, tld: str = "com", 
                 tts_provider: str = "edge", voice: Optional[str] = None) -> bytes:
    """Generate a preview audio sample for voice selection"""
    if tts_provider == "edge":
        if voice is None:
            voices = get_edge_voices_for_language(language_code)
            if not voices:
                tts_provider = "gtts"
            else:
                voice = voices[0]["ShortName"]
        
        if voice:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(preview_voice_edge_tts(text, voice))
                loop.close()
                if result:
                    return result
            except Exception as e:
                st.warning(f"Edge TTS preview failed: {str(e)}, using gTTS")
                tts_provider = "gtts"
    
    # Fallback to gTTS
    if tts_provider == "gtts":
        try:
            tts = gTTS(text=text, lang=language_code, slow=False, tld=tld)
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            return audio_buffer.getvalue()
        except Exception as e:
            st.error(f"gTTS preview error: {str(e)}")
            return None
    
    return None

def create_dubbed_audio(script: List[Dict], language_code: str, total_duration: float, output_path: str, 
                       tld: str = "com", tts_provider: str = "edge", voice: Optional[str] = None) -> bool:
    """Create synchronized dubbed audio track"""
    try:
        # Create silent audio track for the full duration
        silence = AudioSegment.silent(duration=int(total_duration * 1000))
        
        with st.spinner("Generating audio clips..."):
            temp_dir = tempfile.mkdtemp()
            audio_clips = []
            
            for i, segment in enumerate(script):
                start_time = segment['start']
                end_time = segment['end']
                text = segment['translated_text']
                
                if not text or text.strip() == "":
                    continue
                
                # Generate TTS for this segment
                # Both Edge TTS and gTTS can output MP3
                temp_audio_path = os.path.join(temp_dir, f"segment_{i}.mp3")
                if not synthesize_audio(text, language_code, temp_audio_path, tld, tts_provider, voice):
                    continue
                
                # Load the generated audio (both providers output MP3)
                audio_clip = AudioSegment.from_mp3(temp_audio_path)
                
                # Calculate duration and adjust if needed
                segment_duration = (end_time - start_time) * 1000  # Convert to milliseconds
                
                # Speed up or slow down if needed to fit the time slot
                if len(audio_clip) > segment_duration:
                    # Speed up to fit
                    speed_factor = len(audio_clip) / segment_duration
                    audio_clip = audio_clip.speedup(playback_speed=speed_factor)
                elif len(audio_clip) < segment_duration * 0.8:
                    # Slow down if too short (optional, can also pad with silence)
                    speed_factor = len(audio_clip) / (segment_duration * 0.8)
                    audio_clip = audio_clip.speedup(playback_speed=speed_factor)
                
                # Trim to exact duration
                if len(audio_clip) > segment_duration:
                    audio_clip = audio_clip[:int(segment_duration)]
                
                # Place at the correct timestamp
                start_ms = int(start_time * 1000)
                silence = silence.overlay(audio_clip, position=start_ms)
            
            # Export the final audio
            silence.export(output_path, format="wav")
            
            # Cleanup
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)
        
        return True
    except Exception as e:
        st.error(f"Audio creation error: {str(e)}")
        return False

def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def generate_srt_subtitles(script: List[Dict], output_path: str) -> bool:
    """Generate SRT subtitle file from translated script"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(script, 1):
                start_time = format_timestamp(segment['start'])
                end_time = format_timestamp(segment['end'])
                text = segment['translated_text'].strip()
                
                if not text:
                    continue
                
                # Escape special SRT characters
                text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")
        
        return True
    except Exception as e:
        st.error(f"SRT generation error: {str(e)}")
        return False

def combine_video_audio(video_path: str, audio_path: str, output_path: str, subtitle_path: str = None, subtitle_font_size: int = 18, subtitle_font_family: str = "Arial") -> bool:
    """Combine video with optional new audio track and/or subtitles using FFmpeg"""
    try:
        # Build filter complex for subtitles if needed
        video_filters = []
        if subtitle_path and os.path.exists(subtitle_path):
            # Escape the subtitle path for FFmpeg (handle spaces and special characters)
            # Use absolute path and escape single quotes
            abs_subtitle_path = os.path.abspath(subtitle_path).replace("'", "'\\''")
            
            # Burn subtitles into video using FFmpeg subtitles filter
            # Style: White text, black outline, bottom center, customizable font size and family
            # MarginV controls distance from bottom (30px default)
            subtitle_filter = f"subtitles='{abs_subtitle_path}':force_style='FontName={subtitle_font_family},FontSize={subtitle_font_size},PrimaryColour=&Hffffff,OutlineColour=&H000000,Outline=2,Shadow=1,Alignment=2,MarginV=30'"
            video_filters.append(subtitle_filter)
        
        # Build FFmpeg command
        cmd = ['ffmpeg', '-i', video_path]
        
        # Add audio input if dubbing is enabled
        if audio_path:
            cmd.extend(['-i', audio_path])
        
        # Add video filter if subtitles are enabled
        if video_filters:
            cmd.extend(['-vf', ','.join(video_filters)])
        else:
            # No filters, copy video codec (no re-encoding)
            cmd.extend(['-c:v', 'copy'])
        
        # Handle audio
        if audio_path:
            # Use new dubbed audio
            cmd.extend(['-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0'])
        else:
            # Keep original audio
            cmd.extend(['-c:a', 'copy', '-map', '0'])
        
        cmd.extend(['-shortest', '-y', output_path])
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if isinstance(e.stderr, str) else e.stderr.decode('utf-8', errors='ignore')
        st.error(f"FFmpeg error: {error_msg}")
        return False

# Main UI
st.title("🎬 Video Translation & Dubbing Application")
st.markdown("Upload a video, transcribe it, translate the script, and generate a dubbed version!")

# Check FFmpeg installation
if not check_ffmpeg():
    st.error("❌ FFmpeg is not installed. Please install FFmpeg to use this application.")
    st.markdown("""
    **Installation instructions:**
    - **macOS**: `brew install ffmpeg`
    - **Linux**: `sudo apt-get install ffmpeg` or `sudo yum install ffmpeg`
    - **Windows**: Download from https://ffmpeg.org/download.html
    """)
    st.stop()

# Phase 1: Video Upload & Transcription
st.header("Phase 1: Video Upload & Transcription")
uploaded_file = st.file_uploader(
    "Upload a video file",
    type=['mp4', 'mov', 'avi', 'mkv'],
    help="Supported formats: MP4, MOV, AVI, MKV"
)

if uploaded_file is not None:
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        st.session_state.original_video_path = tmp_file.name
    
    st.success(f"✅ Video uploaded: {uploaded_file.name}")
    
    if st.button("🎤 Transcribe Video", type="primary"):
        with st.spinner("Processing video..."):
            # Extract audio
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_audio.close()
            
            if extract_audio_from_video(st.session_state.original_video_path, temp_audio.name):
                # Transcribe
                segments = transcribe_audio_with_timestamps(temp_audio.name)
                
                if segments:
                    st.session_state.transcript_data = segments
                    st.success(f"✅ Transcription complete! Found {len(segments)} segments.")
                    
                    # Display transcript
                    st.subheader("Transcript")
                    transcript_df = pd.DataFrame(segments)
                    st.dataframe(transcript_df, use_container_width=True)
                else:
                    st.error("❌ Transcription failed. Please try again.")
                
                # Cleanup
                os.unlink(temp_audio.name)
            else:
                st.error("❌ Failed to extract audio from video.")

# Phase 2: Translation & Script Editing
if st.session_state.transcript_data:
    st.header("Phase 2: Translation & Script Editing")
    
    # Show detected source language
    source_language = st.session_state.detected_language or "English"
    if st.session_state.detected_language:
        st.info(f"🎤 Detected source language: **{source_language}**")
    
    # Always include all languages, including English, so users can:
    # - Edit English text (translate English to English)
    # - Add English subtitles
    # - Change words in English videos
    available_languages = list(LANGUAGE_CODES.keys())
    
    # Set default index to English if detected language is English, otherwise use first language
    default_index = available_languages.index("English") if "English" in available_languages else 0
    
    target_language = st.selectbox(
        "Select Target Language",
        options=available_languages,
        index=default_index if source_language == "English" else 0,
        help="Choose the language for translation and dubbing. You can select English even if the video is in English to edit text or add subtitles."
    )
    
    if st.button("🌍 Translate Script", type="primary"):
        with st.spinner("Translating script..."):
            translated = translate_segments(st.session_state.transcript_data, target_language, source_language)
            st.session_state.translated_script = translated
            st.success("✅ Translation complete!")
    
    if st.session_state.translated_script:
        st.subheader("Edit Translated Script")
        st.markdown("You can edit the translated text and timestamps in the table below. Adjust timing or text as needed.")
        
        # Create editable dataframe
        script_df = pd.DataFrame(st.session_state.translated_script)
        # Keep start and end as numeric values for editing
        display_df = script_df[['start', 'end', 'original_text', 'translated_text']].copy()
        display_df.columns = ['Start (seconds)', 'End (seconds)', 'Original Text', 'Translated Text']
        
        # Use st.data_editor for editable table
        edited_df = st.data_editor(
            display_df,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "Start (seconds)": st.column_config.NumberColumn("Start (seconds)", min_value=0.0, step=0.1, format="%.2f"),
                "End (seconds)": st.column_config.NumberColumn("End (seconds)", min_value=0.0, step=0.1, format="%.2f"),
                "Original Text": st.column_config.TextColumn("Original Text", disabled=True),
                "Translated Text": st.column_config.TextColumn("Translated Text", width="large")
            }
        )
        
        # Update session state with edited values
        for i in range(len(edited_df)):
            # Update timestamps if changed
            new_start = float(edited_df.iloc[i]['Start (seconds)'])
            new_end = float(edited_df.iloc[i]['End (seconds)'])
            if new_start != st.session_state.translated_script[i]['start'] or new_end != st.session_state.translated_script[i]['end']:
                st.session_state.translated_script[i]['start'] = new_start
                st.session_state.translated_script[i]['end'] = new_end
            
            # Update translated text if changed
            new_text = edited_df.iloc[i]['Translated Text']
            if new_text != st.session_state.translated_script[i]['translated_text']:
                st.session_state.translated_script[i]['translated_text'] = new_text

# Phase 3 & 4: Generate Video
if st.session_state.translated_script and st.session_state.original_video_path:
    st.header("Phase 3 & 4: Generate Video")
    
    st.subheader("Output Options")
    
    # Separate options for dubbing and subtitles
    add_dubbing = st.checkbox(
        "🔊 Add Dubbed Audio",
        value=False,
        help="Replace the original audio with translated TTS audio"
    )
    
    # Voice selection for dubbing (initialize defaults)
    selected_voice_code = LANGUAGE_CODES.get(target_language, "en")
    selected_tld = "com"
    selected_tts_provider = "edge"  # Default to Edge TTS for better quality
    selected_edge_voice = None
    
    if add_dubbing:
        st.markdown("### TTS Provider & Voice Selection")
        
        # TTS Provider selection
        tts_provider = st.selectbox(
            "TTS Provider",
            options=["edge", "gtts"],
            index=0,
            format_func=lambda x: "Edge TTS (Microsoft) - Recommended" if x == "edge" else "Google TTS (gTTS) - Basic",
            help="Edge TTS provides much more natural-sounding voices and is completely free. gTTS is a fallback option."
        )
        selected_tts_provider = tts_provider
        
        if tts_provider == "edge":
            # Edge TTS voice selection
            lang_code = LANGUAGE_CODES.get(target_language, "en")
            edge_voices = get_edge_voices_for_language(lang_code)
            
            if edge_voices:
                # Create voice display names
                voice_options = []
                voice_map = {}
                for voice in edge_voices:
                    gender = voice.get("Gender", "Unknown")
                    locale = voice.get("Locale", "Unknown")
                    name = voice.get("ShortName", "")
                    friendly_name = f"{voice.get('FriendlyName', name)} ({gender}, {locale})"
                    voice_options.append(friendly_name)
                    voice_map[friendly_name] = voice["ShortName"]
                
                selected_voice_display = st.selectbox(
                    "Choose Voice",
                    options=voice_options,
                    index=0,
                    help="Select a voice for dubbing. Edge TTS offers high-quality, natural-sounding voices."
                )
                selected_edge_voice = voice_map[selected_voice_display]
            else:
                st.warning(f"No Edge TTS voices found for {target_language}. Falling back to gTTS.")
                selected_tts_provider = "gtts"
        else:
            # gTTS voice selection (original logic)
            if target_language in VOICE_OPTIONS:
                voice_options = VOICE_OPTIONS[target_language]
                selected_voice_name = st.selectbox(
                    "Choose Voice",
                    options=list(voice_options.keys()),
                    index=0,
                    help="Select a voice for the dubbing. Different options may have slight variations in accent or tone."
                )
                voice_config = voice_options[selected_voice_name]
                if isinstance(voice_config, dict):
                    selected_voice_code = voice_config["lang"]
                    selected_tld = voice_config["tld"]
                else:
                    selected_voice_code = voice_config
                    selected_tld = "com"
            else:
                st.info(f"Using default voice for {target_language}")
                selected_voice_code = LANGUAGE_CODES.get(target_language, "en")
                selected_tld = "com"
        
        # Voice preview
        if st.session_state.translated_script and len(st.session_state.translated_script) > 0:
            preview_text = st.session_state.translated_script[0]['translated_text'][:100]
            if len(st.session_state.translated_script[0]['translated_text']) > 100:
                preview_text += "..."
        else:
            preview_text = "Hello, this is a voice preview."
        
        col1, col2 = st.columns([3, 1])
        with col1:
            preview_text_input = st.text_input(
                "Preview Text",
                value=preview_text,
                help="Text to use for voice preview"
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            if st.button("🔊 Preview Voice", use_container_width=True):
                with st.spinner("Generating preview..."):
                    if selected_tts_provider == "edge" and selected_edge_voice:
                        preview_audio = preview_voice(preview_text_input, selected_voice_code, selected_tld, 
                                                     selected_tts_provider, selected_edge_voice)
                    else:
                        preview_audio = preview_voice(preview_text_input, selected_voice_code, selected_tld, 
                                                     selected_tts_provider)
                    if preview_audio:
                        st.audio(preview_audio, format="audio/mpeg", autoplay=False)
                        st.success("✅ Voice preview ready! Click play to hear it.")
    
    add_subtitles = st.checkbox(
        "📝 Add Subtitles",
        value=False,
        help="Burn translated subtitles directly into the video"
    )
    
    # Subtitle customization options
    subtitle_font_size = 18
    subtitle_font_family = "Arial"
    if add_subtitles:
        st.markdown("### Subtitle Customization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            subtitle_font_size = st.slider(
                "Font Size",
                min_value=12,
                max_value=48,
                value=18,
                step=2,
                help="Size of the subtitle text in pixels"
            )
        
        with col2:
            subtitle_font_family = st.selectbox(
                "Font Family",
                options=["Arial", "Helvetica", "Times New Roman", "Courier New", "Verdana", "Georgia", "Comic Sans MS"],
                index=0,
                help="Font family for the subtitles"
            )
        
        # Preview of subtitle style
        st.markdown("#### Preview:")
        preview_text = "Sample Subtitle Text" if not st.session_state.translated_script else st.session_state.translated_script[0]['translated_text'][:50] + "..."
        
        # FFmpeg font sizes are relative to video resolution (typically 1080p)
        # For accurate preview, we need to account for how videos are displayed
        # Videos are often displayed larger than their pixel dimensions, making subtitles appear bigger
        # Increasing the preview size to better match the actual video subtitle appearance
        preview_font_size = int(subtitle_font_size * 1.2)  # Scale up to better match video display size
        
        # Create HTML preview with the selected style
        # Using a container that simulates video aspect ratio for better accuracy
        preview_html = f"""
        <div style="
            background: linear-gradient(to bottom, #1a1a1a 0%, #2d2d2d 100%);
            padding: 40px 20px;
            border-radius: 8px;
            text-align: center;
            margin: 20px 0;
            position: relative;
            min-height: 200px;
            display: flex;
            align-items: flex-end;
            justify-content: center;
            padding-bottom: 30px;
        ">
            <p style="
                color: white;
                font-family: '{subtitle_font_family}', sans-serif;
                font-size: {preview_font_size}px;
                text-shadow: 
                    -1px -1px 0 #000,
                    1px -1px 0 #000,
                    -1px 1px 0 #000,
                    1px 1px 0 #000,
                    0 0 4px rgba(0, 0, 0, 0.8);
                margin: 0;
                line-height: 1.2;
                font-weight: normal;
            ">{preview_text}</p>
        </div>
        <p style="font-size: 0.85em; color: #666; margin-top: -10px; text-align: center;">
            Preview scaled to approximate video appearance (actual size may vary by video resolution)
        </p>
        """
        st.markdown(preview_html, unsafe_allow_html=True)
    
    # Validation
    if not add_dubbing and not add_subtitles:
        st.info("ℹ️ Please select at least one option: Dubbing or Subtitles (or both)")
    
    if st.button("🎬 Generate Video", type="primary", disabled=(not add_dubbing and not add_subtitles)):
        target_lang_code = LANGUAGE_CODES[target_language]
        
        # Calculate total duration
        total_duration = max([seg['end'] for seg in st.session_state.translated_script])
        
        output_description = []
        if add_dubbing:
            output_description.append("dubbed audio")
        if add_subtitles:
            output_description.append("subtitles")
        desc_text = " and ".join(output_description)
        
        with st.spinner(f"Generating video with {desc_text} (this may take a few minutes)..."):
            # Create temporary files
            temp_audio = None
            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_output.close()
            temp_srt = None
            
            # Step 1: Create dubbed audio if requested
            audio_path = None
            dubbing_success = True
            if add_dubbing:
                temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                temp_audio.close()
                
                if create_dubbed_audio(
                    st.session_state.translated_script,
                    selected_voice_code,
                    total_duration,
                    temp_audio.name,
                    selected_tld,
                    selected_tts_provider,
                    selected_edge_voice
                ):
                    audio_path = temp_audio.name
                    st.success("✅ Audio synthesis complete!")
                else:
                    st.error("❌ Failed to generate dubbed audio.")
                    dubbing_success = False
                    if temp_audio and os.path.exists(temp_audio.name):
                        os.unlink(temp_audio.name)
            
            # Only proceed if dubbing succeeded (or if no dubbing was requested)
            if dubbing_success:
                # Step 2: Generate subtitles if requested
                subtitle_path = None
                if add_subtitles:
                    temp_srt = tempfile.NamedTemporaryFile(delete=False, suffix='.srt', mode='w')
                    temp_srt.close()
                    if generate_srt_subtitles(st.session_state.translated_script, temp_srt.name):
                        subtitle_path = temp_srt.name
                        st.success("✅ Subtitles generated!")
                    else:
                        st.warning("⚠️ Failed to generate subtitles, continuing without them.")
            
                # Step 3: Combine video with audio and/or subtitles
                if combine_video_audio(
                    st.session_state.original_video_path,
                    audio_path,  # Will be None if no dubbing, which means keep original audio
                    temp_output.name,
                    subtitle_path,
                    subtitle_font_size,
                    subtitle_font_family
                ):
                    st.success("✅ Video generation complete!")
                    
                    # Generate filename based on options
                    filename_parts = [uploaded_file.name.rsplit('.', 1)[0]]
                    if add_dubbing:
                        filename_parts.append("dubbed")
                    if add_subtitles:
                        filename_parts.append("subtitled")
                    filename_parts.append(target_language)
                    filename = "_".join(filename_parts) + "." + uploaded_file.name.rsplit('.', 1)[1]
                    
                    # Video Preview
                    st.markdown("### 🎥 Video Preview")
                    st.markdown("Preview your generated video below:")
                    
                    # Read video file for preview
                    with open(temp_output.name, 'rb') as video_file:
                        video_bytes = video_file.read()
                    
                    # Display video player
                    st.video(video_bytes, format="video/mp4")
                    
                    st.markdown("---")
                    
                    # Provide download link
                    st.download_button(
                        label="📥 Download Video",
                        data=video_bytes,
                        file_name=filename,
                        mime="video/mp4",
                        type="primary"
                    )
                else:
                    st.error("❌ Failed to generate video.")
                
                # Cleanup
                if temp_audio and os.path.exists(temp_audio.name):
                    os.unlink(temp_audio.name)
                if temp_srt and os.path.exists(temp_srt.name):
                    os.unlink(temp_srt.name)

# Sidebar with instructions
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    1. **Upload Video**: Upload your video file (MP4, MOV, etc.)
    2. **Transcribe**: Click "Transcribe Video" to extract and transcribe audio
    3. **Translate**: Select target language and click "Translate Script"
    4. **Edit**: Review and edit the translated text as needed
    5. **Generate**: Click "Generate Dubbed Video" to create the final output
    6. **Download**: Download your translated and dubbed video
    """)

