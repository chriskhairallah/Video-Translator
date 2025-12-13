# Video Translation & Dubbing Application

A comprehensive Python web application built with Streamlit that processes uploaded videos, performs speech-to-text transcription, translates the content, and generates a dubbed video with synchronized translated audio.

## Features

- 🎬 **Video Upload**: Support for MP4, MOV, AVI, and MKV formats
- 🎤 **Speech-to-Text**: Accurate transcription with timestamps using Whisper
- 🌍 **Translation**: Multi-language translation using Google Translate (free)
- ✏️ **Editable Scripts**: Review and edit translated text before dubbing
- 🔊 **Text-to-Speech**: Generate natural-sounding dubbed audio using gTTS
- 🎥 **Video Output**: Combine original video with translated audio track
- 📝 **Subtitles**: Optional subtitle generation with customizable fonts and sizes
- 🎙️ **Voice Selection**: Choose from multiple voice options for dubbing

## Prerequisites

### Required Software

1. **Python 3.10+**
2. **FFmpeg** - Required for video/audio processing
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt-get install ffmpeg` or `sudo yum install ffmpeg`
   - **Windows**: Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)

### Optional

- **Google Translate** - Free translation service (no API key required)

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd video-translation-app
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

2. **Follow the 4-phase workflow:**

   **Phase 1: Video Upload & Transcription**
   - Upload your video file
   - Click "Transcribe Video" to extract audio and generate transcript with timestamps

   **Phase 2: Translation & Script Editing**
   - Select target language from the dropdown
   - Click "Translate Script" to translate all segments
   - Review and edit the translated text in the editable table

   **Phase 3 & 4: Generate Video**
   - Choose output options: Dubbing, Subtitles, or both
   - Customize subtitle font size and family (if subtitles enabled)
   - Select voice for dubbing (if dubbing enabled)
   - Preview voice before generating
   - Click "Generate Video" to create final output
   - Preview and download the translated video

## Supported Languages

The application supports translation and dubbing to:
- French, Spanish, German, Italian, Portuguese
- Japanese, Korean, Chinese, Russian, Arabic
- Hindi, Dutch, Polish, Turkish

## Technical Details

### Architecture

- **UI Framework**: Streamlit
- **STT Engine**: Whisper (via whisper-timestamped)
- **Translation**: Google Translate via deep-translator (free)
- **TTS Engine**: Google Text-to-Speech (gTTS)
- **Audio Processing**: pydub
- **Video Processing**: FFmpeg (via subprocess)

### Pipeline Flow

1. **Video → Audio Extraction**: FFmpeg extracts audio track
2. **Audio → Transcript**: Whisper generates timestamped transcript
3. **Transcript → Translation**: Google Translate translates each segment
4. **Translation → TTS**: gTTS generates audio for each segment
5. **TTS → Synchronized Audio**: pydub stitches and aligns audio clips
6. **Audio + Video → Final Output**: FFmpeg combines video and new audio/subtitles

### Key Functions

- `extract_audio_from_video()`: Extracts audio using FFmpeg
- `transcribe_audio_with_timestamps()`: Whisper transcription with timestamps
- `translate_text_with_llm()`: Google Translate translation function
- `synthesize_audio()`: gTTS text-to-speech generation
- `create_dubbed_audio()`: Synchronizes TTS clips with timestamps
- `combine_video_audio()`: Muxes video and audio tracks with optional subtitles

## Deployment

### Streamlit Cloud

This app is configured for deployment on Streamlit Cloud:

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with main file: `app.py`

The `packages.txt` file ensures FFmpeg is installed on Streamlit Cloud.

## Troubleshooting

### FFmpeg Not Found
- Ensure FFmpeg is installed and available in your PATH
- Verify installation: `ffmpeg -version`

### Whisper Model Loading Issues
- First run will download the Whisper model (~150MB for "base" model)
- Ensure stable internet connection for initial download

### Translation Issues
- Google Translate is free but may have rate limits
- The app includes retry logic for rate limiting
- If issues persist, wait a few minutes and try again

### Audio Synchronization
- If audio seems out of sync, the TTS may be too long/short for the time slot
- The app automatically adjusts playback speed to fit timestamps

## Limitations

- gTTS has rate limits for free usage
- Large videos may take significant processing time
- Audio quality depends on gTTS (consider upgrading to premium TTS services)
- Streamlit Cloud free tier has resource limits (1GB RAM)

## Future Enhancements

- Support for premium TTS services (ElevenLabs, Azure Speech)
- Batch processing for multiple videos
- Advanced audio post-processing (noise reduction, normalization)
- Cloud storage integration

## License

This project is provided as-is for educational and development purposes.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.
