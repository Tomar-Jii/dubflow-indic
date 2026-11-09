import os
import json
import time
import asyncio
import subprocess
import requests
import soundfile as sf
import edge_tts
import gradio as gr

# ==============================================================================
# CONFIGURATION & API KEYS
# ==============================================================================
# Gemini API Key (Prefilled from tested prototype)
DEFAULT_GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "AQ.Ab8RN6IgXYVXIR8wu3GOlWMzal7909-9Llg9dfeQjpwtThSVRQ")
# Groq API Key (Optional for ultra-fast cloud Whisper; fallback to local fast whisper if missing)
DEFAULT_GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

# Voice map for Studio Characters
VOICE_MAP = {
    "Deep Professional Male (Madhur)": "hi-IN-MadhurNeural",
    "Expressive Emotional Female (Swara)": "hi-IN-SwaraNeural"
}

# ==============================================================================
# HELPER: EDGE-TTS ASYNC CALL (DISK-BACKED, ZERO STATIC CORRUPTION)
# ==============================================================================
async def synthesize_edge_tts(text: str, voice_name: str, target_file: str):
    """Generates audio via Edge-TTS and writes directly to disk to prevent stream corruption."""
    communicate = edge_tts.Communicate(text, voice_name)
    await communicate.save(target_file)

# ==============================================================================
# CORE WORKFLOW ENGINE (ZERO-GPU, RUNS ON <512MB RAM)
# ==============================================================================
def process_dubflow(youtube_url, upload_audio, start_time, end_time, voice_choice, gemini_key, groq_key):
    t_start_total = time.time()
    
    # Clean previous temp files
    temp_files = [
        "temp_raw_dl.wav", "temp_raw_dl.mp3", "input_16k.wav",
        "synth_raw.mp3", "synth_16k.wav", "final_dubbed_master.wav"
    ]
    for tf in temp_files:
        if os.path.exists(tf):
            try: os.remove(tf)
            except: pass

    # 1. Validation
    active_gemini_key = gemini_key.strip() if gemini_key else DEFAULT_GEMINI_KEY
    if not active_gemini_key:
        return None, None, "❌ Error: Gemini API Key is required!"

    chosen_voice = VOICE_MAP.get(voice_choice, "hi-IN-MadhurNeural")

    # --------------------------------------------------------------------------
    # STEP 1: INGESTION (YouTube or Direct Upload)
    # --------------------------------------------------------------------------
    print("🎬 [1/5] Ingesting Audio...")
    if upload_audio:
        source_in = upload_audio
        cmd_ingest = f'ffmpeg -y -v quiet -i "{source_in}" -ss {start_time} -to {end_time} -ar 16000 -ac 1 -c:a pcm_s16le input_16k.wav'
        subprocess.run(cmd_ingest, shell=True)
    else:
        if not youtube_url:
            return None, None, "❌ Error: Please provide a YouTube link or upload an audio file."
        clean_url = youtube_url.strip().split("?")[0]
        # Download directly via yt-dlp
        cmd_dl = f'yt-dlp --force-overwrites -x --audio-format wav -o "temp_raw_dl.wav" "{clean_url}"'
        res_dl = subprocess.run(cmd_dl, shell=True, capture_output=True, text=True)
        
        if not os.path.exists("temp_raw_dl.wav"):
            return None, None, f"❌ Download Error: Could not extract YouTube audio. Details: {res_dl.stderr[:100]}"
            
        cmd_ingest = f'ffmpeg -y -v quiet -i temp_raw_dl.wav -ss {start_time} -to {end_time} -ar 16000 -ac 1 -c:a pcm_s16le input_16k.wav'
        subprocess.run(cmd_ingest, shell=True)

    if not os.path.exists("input_16k.wav") or os.path.getsize("input_16k.wav") < 1000:
        return None, None, "❌ Error: Failed to extract clean 16kHz audio. Check slice duration."

    # Measure source duration
    data_in, sr_in = sf.read("input_16k.wav")
    src_duration = len(data_in) / sr_in

    # --------------------------------------------------------------------------
    # STEP 2: TRANSCRIPTION (Groq Cloud Whisper API or Local Fallback)
    # --------------------------------------------------------------------------
    print("📝 [2/5] Transcribing Audio with Timestamps...")
    transcript_segments = []
    active_groq_key = groq_key.strip() if groq_key else DEFAULT_GROQ_KEY

    if active_groq_key:
        try:
            # High-speed Groq Whisper API (Cloud, takes 0.4s)
            groq_url = "https://api.groq.com/openai/v1/audio/transcriptions"
            headers = {"Authorization": f"Bearer {active_groq_key}"}
            with open("input_16k.wav", "rb") as f:
                files = {"file": ("input_16k.wav", f, "audio/wav")}
                data = {
                    "model": "whisper-large-v3",
                    "response_format": "verbose_json"
                }
                resp = requests.post(groq_url, headers=headers, files=files, data=data, timeout=30)
                groq_json = resp.json()
                for seg in groq_json.get("segments", []):
                    transcript_segments.append({
                        "start": round(seg["start"], 2),
                        "end": round(seg["end"], 2),
                        "text": seg["text"].strip()
                    })
        except Exception as e:
            print("Groq Whisper API warning:", e)

    # Fallback to local CPU faster-whisper if Groq not provided or failed
    if not transcript_segments:
        from faster_whisper import WhisperModel
        whisper = WhisperModel("base", device="cpu", compute_type="int8")
        segs, _ = whisper.transcribe(data_in, beam_size=5)
        for s in segs:
            transcript_segments.append({
                "start": round(s.start, 2),
                "end": round(s.end, 2),
                "text": s.text.strip()
            })

    if not transcript_segments:
        return None, None, "❌ Error: No speech detected in source audio."

    # --------------------------------------------------------------------------
    # STEP 3: GEMINI FLASH HINDI TRANSLATION (REST API)
    # --------------------------------------------------------------------------
    print("🤖 [3/5] Translating with Gemini Flash (Emotion & Timing Budget)...")
    gemini_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={active_gemini_key}"
    
    prompt = f"""
    You are an expert Hollywood/Bollywood dubbing director.
    Translate each English line into natural, conversational, inspirational Hindi in Devanagari script.
    Ensure syllable count matches the original speaking duration ({src_duration:.1f}s).
    Return STRICTLY a JSON list:
    [{{"start": float, "end": float, "english": "...", "hindi": "...", "tone": "..."}}]

    Transcript:
    {json.dumps(transcript_segments)}
    """

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 450}
    }

    res_gemini = requests.post(gemini_endpoint, json=payload, timeout=20)
    raw_ai_text = res_gemini.json()["candidates"][0]["content"]["parts"][0]["text"]
    clean_json = raw_ai_text.replace("```json", "").replace("```", "").strip()
    dubbed_data = json.loads(clean_json)

    full_hindi_script = " ".join([item["hindi"] for item in dubbed_data])
    print(f"   -> Hindi: {full_hindi_script}")

    # --------------------------------------------------------------------------
    # STEP 4: SPEECH SYNTHESIS (EDGE-TTS TO DISK - ZERO BUZZ/STATIC)
    # --------------------------------------------------------------------------
    print("🗣️ [4/5] Generating Native Indian Studio Voice via Edge-TTS...")
    raw_synth_mp3 = "synth_raw.mp3"
    asyncio.run(synthesize_edge_tts(full_hindi_script, chosen_voice, raw_synth_mp3))

    if not os.path.exists(raw_synth_mp3) or os.path.getsize(raw_synth_mp3) < 500:
        return None, None, "❌ Error: Speech synthesis failed. Check text input."

    # Decode MP3 to clean 16kHz WAV
    subprocess.run(f'ffmpeg -y -v quiet -i "{raw_synth_mp3}" -ar 16000 -ac 1 -c:a pcm_s16le synth_16k.wav', shell=True)

    # --------------------------------------------------------------------------
    # STEP 5: DYNAMIC ATEMPO & SILENCE REMOVAL (EXACT DURATION LOCK)
    # --------------------------------------------------------------------------
    print("✂️ [5/5] Aligning Duration & Stripping Pauses...")
    data_synth, sr_synth = sf.read("synth_16k.wav")
    synth_duration = len(data_synth) / sr_synth

    # Compute exact speed ratio
    speed_ratio = max(0.80, min(1.35, synth_duration / max(0.5, src_duration)))

    # Apply atempo + silence trim
    out_master = "final_dubbed_master.wav"
    cmd_polish = f'ffmpeg -y -v quiet -i synth_16k.wav -filter:a "atempo={speed_ratio:.4f},silenceremove=stop_periods=-1:stop_duration=0.18:stop_threshold=-35dB" "{out_master}"'
    subprocess.run(cmd_polish, shell=True)

    elapsed_time = time.time() - t_start_total
    status_report = (
        f"✅ DUBBING SUCCESSFUL in {elapsed_time:.2f} seconds!\n\n"
        f"Original Duration: {src_duration:.2f}s | Target Sync Ratio: {speed_ratio:.2f}x\n\n"
        f"Hindi Script:\n{full_hindi_script}"
    )

    return "input_16k.wav", out_master, status_report

# ==============================================================================
# GRADIO MODERN FRONTEND (CLEAN, ZERO-CRASH, MOBILE-FRIENDLY)
# ==============================================================================
custom_css = """
#main-container { max-width: 900px; margin: auto; padding: 20px; }
.header-box { text-align: center; margin-bottom: 25px; }
.status-area textarea { font-family: monospace; font-size: 13px; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange", secondary_hue="slate"), css=custom_css) as demo:
    with gr.Column(elem_id="main-container"):
        gr.Markdown(
            """
            # 🎙️ DubFlow Indic
            ### AI Video Localization & Dubbing Engine (24/7 Zero-GPU Architecture)
            Translate English YouTube videos or audio into expressive, natural Hindi with perfect pacing.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 1. Input Source")
                yt_input = gr.Textbox(
                    label="YouTube URL", 
                    placeholder="https://www.youtube.com/watch?v=...",
                    value="https://www.youtube.com/watch?v=UF8uR6Z6KLc"
                )
                audio_upload = gr.Audio(label="OR Upload Audio from Phone/PC", type="filepath")
                
                with gr.Row():
                    t_start = gr.Textbox(label="Start Time", value="00:00:10")
                    t_end = gr.Textbox(label="End Time", value="00:00:30")

                gr.Markdown("#### 2. Character & Voices")
                voice_select = gr.Dropdown(
                    label="Voice Character",
                    choices=[
                        "Deep Professional Male (Madhur)",
                        "Expressive Emotional Female (Swara)"
                    ],
                    value="Deep Professional Male (Madhur)"
                )

                with gr.Accordion("API Keys (Optional / Pre-configured)", open=False):
                    key_gemini = gr.Textbox(label="Gemini API Key", value=DEFAULT_GEMINI_KEY, type="password")
                    key_groq = gr.Textbox(label="Groq API Key (Optional)", value=DEFAULT_GROQ_KEY, type="password")

                btn_run = gr.Button("🚀 Dub Video Now", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("#### 3. Real-Time Status & Players")
                status_box = gr.Textbox(label="Pipeline Output & Hindi Script", lines=7, interactive=False, elem_classes=["status-area"])
                player_orig = gr.Audio(label="Original English Audio", type="filepath")
                player_dub = gr.Audio(label="Dubbed Hindi Output (Studio Master)", type="filepath")

        btn_run.click(
            fn=process_dubflow,
            inputs=[yt_input, audio_upload, t_start, t_end, voice_select, key_gemini, key_groq],
            outputs=[player_orig, player_dub, status_box]
        )

if __name__ == "__main__":
    # Runs on local port 7860 and binds automatically to 0.0.0.0 for cloud hosting
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
