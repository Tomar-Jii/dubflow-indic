import os
import json
import time
import asyncio
import subprocess
import requests
import soundfile as sf
import edge_tts
import gradio as gr
import static_ffmpeg

# Initialize portable static ffmpeg (Zero apt required)
static_ffmpeg.add_paths()

# API Keys
DEFAULT_GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "AQ.Ab8RN6IgXYVXIR8wu3GOlWMzal7909-9Llg9dfeQjpwtThSVRQ")
DEFAULT_GROQ_KEY = os.environ.get("GROQ_API_KEY", "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")

VOICE_MAP = {
    "Deep Professional Male (Madhur)": "hi-IN-MadhurNeural",
    "Expressive Emotional Female (Swara)": "hi-IN-SwaraNeural"
}

async def synthesize_edge_tts(text: str, voice_name: str, target_file: str):
    communicate = edge_tts.Communicate(text, voice_name)
    await communicate.save(target_file)

def process_dubflow(youtube_url, upload_audio, start_time, end_time, voice_choice, gemini_key, groq_key):
    t_start = time.time()
    
    # Clean previous temp files
    for tf in ["input_clean.wav", "synth_raw.mp3", "synth_16k.wav", "final_dubbed_master.wav"]:
        if os.path.exists(tf):
            try: os.remove(tf)
            except: pass

    active_gemini_key = gemini_key.strip() if gemini_key else DEFAULT_GEMINI_KEY
    active_groq_key = groq_key.strip() if groq_key else DEFAULT_GROQ_KEY

    if not active_gemini_key:
        return None, None, "❌ Error: Gemini API Key is required!"

    chosen_voice = VOICE_MAP.get(voice_choice, "hi-IN-MadhurNeural")

    # Step 1: Audio Ingestion (Direct File vs YouTube)
    if upload_audio:
        print("📁 Ingesting Direct Audio Upload...")
        cmd_ingest = f'ffmpeg -y -i "{upload_audio}" -ar 16000 -ac 1 -c:a pcm_s16le input_clean.wav'
        subprocess.run(cmd_ingest, shell=True)
    else:
        if not youtube_url:
            return None, None, "❌ Error: Please provide an audio upload or YouTube link."
        clean_url = youtube_url.strip().split("?")[0]
        cmd_dl = f'yt-dlp --force-overwrites -x --audio-format wav -o "temp_dl.wav" "{clean_url}"'
        subprocess.run(cmd_dl, shell=True)
        if not os.path.exists("temp_dl.wav"):
            return None, None, "❌ YouTube Error: IP Rate-limited. Please upload audio directly from phone!"
        subprocess.run(f'ffmpeg -y -i temp_dl.wav -ss {start_time} -to {end_time} -ar 16000 -ac 1 -c:a pcm_s16le input_clean.wav', shell=True)

    if not os.path.exists("input_clean.wav"):
        return None, None, "❌ Audio processing error."

    # Measure input duration
    data_in, sr_in = sf.read("input_clean.wav")
    src_duration = len(data_in) / sr_in

    # Step 2: Groq Cloud Whisper API (Sub-Second Latency!)
    print("⚡ [Groq Cloud] Transcribing via Whisper Large v3...")
    t_whisper = time.time()
    groq_url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {active_groq_key}"}
    
    with open("input_clean.wav", "rb") as f:
        resp = requests.post(
            groq_url,
            headers=headers,
            files={"file": ("input_clean.wav", f, "audio/wav")},
            data={"model": "whisper-large-v3", "response_format": "verbose_json"},
            timeout=20
        )
    
    groq_data = resp.json()
    if "segments" not in groq_data:
        return None, None, f"❌ Groq ASR Error: {str(groq_data)}"

    transcript = [{"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()} for s in groq_data["segments"]]
    print(f"✅ Transcribed in {time.time()-t_whisper:.2f}s: {len(transcript)} line(s)")

    # Step 3: Gemini Flash Translation (0.8s)
    print("🤖 [Gemini Flash] Translating into Conversational Hindi...")
    t_gemini = time.time()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={active_gemini_key}"
    prompt = f"""
    Translate into natural, conversational, inspirational Hindi (Devanagari).
    Keep length budgeted to match speaking duration ({src_duration:.1f}s).
    Return STRICTLY JSON: [{{"start": float, "end": float, "english": "...", "hindi": "...", "tone": "..."}}]
    Transcript: {json.dumps(transcript)}
    """
    res = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2, "maxOutputTokens": 450}}, timeout=15)
    clean_json = res.json()["candidates"][0]["content"]["parts"][0]["text"].replace("```json", "").replace("```", "").strip()
    dubbed_data = json.loads(clean_json)
    full_hindi = " ".join([d["hindi"] for d in dubbed_data])
    print(f"✅ Translated in {time.time()-t_gemini:.2f}s: {full_hindi}")

    # Step 4: Edge-TTS Native Hindi Speech
    print("🗣️ [Edge-TTS] Synthesizing Studio Voice...")
    asyncio.run(synthesize_edge_tts(full_hindi, chosen_voice, "synth_raw.mp3"))
    subprocess.run('ffmpeg -y -i synth_raw.mp3 -ar 16000 -ac 1 -c:a pcm_s16le synth_16k.wav', shell=True)

    # Step 5: Dynamic atempo & Silence Stripper
    data_synth, sr_synth = sf.read("synth_16k.wav")
    synth_duration = len(data_synth) / sr_synth
    speed_ratio = max(0.80, min(1.35, synth_duration / max(0.5, src_duration)))

    out_master = "final_dubbed_master.wav"
    subprocess.run(f'ffmpeg -y -i synth_16k.wav -filter:a "atempo={speed_ratio:.4f},silenceremove=stop_periods=-1:stop_duration=0.18:stop_threshold=-35dB" "{out_master}"', shell=True)

    total_time = time.time() - t_start
    status_msg = (
        f"⚡ COMPLETED IN {total_time:.2f} SECONDS!\n\n"
        f"Groq Whisper ASR: {time.time()-t_whisper:.2f}s\n"
        f"Gemini Translation: {time.time()-t_gemini:.2f}s\n\n"
        f"Hindi Script:\n{full_hindi}"
    )
    return "input_clean.wav", out_master, status_msg

# Gradio Mobile-Optimized Interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange", secondary_hue="slate")) as demo:
    gr.Markdown("# 🎙️ DubFlow Indic\n### Ultra-Fast AI Video Dubbing Engine (Cloud-Powered 24/7)")
    with gr.Row():
        with gr.Column():
            audio_upload = gr.Audio(label="Upload Audio Directly from Phone (Instant)", type="filepath")
            yt_input = gr.Textbox(label="OR YouTube URL (Optional)", value="")
            with gr.Row():
                t_start = gr.Textbox(label="Start Time", value="00:00:00")
                t_end = gr.Textbox(label="End Time", value="00:00:30")
            voice_select = gr.Dropdown(label="Voice Character", choices=["Deep Professional Male (Madhur)", "Expressive Emotional Female (Swara)"], value="Deep Professional Male (Madhur)")
            with gr.Accordion("API Keys (Configured)", open=False):
                key_gemini = gr.Textbox(label="Gemini API Key", value=DEFAULT_GEMINI_KEY, type="password")
                key_groq = gr.Textbox(label="Groq API Key", value=DEFAULT_GROQ_KEY, type="password")
            btn_run = gr.Button("🚀 Dub Video Now", variant="primary", size="lg")
        with gr.Column():
            status_box = gr.Textbox(label="Status & Hindi Translation", lines=6, interactive=False)
            player_orig = gr.Audio(label="Original Audio", type="filepath")
            player_dub = gr.Audio(label="Dubbed Hindi Output (Studio Master)", type="filepath")

    btn_run.click(
        fn=process_dubflow, 
        inputs=[yt_input, audio_upload, t_start, t_end, voice_select, key_gemini, key_groq], 
        outputs=[player_orig, player_dub, status_box]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
