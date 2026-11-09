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

DEFAULT_GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "AQ.Ab8RN6IgXYVXIR8wu3GOlWMzal7909-9Llg9dfeQjpwtThSVRQ")
VOICE_MAP = {
    "Deep Professional Male (Madhur)": "hi-IN-MadhurNeural",
    "Expressive Emotional Female (Swara)": "hi-IN-SwaraNeural"
}

async def synthesize_edge_tts(text: str, voice_name: str, target_file: str):
    communicate = edge_tts.Communicate(text, voice_name)
    await communicate.save(target_file)

def process_dubflow(youtube_url, upload_audio, start_time, end_time, voice_choice, gemini_key):
    t_start = time.time()
    
    # Clean previous temp files
    for tf in ["input_clean.wav", "synth_raw.mp3", "synth_16k.wav", "final_dubbed_master.wav"]:
        if os.path.exists(tf):
            try: os.remove(tf)
            except: pass

    active_gemini_key = gemini_key.strip() if gemini_key else DEFAULT_GEMINI_KEY
    if not active_gemini_key:
        return None, None, "❌ Error: Gemini API Key is required!"

    chosen_voice = VOICE_MAP.get(voice_choice, "hi-IN-MadhurNeural")

    # Priority 1: Direct Phone Audio Upload (100% Reliable, Zero YouTube block)
    if upload_audio:
        print("📁 Using Direct Audio Upload from Phone...")
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

    # Measure duration
    data_in, sr_in = sf.read("input_clean.wav")
    src_duration = len(data_in) / sr_in

    # Transcribe via Whisper
    from faster_whisper import WhisperModel
    whisper = WhisperModel("base", device="cpu", compute_type="int8")
    segs, info = whisper.transcribe(data_in, beam_size=5)
    
    transcript = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segs]
    if not transcript:
        return None, None, "❌ No speech detected in this audio."

    # Gemini Flash Translation
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={active_gemini_key}"
    prompt = f"""
    Translate into natural, conversational, inspirational Hindi (Devanagari).
    Keep length budgeted to match speaking duration ({src_duration:.1f}s).
    Return STRICTLY JSON: [{{"start": float, "end": float, "english": "...", "hindi": "...", "tone": "..."}}]
    Transcript: {json.dumps(transcript)}
    """
    res = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2, "maxOutputTokens": 450}})
    clean_json = res.json()["candidates"][0]["content"]["parts"][0]["text"].replace("```json", "").replace("```", "").strip()
    dubbed_data = json.loads(clean_json)
    full_hindi = " ".join([d["hindi"] for d in dubbed_data])

    # Synthesize Native Hindi
    asyncio.run(synthesize_edge_tts(full_hindi, chosen_voice, "synth_raw.mp3"))
    subprocess.run('ffmpeg -y -i synth_raw.mp3 -ar 16000 -ac 1 -c:a pcm_s16le synth_16k.wav', shell=True)

    # Time-stretch and silence strip
    data_synth, sr_synth = sf.read("synth_16k.wav")
    synth_duration = len(data_synth) / sr_synth
    speed_ratio = max(0.80, min(1.35, synth_duration / max(0.5, src_duration)))

    out_master = "final_dubbed_master.wav"
    subprocess.run(f'ffmpeg -y -i synth_16k.wav -filter:a "atempo={speed_ratio:.4f},silenceremove=stop_periods=-1:stop_duration=0.18:stop_threshold=-35dB" "{out_master}"', shell=True)

    status_msg = f"✅ DUBBING SUCCESSFUL in {time.time()-t_start:.1f}s!\n\nHindi Script:\n{full_hindi}"
    return "input_clean.wav", out_master, status_msg

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange", secondary_hue="slate")) as demo:
    gr.Markdown("# 🎙️ DubFlow Indic\n### AI Video Localization & Dubbing Engine (24/7 Live)")
    with gr.Row():
        with gr.Column():
            audio_upload = gr.Audio(label="Upload Audio Directly from Phone (Recommended)", type="filepath")
            yt_input = gr.Textbox(label="OR YouTube URL (Optional)", value="")
            with gr.Row():
                t_start = gr.Textbox(label="Start Time", value="00:00:00")
                t_end = gr.Textbox(label="End Time", value="00:00:30")
            voice_select = gr.Dropdown(label="Voice Character", choices=["Deep Professional Male (Madhur)", "Expressive Emotional Female (Swara)"], value="Deep Professional Male (Madhur)")
            key_gemini = gr.Textbox(label="Gemini API Key", value=DEFAULT_GEMINI_KEY, type="password")
            btn_run = gr.Button("🚀 Dub Video Now", variant="primary", size="lg")
        with gr.Column():
            status_box = gr.Textbox(label="Status & Hindi Translation", lines=6, interactive=False)
            player_orig = gr.Audio(label="Original Audio", type="filepath")
            player_dub = gr.Audio(label="Dubbed Hindi Output (Studio Master)", type="filepath")

    btn_run.click(fn=process_dubflow, inputs=[yt_input, audio_upload, t_start, t_end, voice_select, key_gemini], outputs=[player_orig, player_dub, status_box])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
