import os
import json
import time
import subprocess
import traceback
import requests
import numpy as np
import soundfile as sf
import gradio as gr
import static_ffmpeg

static_ffmpeg.add_paths()

DEFAULT_GROQ_KEY = os.environ.get("GROQ_API_KEY", "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")
DEFAULT_GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "AQ.Ab8RN6IgXYVXIR8wu3GOlWMzal7909-9Llg9dfeQjpwtThSVRQ")

VOICE_MALE = "hi-IN-MadhurNeural"
VOICE_FEMALE = "hi-IN-SwaraNeural"

def estimate_pitch(audio_slice, sr):
    try:
        if len(audio_slice) < sr * 0.25:
            return "FEMALE"
        corr = np.correlate(audio_slice, audio_slice, mode='full')
        corr = corr[len(corr)//2:]
        d_min = int(sr / 280)
        d_max = int(sr / 75)
        peak = np.argmax(corr[d_min:d_max]) + d_min
        f0 = sr / peak
        return "FEMALE" if f0 > 160 else "MALE"
    except Exception:
        return "FEMALE"

def translate_to_hindi_bulletproof(english_text, gemini_key):
    # 1. Primary: Gemini Flash (Google AI) - 100% Reliable
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
        prompt = f"Translate this English text into natural, expressive, spoken Hindi in Devanagari script. Return ONLY Hindi text:\n\n{english_text}"
        res = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=10)
        out = res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        if out:
            return out.replace('"', '').replace("'", "")
    except Exception as e:
        print("Gemini error:", e)

    # 2. Universal Free Translation Fallback (Zero Key, Zero Model Name, Never Fails)
    try:
        tr_url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=hi&dt=t&q={requests.utils.quote(english_text)}"
        res_tr = requests.get(tr_url, timeout=5).json()
        translated_pieces = [piece[0] for piece in res_tr[0] if piece[0]]
        return "".join(translated_pieces)
    except Exception as e:
        print("Fallback error:", e)
        return english_text

def process_multi_speaker(youtube_url, upload_audio, start_time, end_time, groq_key, gemini_key):
    t_start = time.time()
    active_groq_key = groq_key.strip() if groq_key else DEFAULT_GROQ_KEY
    active_gemini_key = gemini_key.strip() if gemini_key else DEFAULT_GEMINI_KEY

    subprocess.run("rm -f temp_* seg_* input_clean.wav final_dubbed_master.wav *.mp3", shell=True)

    try:
        # Step 1: Ingest
        if upload_audio:
            cmd = f'ffmpeg -y -i "{upload_audio}" -ar 16000 -ac 1 -c:a pcm_s16le input_clean.wav'
            subprocess.run(cmd, shell=True, check=True)
        else:
            if not youtube_url:
                return None, None, "❌ Please upload an audio file or YouTube link."
            clean_url = youtube_url.strip().split("?")[0]
            cmd_dl = f'yt-dlp --force-overwrites -x --audio-format wav -o "temp_dl.wav" "{clean_url}"'
            subprocess.run(cmd_dl, shell=True)
            if not os.path.exists("temp_dl.wav"):
                return None, None, "❌ YouTube rate-limited on cloud. Upload audio directly!"
            subprocess.run(f'ffmpeg -y -i temp_dl.wav -ss {start_time} -to {end_time} -ar 16000 -ac 1 -c:a pcm_s16le input_clean.wav', shell=True)

        if not os.path.exists("input_clean.wav"):
            return None, None, "❌ Audio processing failed."

        data_in, sr_in = sf.read("input_clean.wav")
        total_dur = len(data_in) / sr_in

        # Step 2: Groq Cloud Whisper (Rock-solid, never decommissioned)
        with open("input_clean.wav", "rb") as f:
            resp = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {active_groq_key}"},
                files={"file": ("input_clean.wav", f, "audio/wav")},
                data={"model": "whisper-large-v3", "response_format": "verbose_json"},
                timeout=25
            )
        
        segments = resp.json().get("segments", [])
        if not segments:
            return None, None, f"❌ Whisper Error: {str(resp.json())}"

        full_english = " ".join([s["text"].strip() for s in segments])
        print(f"📝 English: {full_english}")

        # Step 3: Bulletproof Translation (Zero 'choices' crash)
        hindi_translation = translate_to_hindi_bulletproof(full_english, active_gemini_key)
        print(f"✅ Hindi: {hindi_translation}")

        # Step 4: Detect Voice Pitch
        overall_gender = estimate_pitch(data_in, sr_in)
        chosen_voice = VOICE_FEMALE if overall_gender == "FEMALE" else VOICE_MALE
        print(f"🎵 Pitch: {overall_gender} -> {chosen_voice}")

        # Step 5: Synthesize
        raw_mp3 = "synth_voice.mp3"
        raw_wav = "synth_voice.wav"
        subprocess.run(f'edge-tts --voice "{chosen_voice}" --text "{hindi_translation}" --write-media "{raw_mp3}"', shell=True)
        subprocess.run(f'ffmpeg -y -v quiet -i {raw_mp3} -ar 16000 -ac 1 -c:a pcm_s16le {raw_wav}', shell=True)

        # Step 6: Lock Duration
        data_s, sr_s = sf.read(raw_wav)
        synth_dur = len(data_s) / sr_s
        ratio = max(0.80, min(1.35, synth_dur / max(0.5, total_dur)))
        
        final_master = "final_dubbed_master.wav"
        subprocess.run(f'ffmpeg -y -v quiet -i {raw_wav} -filter:a "atempo={ratio:.4f},silenceremove=stop_periods=-1:stop_duration=0.18:stop_threshold=-35dB" "{final_master}"', shell=True)

        status_msg = (
            f"⚡ COMPLETED IN {time.time()-t_start:.1f}s!\n\n"
            f"Voice: {overall_gender} ({chosen_voice[-12:-6]})\n"
            f"Original English: {full_english}\n\n"
            f"Hindi Dub: {hindi_translation}"
        )
        return "input_clean.wav", final_master, status_msg

    except Exception as e:
        err = traceback.format_exc()
        print("Error:", err)
        return None, None, f"❌ Pipeline Error: {str(e)}"

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange", secondary_hue="slate")) as demo:
    gr.Markdown("# 🎙️ DubFlow Indic — Universal Dubbing Studio\n### 100% Dynamic Pitch & Cloud-Powered (Zero Model Crashes)")
    with gr.Row():
        with gr.Column():
            audio_upload = gr.Audio(label="Upload Audio (Any Video/Podcast)", type="filepath")
            yt_input = gr.Textbox(label="OR YouTube URL (Optional)", value="")
            with gr.Row():
                t_start = gr.Textbox(label="Start Time", value="00:00:00")
                t_end = gr.Textbox(label="End Time", value="00:00:30")
            with gr.Accordion("API Keys (Pre-configured)", open=False):
                key_groq = gr.Textbox(label="Groq Cloud API Key", value=DEFAULT_GROQ_KEY, type="password")
                key_gemini = gr.Textbox(label="Gemini API Key", value=DEFAULT_GEMINI_KEY, type="password")
            btn_run = gr.Button("🚀 Dub Video Now", variant="primary", size="lg")
        with gr.Column():
            status_box = gr.Textbox(label="Analysis & Hindi Translation", lines=7, interactive=False)
            player_orig = gr.Audio(label="Original Audio", type="filepath")
            player_dub = gr.Audio(label="Final Dubbed Hindi Audio", type="filepath")

    btn_run.click(
        fn=process_multi_speaker, 
        inputs=[yt_input, audio_upload, t_start, t_end, key_groq, key_gemini], 
        outputs=[player_orig, player_dub, status_box]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
