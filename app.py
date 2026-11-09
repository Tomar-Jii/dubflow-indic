import os
import json
import time
import asyncio
import subprocess
import traceback
import requests
import soundfile as sf
import edge_tts
import gradio as gr
import static_ffmpeg

static_ffmpeg.add_paths()

DEFAULT_GROQ_KEY = os.environ.get("GROQ_API_KEY", "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")

# Character Mapping
VOICE_MALE = "hi-IN-MadhurNeural"
VOICE_FEMALE = "hi-IN-SwaraNeural"

async def synth_voice(text, voice_name, out_path):
    comm = edge_tts.Communicate(text, voice_name)
    await comm.save(out_path)

def process_multi_speaker(youtube_url, upload_audio, start_time, end_time, groq_key):
    t_start = time.time()
    active_groq_key = groq_key.strip() if groq_key else DEFAULT_GROQ_KEY

    # Clean previous temp files
    subprocess.run("rm -f temp_* seg_* input_clean.wav final_dubbed_master.wav", shell=True)

    try:
        # Step 1: Ingest Audio
        if upload_audio:
            cmd_ingest = f'ffmpeg -y -i "{upload_audio}" -ar 16000 -ac 1 -c:a pcm_s16le input_clean.wav'
            subprocess.run(cmd_ingest, shell=True, check=True)
        else:
            if not youtube_url:
                return None, None, "❌ Please upload an audio file or YouTube link."
            clean_url = youtube_url.strip().split("?")[0]
            cmd_dl = f'yt-dlp --force-overwrites -x --audio-format wav -o "temp_dl.wav" "{clean_url}"'
            subprocess.run(cmd_dl, shell=True)
            if not os.path.exists("temp_dl.wav"):
                return None, None, "❌ YouTube rate-limited. Upload audio directly!"
            subprocess.run(f'ffmpeg -y -i temp_dl.wav -ss {start_time} -to {end_time} -ar 16000 -ac 1 -c:a pcm_s16le input_clean.wav', shell=True)

        data_in, sr_in = sf.read("input_clean.wav")
        total_duration = len(data_in) / sr_in

        # Step 2: Groq Cloud Whisper with Segment Timestamps
        groq_url = "https://api.groq.com/openai/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {active_groq_key}"}
        with open("input_clean.wav", "rb") as f:
            resp = requests.post(
                groq_url,
                headers=headers,
                files={"file": ("input_clean.wav", f, "audio/wav")},
                data={"model": "whisper-large-v3", "response_format": "verbose_json"},
                timeout=25
            )
        
        segments = resp.json().get("segments", [])
        if not segments:
            return None, None, "❌ No speech segments detected."

        raw_transcript = [{"start": s["start"], "end": s["end"], "text": s["text"].strip()} for s in segments]

        # Step 3: LLM Multi-Speaker Diarization & Translation (Groq Llama-3)
        prompt = f"""
        Analyze this transcript with timestamps. 
        Identify speaker turns and gender (MALE or FEMALE) based on context and dialogue flow.
        Translate each turn into natural, expressive Hindi in Devanagari script.
        Keep sentence lengths budgeted to match their original durations.

        Return STRICTLY a JSON list with this exact structure:
        [
          {{
            "speaker_gender": "MALE" or "FEMALE",
            "start": float,
            "end": float,
            "original_english": "...",
            "hindi_dub": "..."
          }}
        ]

        Transcript:
        {json.dumps(raw_transcript)}
        """

        llm_payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": "You are a professional dubbing director. Output valid JSON only, no markdown, no explanation."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"} if False else None
        }
        res_llm = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=llm_payload, timeout=15)
        clean_json_str = res_llm.json()["choices"][0]["message"]["content"].replace("```json", "").replace("```", "").strip()
        parsed_script = json.loads(clean_json_str)
        if isinstance(parsed_script, dict) and "segments" in parsed_script:
            parsed_script = parsed_script["segments"]

        # Step 4: Multi-Speaker Generation & Timestamp Stitching
        segment_files = []
        status_log = "🎙️ MULTI-SPEAKER TIMELINE:\n"

        for idx, seg in enumerate(parsed_script):
            gender = seg.get("speaker_gender", "MALE").upper()
            hindi_text = seg.get("hindi_dub", "").strip()
            target_dur = max(0.5, seg.get("end", 0) - seg.get("start", 0))
            
            chosen_voice = VOICE_MALE if gender == "MALE" else VOICE_FEMALE
            raw_seg_mp3 = f"seg_raw_{idx}.mp3"
            raw_seg_wav = f"seg_raw_{idx}.wav"
            stretched_seg_wav = f"seg_stretched_{idx}.wav"

            # Synthesize segment
            asyncio.run(synth_voice(hindi_text, chosen_voice, raw_seg_mp3))
            subprocess.run(f'ffmpeg -y -v quiet -i {raw_seg_mp3} -ar 16000 -ac 1 -c:a pcm_s16le {raw_seg_wav}', shell=True)

            # Measure & Time-stretch to exact segment duration
            data_s, sr_s = sf.read(raw_seg_wav)
            cur_dur = len(data_s) / sr_s
            speed_ratio = max(0.75, min(1.40, cur_dur / target_dur))
            subprocess.run(f'ffmpeg -y -v quiet -i {raw_seg_wav} -filter:a "atempo={speed_ratio:.4f}" {stretched_seg_wav}', shell=True)

            segment_files.append(stretched_seg_wav)
            status_log += f"[{seg.get('start',0):.1f}s - {seg.get('end',0):.1f}s] {gender} Voice -> {hindi_text}\n"

        # Step 5: Concat all aligned segments into master timeline
        concat_list = "concat_list.txt"
        with open(concat_list, "w") as f:
            for sf_path in segment_files:
                f.write(f"file '{sf_path}'\n")

        master_out = "final_dubbed_master.wav"
        subprocess.run(f'ffmpeg -y -v quiet -f concat -safe 0 -i {concat_list} -c:a pcm_s16le {master_out}', shell=True)

        total_time = time.time() - t_start
        status_summary = f"⚡ COMPLETED IN {total_time:.1f}s!\n\n{status_log}"
        return "input_clean.wav", master_out, status_summary

    except Exception as e:
        err = traceback.format_exc()
        print("Error:", err)
        return None, None, f"❌ Pipeline Error: {str(e)}"

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange", secondary_hue="slate")) as demo:
    gr.Markdown("# 🎙️ DubFlow Indic — Multi-Speaker Studio\n### Auto-Detects Male & Female Voices + Syncs Timestamps!")
    with gr.Row():
        with gr.Column():
            audio_upload = gr.Audio(label="Upload Audio Directly from Phone (Alia Bhatt & Jay Shetty)", type="filepath")
            yt_input = gr.Textbox(label="OR YouTube URL (Optional)", value="")
            with gr.Row():
                t_start = gr.Textbox(label="Start Time", value="00:00:00")
                t_end = gr.Textbox(label="End Time", value="00:00:30")
            key_groq = gr.Textbox(label="Groq Cloud API Key", value=DEFAULT_GROQ_KEY, type="password")
            btn_run = gr.Button("🚀 Dub Multi-Speaker Video Now", variant="primary", size="lg")
        with gr.Column():
            status_box = gr.Textbox(label="Multi-Speaker Timeline & Hindi Dialogue", lines=8, interactive=False)
            player_orig = gr.Audio(label="Original Audio", type="filepath")
            player_dub = gr.Audio(label="Dubbed Multi-Speaker Hindi Output", type="filepath")

    btn_run.click(
        fn=process_multi_speaker, 
        inputs=[yt_input, audio_upload, t_start, t_end, key_groq], 
        outputs=[player_orig, player_dub, status_box]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
