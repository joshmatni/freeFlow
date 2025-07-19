# UI logic for freeFlow via streamlit
# this file will serve as streamlit UI
# it will grab a prompt/word, then display it to user
# then user will record themselves and respond
# practiticing their free association storytelling skills

"""
streamlit run webrtc_recorder.py
"""
import queue
import time
import wave
import threading
from collections import deque
from pathlib import Path
from typing import List

import av
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer


# ------------- TURN / STUN CONFIG ----------------------------------
def get_ice_servers():
    # Use a free public STUN server (replace with TURN creds in prod)
    return [{"urls": ["stun:stun.l.google.com:19302"]}]


# ------------- WAV HELPER ------------------------------------------
def save_wav(path: Path, chunks: List[np.ndarray], sample_rate: int):
    """Write a mono 16‑bit WAV from a list of int16 NumPy chunks."""
    audio = np.concatenate(chunks).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)        # mono
        wf.setsampwidth(2)        # 16‑bit = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())


# ------------- AUDIO‑ONLY RECORDING --------------------------------
def record_audio_only():
    ctx = webrtc_streamer(
        key="rec-audio",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        rtc_configuration={"iceServers": get_ice_servers()},
        media_stream_constraints={"video": False, "audio": True},
    )

    if not ctx.state.playing:
        st.info("Click ▶️ to start recording.")
        return

    st.success("Recording… click ⏹️ to stop.")
    chunks: List[np.ndarray] = []
    sample_rate = None

    while ctx.state.playing:
        try:
            for frame in ctx.audio_receiver.get_frames(timeout=0.5):
                if sample_rate is None:
                    sample_rate = frame.sample_rate
                # Flatten to mono & int16
                pcm = frame.to_ndarray().flatten().astype(np.int16)
                chunks.append(pcm)
        except queue.Empty:
            continue

    # ---- recording stopped ----
    if chunks and sample_rate:
        fname = Path(f"recording_{int(time.time())}.wav")
        save_wav(fname, chunks, sample_rate)
        st.success(f"Saved → {fname}")


# ------------- AUDIO + VIDEO RECORDING -----------------------------
def record_audio_video():
    frames_lock = deque_lock = None  # just to placate linters
    frames_lock = deque_lock = threading.Lock()
    audio_deque: deque = deque()

    async def queue_audio(frames: List[av.AudioFrame]) -> List[av.AudioFrame]:
        with frames_lock:
            audio_deque.extend(frames)

        # Return silent frames so nothing echoes back to the browser
        silent_frames = []
        for f in frames:
            zeros = np.zeros(f.to_ndarray().shape, dtype=f.to_ndarray().dtype)
            blank = av.AudioFrame.from_ndarray(zeros, layout=f.layout.name)
            blank.sample_rate = f.sample_rate
            silent_frames.append(blank)
        return silent_frames

    ctx = webrtc_streamer(
        key="rec-av",
        mode=WebRtcMode.SENDRECV,
        queued_audio_frames_callback=queue_audio,
        rtc_configuration={"iceServers": get_ice_servers()},
        media_stream_constraints={"video": True, "audio": True},
    )

    if not ctx.state.playing:
        st.info("Click ▶️ to start recording (audio + webcam).")
        return

    st.success("Recording… click ⏹️ to stop.")
    chunks: List[np.ndarray] = []
    sample_rate = None

    while ctx.state.playing:
        with frames_lock:
            while audio_deque:
                frame = audio_deque.popleft()
                if sample_rate is None:
                    sample_rate = frame.sample_rate
                pcm = frame.to_ndarray().flatten().astype(np.int16)
                chunks.append(pcm)
        time.sleep(0.05)

    # ---- recording stopped ----
    if chunks and sample_rate:
        fname = Path(f"recording_{int(time.time())}.wav")
        save_wav(fname, chunks, sample_rate)
        st.success(f"Saved → {fname}")


# ------------- MAIN UI ---------------------------------------------
def main():
    st.title("FreeFlow (save to WAV)")
    st.write("Practice your free association storytelling skills by recording audio or video responses to prompts.")
    mode = st.radio("Choose mode:", ("🎤 Audio only", "🎤📷 Audio + Video"))
    if mode.startswith("🎤 Audio only"):
        record_audio_only()
    else:
        record_audio_video()


if __name__ == "__main__":
    main()
