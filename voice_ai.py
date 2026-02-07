import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import tempfile
import requests
import sounddevice as sd
import soundfile as sf
import numpy as np

from faster_whisper import WhisperModel
from indextts.infer_v2 import IndexTTS2
from kokoro import KPipeline

from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader
from docx import Document
import markdown
from bs4 import BeautifulSoup


# =========================
# CONFIGURATION
# =========================

# ---- Audio recording config ----
SAMPLE_RATE = 16000
CHANNELS = 1
# RECORD_SECONDS = 5  # you can increase this if you want longer utterances

# Streaming STT / VAD settings
LISTEN_TIMEOUT = 15.0       # max seconds to wait in a single turn
PHRASE_TIMEOUT = 1.2        # how long of silence (sec) before we stop listening
CHUNK_DURATION = 0.2        # seconds per audio chunk
SILENCE_THRESHOLD = 0.03    # volume threshold — you can tweak if too sensitive / insensitive

# ---- Whisper (STT) config ----
WHISPER_MODEL_SIZE = "distil-large-v3"  # "tiny", "base", "small", "medium", "large-v3", etc.
WHISPER_DEVICE = "cuda"        # change to "cuda" if you have a compatible GPU
WHISPER_COMPUTE_TYPE = "int8_float16"  # e.g. "int8_float16" for GPU, "float32" for CPU

# ---- LM Studio (LLM) config ----
LMSTUDIO_BASE_URL = "http://localhost:1234/v1/chat/completions"
LMSTUDIO_MODEL_NAME = ""  # e.g. "Meta-Llama-3-8B-Instruct-GGUF"

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the provided context if needed."
)

# ---- IndexTTS2 (TTS) config ----
INDEXTTS_CFG_PATH = "checkpoints/config.yaml"
INDEXTTS_MODEL_DIR = "checkpoints"

SPEAKER_AUDIO_PROMPT = "examples/voice_03.wav"   # your reference voice
EMO_AUDIO_PROMPT = "examples/emo_sad.wav"        # or another emotion reference
USE_FP16 = False
USE_CUDA_KERNEL = False
USE_DEEPSPEED = False

# ---- RAG config ----
KNOWLEDGE_DIR = "knowledge"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3

# ---- Colour config ----
BLUE  = "\033[94m"
GREEN = "\033[92m"
PINK  = "\033[95m"
RESET = "\033[0m"

# =========================
# INITIALIZATION
# =========================

print("[*] Loading Whisper model...")
whisper_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE,
)

# IndexTTS2 takes too long
# print("[*] Initializing IndexTTS2...")
# tts = IndexTTS2(
#     cfg_path=INDEXTTS_CFG_PATH,
#     model_dir=INDEXTTS_MODEL_DIR,
#     use_fp16=USE_FP16,
#     use_cuda_kernel=USE_CUDA_KERNEL,
#     use_deepspeed=USE_DEEPSPEED,
# )

print("[*] Initializing Kokoro...")
pipeline = KPipeline(lang_code='a')

print("[*] Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL_NAME)

print("[*] Building vector index...")
documents = None
doc_embeddings = None
index = None

conversation = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

# =========================
# HELPER FUNCTIONS
# =========================

def load_pdf(path):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append((text, {"source": path, "page": i + 1}))
    return pages

def load_docx(path):
    doc = Document(path)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [(text, {"source": path})]

def load_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return [(f.read(), {"source": path})]

def load_md(path):
    with open(path, "r", encoding="utf-8") as f:
        html = markdown.markdown(f.read())
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text()
        return [(text, {"source": path})]

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

def load_documents(folder):
    texts = []
    metadatas = []

    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if not os.path.isfile(path):
            continue

        ext = fname.lower().split(".")[-1]

        if ext == "pdf":
            raw_docs = load_pdf(path)
        elif ext == "docx":
            raw_docs = load_docx(path)
        elif ext == "txt":
            raw_docs = load_txt(path)
        elif ext == "md":
            raw_docs = load_md(path)
        else:
            continue

        for text, meta in raw_docs:
            for chunk in chunk_text(text):
                texts.append(chunk)
                metadatas.append(meta)

    return texts, metadatas

def listen_until_silence(timeout: float = LISTEN_TIMEOUT, phrase_timeout: float = PHRASE_TIMEOUT,):
    """
    Stream microphone audio in small chunks until:
      - user starts talking, then
      - we detect 'phrase_timeout' seconds of silence.
    Returns path to a temporary WAV file, or None if no speech.
    """
    print("\n[REC] Listening... start speaking (Ctrl+C to abort).")

    chunk_samples = int(CHUNK_DURATION * SAMPLE_RATE)
    max_chunks = int(timeout / CHUNK_DURATION)
    max_silence_chunks = int(phrase_timeout / CHUNK_DURATION)

    audio_chunks = []
    num_silent = 0
    started_talking = False

    with sd.InputStream(channels=CHANNELS, samplerate=SAMPLE_RATE, dtype="float32") as stream:
        for _ in range(max_chunks):
            chunk, _ = stream.read(chunk_samples)
            audio_chunks.append(chunk.copy())

            # Simple volume-based VAD
            volume = np.sqrt(np.mean(chunk**2))

            if volume > SILENCE_THRESHOLD:
                # speech
                num_silent = 0
                if not started_talking:
                    started_talking = True
                    print("[REC] Detected speech...")
            else:
                # silence
                if started_talking:
                    num_silent += 1
                    if num_silent >= max_silence_chunks:
                        print("[REC] Silence detected, stopping.")
                        break
        else:
            print("[REC] Timeout reached, stopping.")

    if not audio_chunks:
        print("[REC] No audio captured.")
        return None

    # Concatenate all chunks into a single waveform
    audio = np.concatenate(audio_chunks, axis=0)

    # If user never spoke above threshold, treat as no speech
    if not started_talking:
        print("[REC] No speech detected above threshold.")
        return None

    # Save to temp WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, SAMPLE_RATE)
        print(f"[REC] Saved audio to {tmp.name}")
        return tmp.name

def transcribe_audio(filename: str) -> str:
    """Transcribe audio using faster-whisper."""
    print("[STT] Transcribing audio...")
    segments, _ = whisper_model.transcribe(filename, beam_size=5)

    text_chunks = []
    for seg in segments:
        text_chunks.append(seg.text)

    text = " ".join(text_chunks).strip()
    print(f"{BLUE}[STT/User] Transcription: {text!r}{RESET}")
    return text

def retrieve_context(query):
    q_emb = embedder.encode([query])
    D, I = index.search(q_emb, TOP_K)

    results = []
    for idx in I[0]:
        meta = metadatas[idx]
        prefix = f"[Source: {os.path.basename(meta['source'])}"
        if "page" in meta:
            prefix += f", Page {meta['page']}"
        prefix += "]"

        results.append(prefix + "\n" + texts[idx])

    return "\n\n".join(results)

def call_lmstudio(user_text):
    """Call LM Studio local server using OpenAI-compatible /chat/completions."""
    context = retrieve_context(user_text)

    conversation.extend([
        {"role": "system", "content": f"context:\n{context}"},
        {"role": "user", "content": user_text}
    ])

    payload = {
        "model": LMSTUDIO_MODEL_NAME,
        "messages": conversation,
        #"temperature": 0.7,
        #"max_tokens": 256,
    }

    try:
        resp = requests.post(LMSTUDIO_BASE_URL, json=payload, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as e:
        print("[LLM] Error calling LM Studio:", e)
        return None

    data = resp.json()
    # LM Studio is OpenAI-compatible:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        print("[LLM] Unexpected response structure:", data)
        return None
    
    return content

# IndexTTS2 takes too long
# def tts_generate_and_play_indexTTS2(text: str):
#     """Generate speech with IndexTTS2 and play it."""
#     print("[TTS-IndexTTS2] Generating audio...")
#     with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
#         out_path = tmp.name

#     tts.infer(
#         spk_audio_prompt=SPEAKER_AUDIO_PROMPT,
#         text=text,
#         output_path=out_path,
#         # emo_audio_prompt=EMO_AUDIO_PROMPT,
#         verbose=False,
#     )

#     print("[TTS-IndexTTS2] Playing audio...")
#     audio, sr = sf.read(out_path)

#     # Add silence at the end to prevent audio cutoff
#     silence_duration = 0.5  # seconds
#     silence_samples = int(silence_duration * SAMPLE_RATE)
#     if len(audio.shape) == 2: # If stereo
#         silence = np.zeros((silence_samples, audio.shape[1]), dtype=audio.dtype)
#     else: # else mono
#         silence = np.zeros(silence_samples, dtype=audio.dtype)
#     audio = np.concatenate([audio, silence])

#     sd.play(audio, sr)
#     sd.wait()
#     os.remove(out_path)

def tts_generate_and_play_kokoro(text: str):
    """Generate speech with kokoro and play it."""
    print("[TTS-Kokoro] Generating audio...")

    generator = pipeline(
        text, voice='af_heart', # <= change voice here, af_heart/af_bella
        speed=1#, split_pattern=r'\n+'
    )

    print("[TTS-Kokoro] Playing audio...")
    audio = []
    for result in generator:
        audio.append(result.audio.cpu().numpy())
    audio = np.concatenate(audio)

    # Add silence at the end to prevent audio cutoff
    silence_duration = 0.5  # seconds
    silence_samples = int(silence_duration * SAMPLE_RATE)
    if len(audio.shape) == 2: # If stereo
        silence = np.zeros((silence_samples, audio.shape[1]), dtype=audio.dtype)
    else: # else mono
        silence = np.zeros(silence_samples, dtype=audio.dtype)
    audio = np.concatenate([audio, silence])

    sd.play(audio, samplerate=24000)
    sd.wait()

# =========================
# MAIN LOOP
# =========================

texts, metadatas = load_documents(KNOWLEDGE_DIR)
print(f"[*] Total chunks: {len(texts)}")
embeddings = embedder.encode(texts)
if embeddings.size > 0:
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

def main():
    print("====================================")
    print("Voice AI Agent")
    print("  - faster-whisper (STT)")
    print("  - LM Studio (LLM)")
    print("  - RAG")
    print("  - Kokoro (TTS)")
    print("====================================")

    try:
        while True:
            # 1. Listen until silence
            audio_path = listen_until_silence()
            if audio_path is None:
                print("[*] Nothing captured, try again.")
                continue

            # 2. STT
            user_text = transcribe_audio(audio_path)
            os.remove(audio_path)

            if not user_text:
                print("[*] No speech detected. Try again.")
                continue

            if user_text.lower() in {"quit", "exit", "stop", "quit.", "exit.", "stop."}:
                print("Heard exit command, quitting.")
                break

            if user_text.lower() in {"add info", "add info."}:
                audio_path = listen_until_silence()
                if audio_path is None:
                    print("[*] Nothing captured, try again.")
                    continue
                user_text = transcribe_audio(audio_path)
                os.remove(audio_path)

                with open(KNOWLEDGE_DIR + "/info.txt", "a") as f:
                    f.write(user_text + "\n")
                print(f"{PINK}[INFO] Information has been added.{RESET}")
                texts, metadatas = load_documents(KNOWLEDGE_DIR)
                embeddings = embedder.encode(texts)
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(embeddings)
                continue

            if user_text.lower() in {"delete info", "delete info."}:
                user_input = input(f"{PINK}[INFO] Are you sure you want to delete? [y]/[n]: {RESET}")
                while user_input not in {"y", "n"}:
                    user_input = input(f"{PINK}[INFO] Unknown command. Enter 'y' for yes or 'n' for no: {RESET}")
                
                if user_input.lower() == "y":
                    if os.path.exists(KNOWLEDGE_DIR + "/info.txt"):
                        os.remove(KNOWLEDGE_DIR + "/info.txt")
                        print(f"{PINK}[INFO] Info deleted.{RESET}")
                        texts, metadatas = load_documents(KNOWLEDGE_DIR)
                        embeddings = embedder.encode(texts)
                        if embeddings.size > 0:
                            index = faiss.IndexFlatL2(embeddings.shape[1])
                            index.add(embeddings)
                        else: index = None
                    else:
                        print(f"{PINK}[INFO] Info does not exist.{RESET}")
                else:
                    print(f"{PINK}[INFO] Deletion cancelled.{RESET}")
                continue

            if user_text.lower() in {"print info", "print info."}:
                if os.path.exists(KNOWLEDGE_DIR + "/info.txt"):
                    with open(KNOWLEDGE_DIR + "/info.txt", "r") as f:
                        print(f"{PINK}[INFO] {f.read()}{RESET}")
                else:
                    print(f"{PINK}[INFO] Information does not exist.{RESET}")
                continue

            # 3. LLM
            print("[LLM] Thinking...")
            assistant_text = call_lmstudio(user_text)
            if assistant_text is None:
                print("[LLM] Failed to get a response.")
                continue

            conversation.append({"role": "assistant", "content": assistant_text})

            print(f"\n{GREEN}Assistant: {assistant_text}{RESET}\n")

            # 4. TTS
            # tts_generate_and_play_indexTTS2(assistant_text)
            tts_generate_and_play_kokoro(assistant_text)

    except KeyboardInterrupt:
        print("\n[!] Interrupted by user. Exiting.")


if __name__ == "__main__":
    main()
