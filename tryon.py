#!/usr/bin/env python3
"""
Kolors Virtual Try-On CLI — Interactive Menu
Uses Kwai-Kolors/Kolors-Virtual-Try-On HuggingFace Space via Gradio Queue API.
Photo-to-Video via Hugging Face Inference API (LTX-Video / Wan2.1).

Usage:
  python3 tryon.py          # interactive menu
  python3 tryon.py --cli    # direct CLI mode (old behavior)
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load .env from script directory
SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed. Run: pip install httpx questionary python-dotenv")
    sys.exit(1)

try:
    import questionary
    from questionary import Style
except ImportError:
    print("ERROR: questionary not installed. Run: pip install questionary")
    sys.exit(1)
try:
    from PIL import Image, ImageFilter, ImageEnhance
except ImportError:
    import subprocess
    print("📦 Installing Pillow...")
    pip_args = [sys.executable, "-m", "pip", "install", "Pillow", "-q"]
    try:
        subprocess.check_call(pip_args)
    except subprocess.CalledProcessError:
        # PEP 668 — try with --break-system-packages
        subprocess.check_call(pip_args + ["--break-system-packages"])
    from PIL import Image, ImageFilter, ImageEnhance


# ── Constants ──────────────────────────────────────────────────────────────────

BASE_URL = "https://kwai-kolors-kolors-virtual-try-on.hf.space"
MAX_SEED = 999999
DEFAULT_TIMEOUT = 180

MODELS_DIR = SCRIPT_DIR / "models"
GARMENTS_DIR = SCRIPT_DIR / "garments"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_VIDEO_DIR = SCRIPT_DIR / "results_video"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

for d in [MODELS_DIR, GARMENTS_DIR, RESULTS_DIR, RESULTS_VIDEO_DIR]:
    d.mkdir(exist_ok=True)

# Video Generation — Gradio Space (HF Inference API deprecated)
VIDEO_SPACE_URL = "https://presalesaiautomation-imagetovideo.hf.space"
VIDEO_SPACE_API = f"{VIDEO_SPACE_URL}/gradio_api"
VIDEO_SPACE_FN = 0  # generate_video_sync endpoint


class TokenManager:
    """
    HF Token Rotation — auto-switch to next token on ZeroGPU quota error.
    Reads HF_TOKENS (comma-separated) from env. Falls back to HF_TOKEN (legacy).
    """
    QUOTA_KEYWORDS = ["exceeded your free zerogpu quota", "zerogpu quota", "quota exceeded", "rate limit"]

    def __init__(self):
        raw = os.getenv("HF_TOKENS", "")
        if not raw:
            raw = os.getenv("HF_TOKEN", "")  # backward compat
        self._tokens = [t.strip() for t in raw.split(",") if t.strip()]
        self._index = 0

    @property
    def count(self) -> int:
        return len(self._tokens)

    @property
    def current(self) -> str:
        if not self._tokens:
            return ""
        return self._tokens[self._index]

    @property
    def current_masked(self) -> str:
        t = self.current
        if len(t) <= 8:
            return t
        return f"...{t[-6:]}"

    def headers(self) -> dict:
        """Return Authorization header dict for current token."""
        h = {}
        if self.current:
            h["Authorization"] = f"Bearer {self.current}"
        return h

    def is_quota_error(self, error: Exception) -> bool:
        """Check if an error is a ZeroGPU quota exhaustion."""
        msg = str(error).lower()
        return any(kw in msg for kw in self.QUOTA_KEYWORDS)

    def rotate(self) -> bool:
        """
        Switch to next token. Returns True if rotation succeeded,
        False if no more tokens available.
        """
        if len(self._tokens) <= 1:
            return False
        old_idx = self._index
        self._index = (self._index + 1) % len(self._tokens)
        print(f"  [!] Kuota token habis, mencoba beralih ke Token ke-{self._index + 1}/{len(self._tokens)} ({self.current_masked})...")
        return True

    def __str__(self):
        return f"TokenManager({self.count} tokens, active={self.current_masked})"


# Initialize global token manager
HF_TOKEN_MANAGER = TokenManager()

VIDEO_MODELS = {
    "AI Video Sync (recommended)": "preSalesAIAutomation/imagetovideo",
}

VIDEO_DURATIONS = {
    "2 detik": 2.0,
    "3.5 detik (default)": 3.5,
    "5 detik": 5.0,
    "8 detik": 8.0,
}
VIDEO_PROMPTS = {
    "🔄 360° Spin (Mutar Pelan)": "A 360-degree slow rotation showcasing the outfit. Close-up on the textures, pleats, and fit of the clothing. The camera stays strictly focused from the neck down to the waist. Smooth movement, bright clean background, no face visible.",
    "🔍 Macro Detail (Serat Kain)": "Cinematic slow-motion video with a macro close-up focus on the clothing texture and stitching details. The camera smoothly glides sideways across the fabric, capturing the light reflecting on the material. Strict framing from chest to waist, no face visible, clean and elegant background.",
    "💨 Gentle Flow (Ayunan Baju)": "Elegant studio shot with a subtle breeze causing the clothing to flow gently. Cinematic slow motion capturing the natural movement and drapery of the outfit. The camera remains centered on the torso, tracking the moving fabric. Bright studio lighting, clean backdrop, strictly no face in frame.",
    "🎛 Cinematic Slide (Geser Kiri-Kanan)": "A professional fashion lookbook video with a smooth horizontal camera slide (panning) from left to right. The focus is entirely on the fit, pleats, and silhouette of the garment. Close-up framing from the neck down to the hips. Bright, minimalist background, smooth motion, no face visible.",
    "✍️ Custom Prompt (Ketik Manual)": "__custom__",
}

DEFAULT_NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"

# Upscale — Gradio Space (Real-ESRGAN)
UPSCALE_SPACE_URL = "https://fabrice-tiercelin-realesrgan.hf.space"
UPSCALE_SPACE_API = f"{UPSCALE_SPACE_URL}/gradio_api"
UPSCALE_SPACE_FN = 2  # predict: image + radio -> upscaled image
UPSCALE_HF_TOKEN_MANAGER = HF_TOKEN_MANAGER  # reuse same manager
UPSCALE_SCALES = {"x2": 2, "x4": 4, "x8": 8}

custom_style = Style([
    ("qmark",       "fg:#673ab7 bold"),
    ("question",    "bold"),
    ("answer",      "fg:#00c853 bold"),
    ("pointer",     "fg:#673ab7 bold"),
    ("highlighted", "fg:#673ab7 bold"),
    ("selected",    "fg:#00c853"),
    ("separator",   "fg:#666666"),
    ("instruction", "fg:#999999"),
    ("text",        ""),
])


# ── Core API — Try-On ─────────────────────────────────────────────────────────

def upload_image(client: httpx.Client, image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    ext = path.suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    mime = mime_map.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        resp = client.post(f"{BASE_URL}/upload", files={"files": (path.name, f, mime)})
    resp.raise_for_status()
    uploaded = resp.json()
    if not uploaded:
        raise RuntimeError(f"Upload failed (HTTP {resp.status_code}): {resp.text[:200]}")
    return uploaded[0]


def upload_from_url(client: httpx.Client, url: str) -> str:
    # SSRF protection: validate URL scheme and block private IPs
    from urllib.parse import urlparse
    import ipaddress
    parsed = urlparse(url)
    if parsed.scheme not in ("https", "http"):
        raise ValueError(f"Invalid URL scheme: {parsed.scheme}")
    hostname = parsed.hostname or ""
    if hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
        raise ValueError("Cannot upload from localhost URLs")
    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            raise ValueError(f"Cannot upload from private IP: {hostname}")
    except ValueError as e:
        if "Cannot upload" in str(e):
            raise
    resp = client.get(url, follow_redirects=True, timeout=30)
    resp.raise_for_status()
    fname = url.split("?")[0].split("/")[-1] or "image.jpg"
    content_type = resp.headers.get("content-type", "image/jpeg")
    if "png" in content_type:
        fname = fname if fname.endswith(".png") else "image.png"
    elif "webp" in content_type:
        fname = fname if fname.endswith(".webp") else "image.webp"
    upload_resp = client.post(f"{BASE_URL}/upload", files={"files": (fname, resp.content, content_type)})
    upload_resp.raise_for_status()
    uploaded = upload_resp.json()
    if not uploaded:
        raise RuntimeError(f"Upload failed (HTTP {upload_resp.status_code}): {upload_resp.text[:200]}")
    return uploaded[0]


def submit_tryon(
    person_path: str,
    garment_path: str,
    seed: int = 42,
    randomize_seed: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    output_path: str = "result.png",
    verbose: bool = True,
) -> dict:
    """Run virtual try-on. Returns dict with result info."""
    with httpx.Client(timeout=60) as client:
        if verbose:
            print("  ⬆️  Uploading person image...")
        if person_path.startswith(("http://", "https://")):
            person_remote = upload_from_url(client, person_path)
        else:
            person_remote = upload_image(client, person_path)

        if verbose:
            print("  ⬆️  Uploading garment image...")
        if garment_path.startswith(("http://", "https://")):
            garment_remote = upload_from_url(client, garment_path)
        else:
            garment_remote = upload_image(client, garment_path)

        session_hash = f"cli_{int(time.time())}_{os.getpid()}"
        payload = {
            "data": [
                {"path": person_remote, "meta": {"_type": "gradio.FileData"}},
                {"path": garment_remote, "meta": {"_type": "gradio.FileData"}},
                seed,
                randomize_seed,
            ],
            "fn_index": 2,
            "session_hash": session_hash,
        }

        if verbose:
            print(f"  📤 Submitting to queue (seed={seed}, random={randomize_seed})...")

        resp = client.post(f"{BASE_URL}/queue/join", json=payload, timeout=30)
        resp.raise_for_status()

        sse_url = f"{BASE_URL}/queue/data?session_hash={session_hash}"
        start = time.time()
        result_path = None
        result_seed = None

        with client.stream("GET", sse_url, timeout=(5, timeout)) as sse_resp:
            sse_resp.raise_for_status()
            for raw_line in sse_resp.iter_lines():
                elapsed = time.time() - start
                if elapsed > timeout:
                    raise TimeoutError(f"No result after {timeout}s")

                line = raw_line.strip()
                if not line:
                    continue

                if line.startswith("data:"):
                    data_str = line.split(":", 1)[1].strip()
                    try:
                        evt = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    msg = evt.get("msg", "")

                    if msg == "estimation":
                        rank = evt.get("rank", "?")
                        eta = evt.get("rank_eta", 0)
                        if verbose:
                            print(f"  ⏳ Queue rank: {rank}, ETA: {eta:.0f}s")

                    elif msg == "process_starts":
                        if verbose:
                            eta_val = evt.get("eta", 0)
                            print(f"  🔄 Processing (ETA: {eta_val:.0f}s)...")

                    elif msg == "heartbeat":
                        if verbose:
                            print(f"  💓 [{elapsed:.0f}s]")

                    elif msg == "process_completed":
                        output = evt.get("output", {})
                        success = evt.get("success", False)
                        if not success:
                            error_msg = output.get("error", "Unknown error")
                            raise RuntimeError(f"Try-on failed: {str(error_msg)[:300]}")

                        data_items = output.get("data", [])
                        if not data_items:
                            raise RuntimeError("No output data returned")

                        for item in data_items:
                            if isinstance(item, dict) and "url" in item:
                                img_url = item["url"]
                                if not img_url.startswith("http"):
                                    img_url = BASE_URL + img_url
                                img_resp = client.get(img_url, follow_redirects=True, timeout=30)
                                img_resp.raise_for_status()
                                with open(output_path, "wb") as f:
                                    f.write(img_resp.content)
                                result_path = output_path
                            elif isinstance(item, (int, float)):
                                result_seed = int(item)
                        break

                    elif msg == "close_stream":
                        break

        elapsed = time.time() - start
        if not result_path:
            raise RuntimeError("No result received")

        fsize = os.path.getsize(result_path)
        return {
            "path": result_path,
            "seed": result_seed,
            "elapsed": elapsed,
            "size": fsize,
            "person": person_path,
            "garment": garment_path,
        }


# ── Core API — Photo to Video ─────────────────────────────────────────────────

def _upload_video_image(client: httpx.Client, image_path: str) -> str:
    """Upload image to video Gradio Space. Returns remote path."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    ext = path.suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    mime = mime_map.get(ext, "image/png")
    with open(path, "rb") as f:
        resp = client.post(f"{VIDEO_SPACE_API}/upload", files={"files": (path.name, f, mime)}, timeout=30)
    resp.raise_for_status()
    uploaded = resp.json()
    if not uploaded:
        raise RuntimeError(f"Upload failed (HTTP {resp.status_code}): {resp.text[:200]}")
    return uploaded[0]


def _poll_sse(client: httpx.Client, session_hash: str, timeout: int, verbose: bool = True) -> dict:
    """Poll Gradio SSE stream. Returns parsed output dict."""
    sse_url = f"{VIDEO_SPACE_API}/queue/data?session_hash={session_hash}"
    start = time.time()

    headers = HF_TOKEN_MANAGER.headers()

    with client.stream("GET", sse_url, timeout=(5, timeout), headers=headers) as sse_resp:
        sse_resp.raise_for_status()
        for raw_line in sse_resp.iter_lines():
            elapsed = time.time() - start
            if elapsed > timeout:
                raise TimeoutError(f"No result after {timeout}s")

            line = raw_line.strip()
            if not line or not line.startswith("data:"):
                continue

            data_str = line.split(":", 1)[1].strip()
            try:
                evt = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            msg = evt.get("msg", "")

            if msg == "estimation":
                rank = evt.get("rank", "?")
                eta = evt.get("rank_eta", 0)
                if verbose:
                    print(f"  ⏳ Queue rank: {rank}, ETA: {eta:.0f}s")

            elif msg == "process_starts":
                if verbose:
                    print(f"  🔄 Processing [{elapsed:.0f}s]...")

            elif msg == "heartbeat":
                if verbose:
                    print(f"  💓 [{elapsed:.0f}s]")

            elif msg == "process_completed":
                output = evt.get("output", {})
                success = evt.get("success", False)
                if not success:
                    error_msg = output.get("error", "Unknown error")
                    raise RuntimeError(f"Generation failed: {str(error_msg)[:300]}")
                return output

            elif msg == "close_stream":
                break

    raise RuntimeError("SSE stream closed without process_completed")


def hf_image_to_video(
    image_path: str,
    prompt: str,
    model_id: str = "preSalesAIAutomation/imagetovideo",
    duration: float = 3.5,
    negative_prompt: str = "",
    seed: int = 42,
    randomize_seed: bool = True,
    steps: int = 6,
    timeout: int = 600,
    verbose: bool = True,
) -> str:
    """
    Generate video from image using Gradio Space via raw Queue API.
    Returns path to saved .mp4 file.
    """
    if not negative_prompt:
        negative_prompt = DEFAULT_NEGATIVE_PROMPT

    session_hash = f"cli_{int(time.time())}_{os.getpid()}"

    start = time.time()
    if verbose:
        dur_str = f"{duration:.0f}" if duration == int(duration) else f"{duration:.1f}"
        print(f"  🤖 Model   : {model_id}")
        print(f"  📝 Prompt  : {prompt}")
        print(f"  ⏱️  Duration: {dur_str}s")

    headers = HF_TOKEN_MANAGER.headers()

    with httpx.Client(timeout=600) as client:
        # 1. Upload image
        if verbose:
            print("  ⬆️  Uploading image...")
        remote_path = _upload_video_image(client, image_path)
        if verbose:
            print(f"  ✅ Uploaded: {remote_path}")

        # 2. Join queue + poll with token rotation on quota error
        max_retries = HF_TOKEN_MANAGER.count
        for attempt in range(max_retries):
            headers = HF_TOKEN_MANAGER.headers()
            session_hash = f"cli_{int(time.time())}_{os.getpid()}_{attempt}"
            payload = {
                "data": [
                    {"path": remote_path, "meta": {"_type": "gradio.FileData"}},  # [5] Input Image
                    prompt,               # [6] prompt
                    steps,                # [12] steps
                    negative_prompt,      # [9] negative_prompt
                    duration,             # [7] duration_seconds
                    1,                    # [13] guidance_scale
                    1,                    # [14] guidance_scale_2
                    seed,                 # [10] seed
                    randomize_seed,       # [11] randomize
                ],
                "fn_index": VIDEO_SPACE_FN,
                "session_hash": session_hash,
            }

            if verbose:
                print(f"  📤 Submitting to queue... (Token {HF_TOKEN_MANAGER._index + 1}/{HF_TOKEN_MANAGER.count})")

            try:
                resp = client.post(f"{VIDEO_SPACE_API}/queue/join", json=payload, timeout=30, headers=headers)
                resp.raise_for_status()
                output = _poll_sse(client, session_hash, timeout, verbose)
                break  # success — exit retry loop
            except Exception as e:
                if HF_TOKEN_MANAGER.is_quota_error(e) and attempt < max_retries - 1:
                    if not HF_TOKEN_MANAGER.rotate():
                        raise RuntimeError(f"Semua token habis kuota! Error: {e}")
                    continue
                raise

        # 4. Download video
        data_items = output.get("data", [])
        if not data_items:
            raise RuntimeError("No output data returned")

        video_url = None
        for item in data_items:
            if isinstance(item, dict):
                # Nested: {"video": {"url": "...", "path": "..."}, "subtitles": null}
                if "video" in item and isinstance(item["video"], dict):
                    video_url = item["video"].get("url") or item["video"].get("path")
                    break
                # Flat: {"url": "..."}
                if "url" in item:
                    video_url = item["url"]
                    break
                if "path" in item:
                    video_url = item["path"]
                    break

        if not video_url:
            raise RuntimeError(f"No video URL in output (got {len(data_items)} items)")

        if not video_url.startswith("http"):
            video_url = VIDEO_SPACE_API + "/file=" + video_url

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"video_{ts}.mp4"
        out_path = RESULTS_VIDEO_DIR / out_name

        if verbose:
            print(f"  ⬇️  Downloading video...")

        vid_resp = client.get(video_url, follow_redirects=True, timeout=120, headers=HF_TOKEN_MANAGER.headers())
        vid_resp.raise_for_status()

        with open(out_path, "wb") as f:
            f.write(vid_resp.content)

    elapsed = time.time() - start
    fsize = os.path.getsize(out_path)

    if verbose:
        elapsed_str = f"{elapsed:.1f}"
        size_kb = fsize / 1024
        size_mb = size_kb / 1024
        print()
        print("  ╭────────── ✅ Video Ready! ──────────╮")
        print(f"  │  Output : {out_name:<24} │")
        if size_mb >= 1:
            size_str = f"{size_mb:.1f}"
            pad = max(0, 20 - len(size_str) - 2)
            print(f"  │  Size   : {size_str} MB{' ' * pad}│")
        else:
            size_str = f"{size_kb:.0f}"
            pad = max(0, 20 - len(size_str) - 2)
            print(f"  │  Size   : {size_str} KB{' ' * pad}│")
        pad_e = max(0, 20 - len(elapsed_str) - 1)
        print(f"  │  Time   : {elapsed_str}s{' ' * pad_e}│")
        print("  ╰─────────────────────────────────────╯")
        print()

    return str(out_path)



# ── Core API — Image Upscale ──────────────────────────────────────────────────

def upscale_pillow(
    image_path: str,
    scale: int = 4,
    output_path: str = "",
    verbose: bool = True,
) -> str:
    """
    Fast local upscale using Pillow (Lanczos + sharpen).
    ~2 seconds, works on any machine, good quality for 2-4x.
    """
    if Image is None:
        raise RuntimeError("Pillow import failed unexpectedly. Try: pip install --force-reinstall Pillow")
    if scale not in (2, 4, 8):
        raise ValueError(f"Scale must be 2, 4, or 8, got {scale}")

    if not output_path:
        p = Path(image_path)
        output_path = str(p.parent / f"{p.stem}_x{scale}{p.suffix}")

    start = time.time()

    img = Image.open(image_path)
    orig_w, orig_h = img.size
    new_w, new_h = orig_w * scale, orig_h * scale

    if verbose:
        print(f"  🔍 Upscale   : x{scale} ({orig_w}×{orig_h} → {new_w}×{new_h})")

    # Step 1: High-quality Lanczos resize
    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Step 2: UnsharpMask for edge sharpening
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    # Step 3: Subtle contrast boost
    img = ImageEnhance.Contrast(img).enhance(1.1)

    # Step 4: Minor detail enhancement
    img = img.filter(ImageFilter.DETAIL)

    # Save
    if output_path.lower().endswith(".webp"):
        img.save(output_path, "WEBP", quality=95)
    elif output_path.lower().endswith(".png"):
        img.save(output_path, "PNG", optimize=True)
    else:
        img.save(output_path, "JPEG", quality=95, optimize=True)

    elapsed = time.time() - start
    fsize = os.path.getsize(output_path)
    new_w2, new_h2 = img.size

    if verbose:
        size_kb = fsize / 1024
        size_mb = size_kb / 1024
        print()
        print("  ╭────────── ✅ Upscaled! ──────────╮")
        print(f"  │  Output : {Path(output_path).name:<20} │")
        print(f"  │  Scale  : x{scale} ({new_w2}×{new_h2}){' ' * max(0, 11 - len(f'{new_w2}×{new_h2}'))}│")
        print(f"  │  Method : Pillow Lanczos{' ' * 6}│")
        if size_mb >= 1:
            print(f"  │  Size   : {size_mb:.1f} MB{' ' * 15}│")
        else:
            print(f"  │  Size   : {size_kb:.0f} KB{' ' * max(0, 15 - len(f'{size_kb:.0f}'))}│")
        elapsed_str = f"{elapsed:.2f}"
        print(f"  │  Time   : {elapsed_str}s{' ' * max(0, 17 - len(elapsed_str))}│")
        print("  ╰─────────────────────────────────╯")
        print()

    return output_path


def _upload_upscale_image(client: httpx.Client, image_path: str) -> str:
    """Upload image to HF Gradio Space for AI upscale."""
    path = Path(image_path)
    ext = path.suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    mime = mime_map.get(ext, "image/png")
    with open(path, "rb") as f:
        resp = client.post(f"{UPSCALE_SPACE_API}/upload", files={"files": (path.name, f, mime)}, timeout=30)
    resp.raise_for_status()
    uploaded = resp.json()
    if not uploaded:
        raise RuntimeError(f"Upload failed (HTTP {resp.status_code}): {resp.text[:200]}")
    return uploaded[0]


def _poll_upscale_sse(client: httpx.Client, session_hash: str, timeout: int, verbose: bool = True) -> dict:
    """Poll Gradio SSE stream for AI upscale."""
    sse_url = f"{UPSCALE_SPACE_API}/queue/data?session_hash={session_hash}"
    start = time.time()
    headers = HF_TOKEN_MANAGER.headers()

    with client.stream("GET", sse_url, timeout=(5, timeout), headers=headers) as sse_resp:
        sse_resp.raise_for_status()
        for raw_line in sse_resp.iter_lines():
            elapsed = time.time() - start
            if elapsed > timeout:
                raise TimeoutError(f"No result after {timeout}s")
            line = raw_line.strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line.split(":", 1)[1].strip()
            try:
                evt = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            msg = evt.get("msg", "")
            if msg == "estimation":
                rank = evt.get("rank", "?")
                eta = evt.get("rank_eta", 0)
                if verbose:
                    print(f"  ⏳ Queue rank: {rank}, ETA: {eta:.0f}s")
            elif msg == "process_starts":
                if verbose:
                    print(f"  🔄 AI processing [{elapsed:.0f}s]...")
            elif msg == "heartbeat":
                if verbose:
                    print(f"  💓 [{elapsed:.0f}s]")
            elif msg == "process_completed":
                output = evt.get("output", {})
                if not evt.get("success", False):
                    raise RuntimeError(f"AI upscale failed: {output.get('error', '?')}")
                return output
            elif msg == "close_stream":
                break
    raise RuntimeError("SSE closed without process_completed")


def upscale_ai(
    image_path: str,
    scale: int = 4,
    output_path: str = "",
    timeout: int = 600,
    verbose: bool = True,
) -> str:
    """
    AI upscale via Real-ESRGAN HF Gradio Space.
    Better quality than Pillow but slower (~5-18 min, CPU queue).
    Falls back to Pillow on failure.
    """
    if scale not in (2, 4, 8):
        raise ValueError(f"Scale must be 2, 4, or 8, got {scale}")

    if not output_path:
        p = Path(image_path)
        output_path = str(p.parent / f"{p.stem}_x{scale}_ai{p.suffix}")

    if verbose:
        print(f"  🤖 AI Upscale : x{scale} (Real-ESRGAN, ~5-18 min)")

    start = time.time()
    headers = HF_TOKEN_MANAGER.headers()

    with httpx.Client(timeout=300) as client:
        if verbose:
            print("  ⬆️  Uploading image...")
        remote_path = _upload_upscale_image(client, image_path)

        # Queue join + poll with token rotation on quota error
        max_retries = HF_TOKEN_MANAGER.count
        output = None
        for attempt in range(max_retries):
            headers = HF_TOKEN_MANAGER.headers()
            session_hash = f"ups_{int(time.time())}_{os.getpid()}_{attempt}"
            payload = {
                "data": [
                    {"path": remote_path, "meta": {"_type": "gradio.FileData"}},
                    scale,
                ],
                "fn_index": UPSCALE_SPACE_FN,
                "session_hash": session_hash,
            }

            if verbose:
                print(f"  📤 Submitting to queue (x{scale})... (Token {HF_TOKEN_MANAGER._index + 1}/{HF_TOKEN_MANAGER.count})")
            try:
                resp = client.post(f"{UPSCALE_SPACE_API}/queue/join", json=payload, timeout=30, headers=headers)
                resp.raise_for_status()
                output = _poll_upscale_sse(client, session_hash, timeout, verbose)
                break  # success
            except Exception as e:
                if HF_TOKEN_MANAGER.is_quota_error(e) and attempt < max_retries - 1:
                    if not HF_TOKEN_MANAGER.rotate():
                        raise RuntimeError(f"Semua token habis kuota! Error: {e}")
                    continue
                raise
        if output is None:
            raise RuntimeError("Upscale failed after all token attempts")

        data_items = output.get("data", [])
        if not data_items:
            raise RuntimeError("No upscaled image returned")

        img_url = None
        for item in data_items:
            if isinstance(item, dict) and "url" in item:
                img_url = item["url"]
                break
        if not img_url:
            raise RuntimeError(f"No image URL in output (got {len(data_items)} items)")
        if not img_url.startswith("http"):
            img_url = UPSCALE_SPACE_API + "/file=" + img_url

        if verbose:
            print("  ⬇️  Downloading...")
        img_resp = client.get(img_url, follow_redirects=True, timeout=60, headers=HF_TOKEN_MANAGER.headers())
        img_resp.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(img_resp.content)

    elapsed = time.time() - start
    fsize = os.path.getsize(output_path)
    if verbose:
        size_kb = fsize / 1024
        print()
        print("  ╭────────── ✅ AI Upscaled! ─────────╮")
        print(f"  │  Output : {Path(output_path).name:<20} │")
        print(f"  │  Scale  : x{scale} (Real-ESRGAN){' ' * max(0, 10 - len(f'x{scale} (Real-ESRGAN)'))}│")
        if size_kb >= 1024:
            print(f"  │  Size   : {size_kb/1024:.1f} MB{' ' * 15}│")
        else:
            print(f"  │  Size   : {size_kb:.0f} KB{' ' * max(0, 15 - len(f'{size_kb:.0f}'))}│")
        elapsed_str = f"{elapsed:.1f}"
        print(f"  │  Time   : {elapsed_str}s{' ' * max(0, 16 - len(elapsed_str))}│")
        print("  ╰─────────────────────────────────────╯")
        print()
    return output_path


def upscale_image(
    image_path: str,
    scale: int = 4,
    output_path: str = "",
    mode: str = "fast",
    timeout: int = 600,
    verbose: bool = True,
) -> str:
    """
    Upscale image. mode='fast' uses Pillow (~2s), mode='ai' uses Real-ESRGAN (~5-18 min).
    'auto' tries AI first, falls back to fast on failure.
    """
    if mode == "ai":
        return upscale_ai(image_path, scale, output_path, timeout, verbose)
    elif mode == "auto":
        try:
            return upscale_ai(image_path, scale, output_path, timeout, verbose)
        except Exception as e:
            if verbose:
                print(f"  ⚠️  AI upscale failed, falling back to Pillow: {e}")
            return upscale_pillow(image_path, scale, output_path, verbose)
    else:  # "fast" or default
        return upscale_pillow(image_path, scale, output_path, verbose)


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_images(directory: Path) -> list[Path]:
    return sorted(f for f in directory.iterdir() if f.suffix.lower() in IMAGE_EXTS)


def timestamp_filename() -> str:
    return datetime.now().strftime("tryon_%Y%m%d_%H%M%S") + ".png"


def print_header():
    token_count = HF_TOKEN_MANAGER.count
    token_info = f"  🔑 Token Rotation: {token_count} token(s) siap" if token_count > 1 else "  🔑 1 token (tambah token lain untuk rotasi otomatis)"
    print()
    print("╔═══════════════════════════════════════════════╗")
    print("║       👕  Kolors Virtual Try-On CLI  👕      ║")
    print("║   Kwai-Kolors HuggingFace Space — CLI Menu   ║")
    print("╚═══════════════════════════════════════════════╝")
    print(token_info)
    print()


def print_result(result: dict):
    print()
    print("  ╭────────── ✅ Success! ──────────╮")
    print(f"  │  Output : {result['path']:<20} │")
    print(f"  │  Seed   : {result['seed']:<20} │")
    elapsed_str = f"{result['elapsed']:.1f}"
    pad_e = max(0, 19 - len(elapsed_str))
    print(f"  │  Time   : {elapsed_str}s{' ' * pad_e}│")
    size_str = f"{result['size']:,}"
    pad = max(0, 13 - len(size_str))
    print(f"  │  Size   : {size_str} bytes{' ' * pad}│")
    print("  ╰─────────────────────────────────╯")
    print()


# ── Sub-menus ──────────────────────────────────────────────────────────────────

def menu_try_on(state: dict = None):
    """Try-on workflow sub-menu."""
    if state is None:
        state = {"auto_upscale": True, "upscale_scale": 4}
    selected_model = None
    selected_garment = None
    seed = 42
    random_seed = False

    while True:
        model_label = selected_model.name if selected_model else "(belum dipilih)"
        garment_label = selected_garment.name if selected_garment else "(belum dipilih)"
        seed_label = "Random" if random_seed else str(seed)

        choices = [
            questionary.Choice(
                title=f"📷  Pilih Model    → {model_label}",
                value="model",
            ),
            questionary.Choice(
                title=f"👔  Pilih Garment  → {garment_label}",
                value="garment",
            ),
            questionary.Choice(
                title=f"🌱  Set Seed       → {seed_label}",
                value="seed",
            ),
            questionary.Choice(
                title=f"🎲  Random Seed    → {'ON' if random_seed else 'OFF'}",
                value="toggle_random",
            ),
            questionary.Separator(),
            questionary.Choice(
                title="▶️   RUN Try-On",
                value="run",
                disabled=(
                    "pilih model & garment dulu"
                    if (selected_model is None or selected_garment is None)
                    else False
                ),
            ),
            questionary.Choice(
                title="←  Kembali ke Menu Utama",
                value="back",
            ),
        ]

        action = questionary.select(
            "  Try-On Menu:",
            choices=choices,
            style=custom_style,
            use_shortcuts=True,
        ).ask()

        if action is None or action == "back":
            return

        elif action == "model":
            imgs = get_images(MODELS_DIR)
            if not imgs:
                print("\n  ⚠️  Folder models/ kosong! Taruh foto orang di sana.\n")
                continue
            pick = questionary.select(
                "  Pilih foto model:",
                choices=[questionary.Choice(title=f.name, value=f) for f in imgs],
                style=custom_style,
            ).ask()
            if pick:
                selected_model = pick

        elif action == "garment":
            imgs = get_images(GARMENTS_DIR)
            if not imgs:
                print("\n  ⚠️  Folder garments/ kosong! Taruh foto baju di sana.\n")
                continue
            pick = questionary.select(
                "  Pilih foto baju:",
                choices=[questionary.Choice(title=f.name, value=f) for f in imgs],
                style=custom_style,
            ).ask()
            if pick:
                selected_garment = pick

        elif action == "seed":
            val = questionary.text(
                "  Seed (0-999999):",
                default=str(seed),
                validate=lambda t: t.isdigit() and 0 <= int(t) <= MAX_SEED,
                style=custom_style,
            ).ask()
            if val is not None:
                seed = int(val)
                random_seed = False

        elif action == "toggle_random":
            random_seed = not random_seed

        elif action == "run":
            if selected_model is None or selected_garment is None:
                continue

            out_name = timestamp_filename()
            out_path = RESULTS_DIR / out_name

            print()
            print("  🚀 Starting Try-On...")
            print(f"  Model   : {selected_model.name}")
            print(f"  Garment : {selected_garment.name}")
            print(f"  Seed    : {'Random' if random_seed else seed}")
            print()

            try:
                result = submit_tryon(
                    person_path=str(selected_model),
                    garment_path=str(selected_garment),
                    seed=seed,
                    randomize_seed=random_seed,
                    output_path=str(out_path),
                    verbose=True,
                )
                print_result(result)

                # Auto-upscale after try-on
                if state.get("auto_upscale"):
                    scale = state.get("upscale_scale", 4)
                    mode = state.get("upscale_mode", "fast")
                    mode_label = {"fast": "Pillow", "ai": "HF AI", "auto": "Auto"}.get(mode, mode)
                    print(f"  🔍 Auto-upscaling x{scale} ({mode_label})...")
                    try:
                        upscaled_path = upscale_image(
                            image_path=result["path"],
                            scale=scale,
                            mode=mode,
                            verbose=True,
                        )
                        print(f"  💎 HD result saved: {upscaled_path}")
                    except Exception as ups_err:
                        print(f"  ⚠️  Upscale failed (try-on result still saved): {ups_err}")
            except Exception as e:
                print(f"\n  ❌ Error: {e}\n")

            cont = questionary.confirm("Mau coba lagi?", default=True, style=custom_style).ask()
            if not cont:
                return


def menu_photo_to_video():
    """Photo-to-Video workflow sub-menu."""
    # Scan results/ for images
    result_images = get_images(RESULTS_DIR)
    if not result_images:
        print("\n  ⚠️  Folder results/ kosong! Jalankan Try-On dulu buat dapetin foto.\n")
        return

    # 1. Select image
    pick = questionary.select(
      "  Pilih foto dari results/:",
      choices=[questionary.Choice(title=f.name, value=f) for f in result_images],
      style=custom_style,
    ).ask()
    if pick is None:
        return

    # 2. Select prompt preset
    preset_choice = questionary.select(
      "  Pilih gerakan video:",
      choices=[questionary.Choice(title=k, value=v) for k, v in VIDEO_PROMPTS.items()],
      style=custom_style,
    ).ask()
    if preset_choice is None:
        return

    if preset_choice == "__custom__":
        prompt = questionary.text(
          "  Prompt (ketik sendiri):",
          default="The person starts walking towards the camera with a gentle smile",
          style=custom_style,
        ).ask()
        if not prompt:
            return
    else:
        prompt = preset_choice

    # 3. Select model
    model_choice = questionary.select(
      "  Pilih model:",
      choices=[questionary.Choice(title=k, value=v) for k, v in VIDEO_MODELS.items()],
      style=custom_style,
    ).ask()
    if model_choice is None:
        return

    # 4. Select duration
    duration_choice = questionary.select(
      "  Pilih durasi:",
      choices=[questionary.Choice(title=k, value=v) for k, v in VIDEO_DURATIONS.items()],
      style=custom_style,
    ).ask()
    if duration_choice is None:
        return

    # 5. Run
    print()
    print("  🎬 Starting Photo to Video...")
    print(f"  Image   : {pick.name}")
    prompt_short = prompt[:60] + "..." if len(prompt) > 60 else prompt
    print(f"  Prompt  : {prompt_short}")
    print(f"  Model   : AI Video Sync")
    dur_label = f"{duration_choice:.0f}" if duration_choice == int(duration_choice) else f"{duration_choice:.1f}"
    print(f"  Duration: {dur_label}s")
    print()

    try:
        out_path = hf_image_to_video(
            image_path=str(pick),
            prompt=prompt,
            duration=duration_choice,
            verbose=True,
        )
        print(f"  💾 Saved: {out_path}")
    except Exception as e:
        print(f"\n  ❌ Error: {e}\n")

    cont = questionary.confirm("Mau coba lagi?", default=True, style=custom_style).ask()
    if cont:
        menu_photo_to_video()


def menu_view_files():
    """View files in all folders."""
    sections = [
        ("📁 models/", MODELS_DIR),
        ("📁 garments/", GARMENTS_DIR),
        ("📁 results/", RESULTS_DIR),
        ("📁 results_video/", RESULTS_VIDEO_DIR),
    ]

    print()
    for label, directory in sections:
        print(f"  {label}")
        if not directory.exists():
            print("     (folder belum ada)")
            continue
        imgs = sorted(
            f for f in directory.iterdir()
            if f.suffix.lower() in IMAGE_EXTS or f.name == "README.md"
        )
        if not imgs:
            print("     (kosong)")
        else:
            for f in imgs:
                size = f.stat().st_size
                mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                print(f"     {f.name:<30} {size:>8,} bytes   {mtime}")
        print()

    questionary.press_any_key_to_continue(style=custom_style).ask()


def menu_settings(state: dict):
    """Settings sub-menu."""
    while True:
        action = questionary.select(
            "  Settings:",
            choices=[
                questionary.Choice(
                    title=f"🌱  Default Seed      → {state['seed']}",
                    value="seed",
                ),
                questionary.Choice(
                    title=f"🎲  Random Seed       → {'ON' if state['random_seed'] else 'OFF'}",
                    value="random",
                ),
                questionary.Choice(
                    title=f"⏱️   Timeout           → {state['timeout']}s",
                    value="timeout",
                ),
                questionary.Choice(
                    title=f"🔍  Auto Upscale     → {'ON' if state['auto_upscale'] else 'OFF'} (x{state['upscale_scale']})",
                    value="upscale",
                ),
                questionary.Separator(),
                questionary.Choice(title="←  Kembali", value="back"),
            ],
            style=custom_style,
        ).ask()

        if action is None or action == "back":
            return

        elif action == "seed":
            val = questionary.text(
                "  Seed (0-999999):",
                default=str(state["seed"]),
                validate=lambda t: t.isdigit() and 0 <= int(t) <= MAX_SEED,
                style=custom_style,
            ).ask()
            if val is not None:
                state["seed"] = int(val)
                state["random_seed"] = False

        elif action == "random":
            state["random_seed"] = not state["random_seed"]

        elif action == "timeout":
            val = questionary.text(
                "  Timeout (detik):",
                default=str(state["timeout"]),
                validate=lambda t: t.isdigit() and int(t) > 0,
                style=custom_style,
            ).ask()
            if val is not None:
                state["timeout"] = int(val)

        elif action == "upscale":
            sub = questionary.select(
                "  Upscale settings:",
                choices=[
                    questionary.Choice(
                        title=f"  Toggle Auto Upscale → {'ON' if state['auto_upscale'] else 'OFF'}",
                        value="toggle",
                    ),
                    questionary.Choice(
                        title=f"  Scale → x{state['upscale_scale']}",
                        value="scale",
                    ),
                    questionary.Choice(
                        title=f"  Mode → {state['upscale_mode'].upper()} ({'Pillow (fast)' if state['upscale_mode'] == 'fast' else 'HF Real-ESRGAN (~5-18 min)' if state['upscale_mode'] == 'ai' else 'AI → Pillow fallback'})",
                        value="mode",
                    ),
                    questionary.Separator(),
                    questionary.Choice(title="←  Kembali", value="back"),
                ],
                style=custom_style,
            ).ask()
            if sub == "toggle":
                state["auto_upscale"] = not state["auto_upscale"]
            elif sub == "scale":
                s = questionary.select(
                    "  Pilih scale:",
                    choices=[
                        questionary.Choice(title="x2 (cepat, ringan)", value=2),
                        questionary.Choice(title="x4 (recommended)", value=4),
                        questionary.Choice(title="x8 (super tajam, besar)", value=8),
                    ],
                    style=custom_style,
                ).ask()
                if s is not None:
                    state["upscale_scale"] = s
            elif sub == "mode":
                m = questionary.select(
                    "  Upscale mode:",
                    choices=[
                        questionary.Choice(title="⚡  Fast (Pillow Lanczos, ~2s)", value="fast"),
                        questionary.Choice(title="🤖 AI (HF Real-ESRGAN, ~5-18 min)", value="ai"),
                        questionary.Choice(title="🔄 Auto (AI → fallback Pillow)", value="auto"),
                    ],
                    style=custom_style,
                ).ask()
                if m is not None:
                    state["upscale_mode"] = m


# ── Main Menu ──────────────────────────────────────────────────────────────────

def interactive_menu():
    """Main interactive menu loop."""
    state = {
        "seed": 42,
        "random_seed": False,
        "timeout": DEFAULT_TIMEOUT,
        "auto_upscale": True,
        "upscale_scale": 4,
        "upscale_mode": "fast",  # fast=local Pillow, ai=HF Real-ESRGAN, auto=AI->fallback
    }

    print_header()

    while True:
        model_count = len(get_images(MODELS_DIR))
        garment_count = len(get_images(GARMENTS_DIR))
        result_count = len(get_images(RESULTS_DIR))
        video_count = len(get_images(RESULTS_VIDEO_DIR))


        action = questionary.select(
            "  Main Menu:",
            choices=[
                questionary.Choice(
                    title=f"🔥  Try On           (models: {model_count}, garments: {garment_count})",
                    value="tryon",
                ),
                questionary.Choice(
                    title=f"🎬  Photo to Video   (results: {result_count})",
                    value="video",
                ),
                questionary.Choice(
                    title=f"📁  Lihat Files      (results: {result_count}, video: {video_count})",
                    value="files",
                ),
                questionary.Choice(
                    title=f"⚙️   Settings         (seed: {'Random' if state['random_seed'] else state['seed']}, upscale: {'ON x'+str(state['upscale_scale']) if state['auto_upscale'] else 'OFF'})",
                    value="settings",
                ),
                questionary.Separator(),
                questionary.Choice(title="🚪  Exit", value="exit"),
            ],
            style=custom_style,
            use_shortcuts=True,
        ).ask()

        if action is None or action == "exit":
            print("\n  👋 Bye!\n")
            return

        elif action == "tryon":
            menu_try_on(state)

        elif action == "video":
            menu_photo_to_video()

        elif action == "files":
            menu_view_files()

        elif action == "settings":
            menu_settings(state)


# ── Direct CLI mode ────────────────────────────────────────────────────────────

def cli_mode():
    """Original argparse CLI for scripting / non-interactive use."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Kolors Virtual Try-On CLI",
        epilog="Example: python3 tryon.py -p model.jpg -g shirt.jpg -o result.png -v",
    )
    parser.add_argument("--person", "-p", help="Person image (file or URL)")
    parser.add_argument("--garment", "-g", help="Garment image (file or URL)")
    parser.add_argument("--output", "-o", default=str(RESULTS_DIR / "result.png"), help="Output path")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Seed (0-999999)")
    parser.add_argument("--random-seed", action="store_true", help="Random seed")
    parser.add_argument("--timeout", "-t", type=int, default=DEFAULT_TIMEOUT, help="Timeout (sec)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--upscale", "-u", type=int, choices=[0, 2, 4, 8], default=4, help="Upscale result (0=disable, 2/4/8=x scale, default 4)")
    parser.add_argument("--upscale-mode", "-m", choices=["fast", "ai", "auto"], default="fast", help="Upscale method: fast=Pillow(~2s), ai=HF Real-ESRGAN(~5-18m), auto=AI+fallback (default: fast)")
    parser.add_argument("--list", "-l", action="store_true", help="List local images")

    args = parser.parse_args()

    if args.list:
        for label, d in [("models/", MODELS_DIR), ("garments/", GARMENTS_DIR), ("results/", RESULTS_DIR), ("results_video/", RESULTS_VIDEO_DIR)]:
            print(f"\n📁 {label}")
            for f in sorted(f for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS):
                print(f"   {f.name}")
        return

    if not args.person or not args.garment:
        parser.error("--person and --garment are required")

    try:
        result = submit_tryon(
            person_path=args.person,
            garment_path=args.garment,
            seed=args.seed,
            randomize_seed=args.random_seed,
            timeout=args.timeout,
            output_path=args.output,
            verbose=args.verbose,
        )
        print_result(result)

        if args.upscale > 0:
            try:
                up_out = upscale_image(
                    image_path=result["path"],
                    scale=args.upscale,
                    mode=args.upscale_mode,
                    verbose=args.verbose,
                )
            except Exception as e:
                print(f"\n  ⚠️  Upscale failed: {e}\n", file=sys.stderr)
    except Exception as e:
        print(f"\n  ❌ Error: {e}\n", file=sys.stderr)
        sys.exit(1)


# ── Entry ──────────────────────────────────────────────────────────────────────

def main():
    if "--cli" in sys.argv or "-p" in sys.argv or "--person" in sys.argv:
        sys.argv = [a for a in sys.argv if a != "--cli"]
        cli_mode()
    elif not sys.stdin.isatty():
        print("  ⚠️  No TTY detected. Use: python3 tryon.py -p person.jpg -g garment.jpg -o result.png")
        sys.exit(1)
    else:
        try:
            interactive_menu()
        except KeyboardInterrupt:
            print("\n\n  👋 Bye!\n")


if __name__ == "__main__":
    main()
