"""
Buki — Voice to Text
Push-to-talk dictation powered by faster-whisper.
Hold the configured hotkey to record. Release to transcribe and paste.
Default hotkey: Insert
"""

import sys
import os
import json
import subprocess
import threading
import time
import queue

# ── CUDA DLL fix (nvidia-cublas-cu12 wheel) ───────────────────────────────────
def _add_nvidia_dlls():
    import site
    for sp in site.getsitepackages():
        candidate = os.path.join(sp, "nvidia", "cublas", "bin")
        if os.path.isdir(candidate):
            if candidate not in os.environ.get("PATH", ""):
                os.environ["PATH"] = candidate + os.pathsep + os.environ.get("PATH", "")
            break

_add_nvidia_dlls()

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import pyperclip
import pyautogui
import pystray
import customtkinter as ctk
from PIL import Image, ImageDraw
from pynput.mouse import Button, Listener as MouseListener
from pynput.keyboard import Key, KeyCode, Listener as KeyboardListener

# ── Paths ─────────────────────────────────────────────────────────────────────
APP_DIR       = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "Buki")
SETTINGS_FILE = os.path.join(APP_DIR, "settings.json")
os.makedirs(APP_DIR, exist_ok=True)

# ── Settings ──────────────────────────────────────────────────────────────────
DEFAULT_SETTINGS = {"device": "auto", "model": "auto", "hotkey": "key:insert"}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f:
                s = json.load(f)
                return {**DEFAULT_SETTINGS, **s}
        except Exception:
            pass
    return dict(DEFAULT_SETTINGS)

def save_settings(s):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(s, f, indent=2)

# ── Hotkey helpers ────────────────────────────────────────────────────────────
_MOUSE_DISPLAY = {
    "left": "Left Click", "right": "Right Click",
    "middle": "Middle Click", "x1": "Mouse 4", "x2": "Mouse 5",
}

def parse_hotkey_str(s):
    """Return (kind, value): kind in 'mouse'|'key'|'char'."""
    try:
        if s and s.startswith("mouse:"):
            return ("mouse", getattr(Button, s[6:]))
        elif s and s.startswith("key:"):
            return ("key", getattr(Key, s[4:]))
        elif s and s.startswith("char:"):
            return ("char", KeyCode.from_char(s[5:]))
    except AttributeError:
        pass
    return ("key", Key.insert)

def hotkey_display_name(s):
    if not s:
        return "Insert"
    if s.startswith("mouse:"):
        name = s[6:]
        return _MOUSE_DISPLAY.get(name, name.replace("_", " ").title())
    elif s.startswith("key:"):
        name = s[4:]
        return name.replace("_", " ").title()
    elif s.startswith("char:"):
        return s[5:].upper()
    return s

# ── Hardware auto-detection ───────────────────────────────────────────────────
def detect_hardware():
    """Returns (device, compute_type, model_size)."""
    try:
        import ctranslate2
        n = ctranslate2.get_cuda_device_count()
        if n > 0:
            vram = _get_vram_mb()
            if vram >= 6000:
                return "cuda", "float16", "large-v3-turbo"
            elif vram >= 3000:
                return "cuda", "float16", "small"
            else:
                return "cuda", "int8", "small"
    except Exception:
        pass
    return "cpu", "int8", "small"

def _get_vram_mb():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        return int(r.stdout.strip().split("\n")[0])
    except Exception:
        return 0

def resolve_config(settings):
    """Turn settings dict into (device, compute_type, model_size)."""
    device_pref = settings.get("device", "auto")
    model_pref  = settings.get("model",  "auto")

    auto_device, auto_compute, auto_model = detect_hardware()

    if device_pref == "auto":
        device  = auto_device
        compute = auto_compute
    elif device_pref == "gpu":
        device  = "cuda"
        compute = "float16"
    else:  # cpu
        device  = "cpu"
        compute = "int8"

    model = auto_model if model_pref == "auto" else model_pref
    return device, compute, model

# ── Tray icons ────────────────────────────────────────────────────────────────
def _make_tray_icon(rec=False):
    SIZE = 64
    img  = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    bg   = (50, 14, 14) if rec else (28, 28, 28)
    draw.ellipse([0, 0, SIZE - 1, SIZE - 1], fill=bg)
    mc = (220, 40, 40) if rec else (175, 175, 175)
    sc = (190, 55, 55) if rec else (130, 130, 130)
    draw.rounded_rectangle([22, 7, 42, 36], radius=10, fill=mc)
    draw.arc([12, 23, 52, 46], start=0, end=180, fill=sc, width=3)
    cx = SIZE // 2
    draw.line([cx, 46, cx, 56], fill=sc, width=3)
    draw.line([cx - 8, 56, cx + 8, 56], fill=sc, width=3)
    return img

TRAY_IDLE = _make_tray_icon(False)
TRAY_REC  = _make_tray_icon(True)

# ── Global state ──────────────────────────────────────────────────────────────
recording          = False
audio_frames       = []
model_obj          = None
tray_icon          = None
app_gui            = None
status_lock        = threading.Lock()
ui_queue           = queue.Queue()
SAMPLE_RATE        = 16000

_current_hotkey_str = "key:insert"   # updated from settings at startup / apply
_trigger_down       = False           # tracks press state regardless of kind
_capturing_hotkey   = False           # True while waiting for capture input
_capture_queue      = queue.Queue()   # receives "kind:name" string from listeners

# ── Model loader ──────────────────────────────────────────────────────────────
def load_model(device, compute, model_size, on_done=None, on_error=None):
    global model_obj
    try:
        model_obj = WhisperModel(model_size, device=device, compute_type=compute)
        if on_done:
            on_done(device, compute, model_size)
    except Exception as e:
        if on_error:
            on_error(str(e))

# ── Audio ─────────────────────────────────────────────────────────────────────
def audio_callback(indata, frames, time_info, status):
    if recording:
        audio_frames.append(indata.copy())

def _hold_hint():
    return f"Hold {hotkey_display_name(_current_hotkey_str)} to record"

def start_recording():
    global audio_frames
    audio_frames = []
    if tray_icon:
        tray_icon.icon  = TRAY_REC
        tray_icon.title = "Buki — Recording..."
    ui_queue.put(("recording", True))

def stop_and_transcribe():
    if tray_icon:
        tray_icon.icon  = TRAY_IDLE
        tray_icon.title = "Buki — idle"
    ui_queue.put(("recording", False))
    if not audio_frames:
        ui_queue.put(("status", _hold_hint(), "#555555"))
        return
    data = np.concatenate(audio_frames, axis=0).flatten().astype(np.float32)
    threading.Thread(target=_transcribe, args=(data,), daemon=True).start()

def _transcribe(audio_data):
    if model_obj is None:
        ui_queue.put(("log", "Model not loaded yet.", "error"))
        return
    try:
        segments, info = model_obj.transcribe(
            audio_data, language=None, beam_size=5, vad_filter=True
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if text:
            ui_queue.put(("log", f"[{info.language.upper()}] {text}", "ok"))
            ui_queue.put(("status", _hold_hint(), "#555555"))
            prev = pyperclip.paste()
            pyperclip.copy(text)
            pyautogui.hotkey("ctrl", "v")
            time.sleep(0.1)
            pyperclip.copy(prev)
        else:
            ui_queue.put(("log", "— no speech detected", "dim"))
            ui_queue.put(("status", _hold_hint(), "#555555"))
    except Exception as e:
        ui_queue.put(("log", f"Error: {e}", "error"))
        ui_queue.put(("status", _hold_hint(), "#555555"))

# ── Mouse listener ────────────────────────────────────────────────────────────
def on_mouse_click(x, y, button, pressed):
    global recording, _trigger_down, _capturing_hotkey

    # Capture mode: record and exit
    if _capturing_hotkey and pressed:
        _capture_queue.put(f"mouse:{button.name}")
        return

    kind, value = parse_hotkey_str(_current_hotkey_str)
    if kind != "mouse" or button != value:
        return

    if pressed and not _trigger_down:
        _trigger_down = True
        with status_lock:
            recording = True
        start_recording()
    elif not pressed and _trigger_down:
        _trigger_down = False
        with status_lock:
            recording = False
        stop_and_transcribe()

# ── Keyboard listener ─────────────────────────────────────────────────────────
def on_key_press(key):
    global recording, _trigger_down, _capturing_hotkey

    # Capture mode: record and exit
    if _capturing_hotkey:
        if isinstance(key, Key):
            _capture_queue.put(f"key:{key.name}")
        elif isinstance(key, KeyCode) and key.char:
            _capture_queue.put(f"char:{key.char}")
        return

    kind, value = parse_hotkey_str(_current_hotkey_str)
    if kind not in ("key", "char"):
        return
    if key != value:
        return
    if not _trigger_down:
        _trigger_down = True
        with status_lock:
            recording = True
        start_recording()

def on_key_release(key):
    global recording, _trigger_down

    kind, value = parse_hotkey_str(_current_hotkey_str)
    if kind not in ("key", "char"):
        return
    if key != value:
        return
    if _trigger_down:
        _trigger_down = False
        with status_lock:
            recording = False
        stop_and_transcribe()

# ── Tray ──────────────────────────────────────────────────────────────────────
def _show_window(icon=None, item=None):
    if app_gui:
        app_gui.after(0, app_gui.deiconify)

def _quit_tray(icon, item):
    icon.stop()
    os._exit(0)

def run_tray():
    global tray_icon
    menu = pystray.Menu(
        pystray.MenuItem("Buki — Voice to Text", None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Show", _show_window, default=True),
        pystray.MenuItem("Quit", _quit_tray),
    )
    tray_icon = pystray.Icon("Buki", TRAY_IDLE, "Buki", menu)
    tray_icon.run()

# ── GUI ───────────────────────────────────────────────────────────────────────
MODEL_OPTIONS  = ["auto", "tiny", "small", "medium", "large-v3-turbo"]
DEVICE_OPTIONS = ["auto", "cpu", "gpu"]

MODEL_DESCRIPTIONS = {
    "auto":            "Auto — best for your hardware",
    "tiny":            "Tiny — fastest, basic quality",
    "small":           "Small — balanced (recommended for CPU)",
    "medium":          "Medium — high quality, slow on CPU",
    "large-v3-turbo":  "Large v3 Turbo — best quality (GPU recommended)",
}

class BukiApp(ctk.CTk):
    W, H = 320, 420

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.title("Buki")
        self.geometry(self._bottom_right())
        self.resizable(False, False)
        self.attributes("-topmost", True)
        self.attributes("-toolwindow", True)
        self.overrideredirect(True)
        self.configure(fg_color="#0f0f0f")

        self._drag_x = self._drag_y = 0
        self._settings = load_settings()
        self._view = "main"   # "main" or "settings"

        self._build_main()
        self._build_settings()
        self._show_main()

        self.after(80, self._process_ui_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Layout ────────────────────────────────────────────────────────────────
    def _bottom_right(self):
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        return f"{self.W}x{self.H}+{sw - self.W - 24}+{sh - self.H - 56}"

    def _build_main(self):
        self._main_frame = ctk.CTkFrame(self, fg_color="transparent")

        # ── Header
        hdr = ctk.CTkFrame(self._main_frame, fg_color="#181818", corner_radius=0, height=48)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        hdr.bind("<ButtonPress-1>", self._drag_start)
        hdr.bind("<B1-Motion>",     self._drag_move)

        self._dot = ctk.CTkLabel(hdr, text="●", font=("Segoe UI", 16),
                                 text_color="#3a3a3a", width=26)
        self._dot.pack(side="left", padx=(12, 4))

        ctk.CTkLabel(hdr, text="Buki", font=("Segoe UI Semibold", 15),
                     text_color="#e8e8e8").pack(side="left")

        # header buttons right-to-left
        for symbol, cmd, hover in [
            ("✕", self._on_close,      "#3a1010"),
            ("─", self._minimize,      "#222222"),
            ("⚙", self._show_settings, "#1e2a1e"),
        ]:
            ctk.CTkButton(hdr, text=symbol, width=32, height=28,
                          fg_color="transparent", hover_color=hover,
                          text_color="#777777", font=("Segoe UI", 13),
                          command=cmd).pack(side="right", padx=(0, 4))

        # ── Status
        self._status = ctk.CTkLabel(self._main_frame,
                                    text="Loading model...",
                                    font=("Segoe UI", 11), text_color="#555555")
        self._status.pack(pady=(12, 4))

        # ── Rec bar
        self._bar = ctk.CTkProgressBar(self._main_frame, width=280, height=5,
                                       progress_color="#2a2a2a", fg_color="#1a1a1a")
        self._bar.set(0)
        self._bar.pack(pady=(0, 10))

        # ── Model info label
        self._model_info = ctk.CTkLabel(self._main_frame, text="",
                                        font=("Segoe UI", 9), text_color="#333333")
        self._model_info.pack()

        # ── Divider
        ctk.CTkFrame(self._main_frame, height=1, fg_color="#222222").pack(
            fill="x", padx=16, pady=(8, 0))

        ctk.CTkLabel(self._main_frame, text="TRANSCRIPTIONS",
                     font=("Segoe UI", 9), text_color="#333333").pack(
            anchor="w", padx=20, pady=(8, 4))

        # ── Log
        self._log = ctk.CTkTextbox(self._main_frame, width=280, height=210,
                                   font=("Consolas", 11), fg_color="#0a0a0a",
                                   text_color="#c0c0c0", border_color="#222222",
                                   border_width=1, corner_radius=6,
                                   wrap="word", state="disabled")
        self._log.pack(padx=16, pady=(0, 10))

        self._hold_label = ctk.CTkLabel(
            self._main_frame,
            text=_hold_hint(),
            font=("Segoe UI", 9), text_color="#2a2a2a"
        )
        self._hold_label.pack(pady=(0, 8))

    def _build_settings(self):
        self._settings_frame = ctk.CTkFrame(self, fg_color="transparent")

        # ── Header
        hdr = ctk.CTkFrame(self._settings_frame, fg_color="#181818",
                           corner_radius=0, height=48)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        ctk.CTkButton(hdr, text="←", width=36, height=28,
                      fg_color="transparent", hover_color="#222222",
                      text_color="#777777", font=("Segoe UI", 15),
                      command=self._show_main).pack(side="left", padx=8)

        ctk.CTkLabel(hdr, text="Settings", font=("Segoe UI Semibold", 14),
                     text_color="#e8e8e8").pack(side="left")

        ctk.CTkButton(hdr, text="✕", width=32, height=28,
                      fg_color="transparent", hover_color="#3a1010",
                      text_color="#777777", font=("Segoe UI", 13),
                      command=self._on_close).pack(side="right", padx=8)

        body = ctk.CTkScrollableFrame(self._settings_frame, fg_color="transparent",
                                      scrollbar_button_color="#222222",
                                      scrollbar_button_hover_color="#333333")
        body.pack(fill="both", expand=True, padx=0, pady=8)
        # inner padding via a child frame so content has left/right margin
        _inner = ctk.CTkFrame(body, fg_color="transparent")
        _inner.pack(fill="both", expand=True, padx=16)
        body = _inner

        # ── Hotkey
        ctk.CTkLabel(body, text="HOTKEY", font=("Segoe UI", 9),
                     text_color="#444444").pack(anchor="w", pady=(0, 6))

        current_hk = self._settings.get("hotkey", "key:insert")
        self._hotkey_btn = ctk.CTkButton(
            body,
            text=f"[ {hotkey_display_name(current_hk)} ]",
            width=280, height=36,
            fg_color="#1a1a1a", hover_color="#252525",
            font=("Segoe UI Semibold", 13),
            command=self._capture_hotkey_start,
        )
        self._hotkey_btn.pack(pady=(0, 4))

        self._hotkey_hint = ctk.CTkLabel(
            body, text="Click to reassign · supports keys and mouse buttons",
            font=("Segoe UI", 9), text_color="#3a3a3a",
            wraplength=270, justify="left",
        )
        self._hotkey_hint.pack(anchor="w", pady=(0, 14))

        # ── Device
        ctk.CTkLabel(body, text="DEVICE", font=("Segoe UI", 9),
                     text_color="#444444").pack(anchor="w", pady=(0, 6))

        self._device_var = ctk.StringVar(value=self._settings.get("device", "auto"))
        self._device_seg = ctk.CTkSegmentedButton(
            body, values=["auto", "cpu", "gpu"],
            variable=self._device_var,
            font=("Segoe UI", 12), width=280,
            fg_color="#1a1a1a", selected_color="#1e3a1e",
            selected_hover_color="#254a25", unselected_color="#1a1a1a",
        )
        self._device_seg.pack(pady=(0, 6))

        self._device_hint = ctk.CTkLabel(body, text=self._device_hint_text(),
                                         font=("Segoe UI", 9), text_color="#3a3a3a",
                                         wraplength=270, justify="left")
        self._device_hint.pack(anchor="w", pady=(0, 14))
        self._device_var.trace_add("write", lambda *_: self._device_hint.configure(
            text=self._device_hint_text()))

        # ── Model
        ctk.CTkLabel(body, text="MODEL", font=("Segoe UI", 9),
                     text_color="#444444").pack(anchor="w", pady=(0, 6))

        self._model_var = ctk.StringVar(value=self._settings.get("model", "auto"))
        self._model_menu = ctk.CTkOptionMenu(
            body, variable=self._model_var, values=MODEL_OPTIONS,
            width=280, font=("Segoe UI", 12),
            fg_color="#1a1a1a", button_color="#252525",
            button_hover_color="#303030", dropdown_fg_color="#181818",
            command=lambda _: self._model_hint.configure(
                text=MODEL_DESCRIPTIONS.get(self._model_var.get(), ""))
        )
        self._model_menu.pack(pady=(0, 6))

        self._model_hint = ctk.CTkLabel(
            body,
            text=MODEL_DESCRIPTIONS.get(self._settings.get("model", "auto"), ""),
            font=("Segoe UI", 9), text_color="#3a3a3a",
            wraplength=270, justify="left"
        )
        self._model_hint.pack(anchor="w", pady=(0, 14))

        # ── Hardware info
        self._hw_label = ctk.CTkLabel(body, text="", font=("Consolas", 9),
                                      text_color="#2e2e2e", wraplength=270, justify="left")
        self._hw_label.pack(anchor="w", pady=(0, 10))
        threading.Thread(target=self._fill_hw_info, daemon=True).start()

        # ── Apply
        ctk.CTkButton(body, text="Apply & Reload Model", width=280, height=36,
                      fg_color="#1e3a1e", hover_color="#254a25",
                      font=("Segoe UI Semibold", 13),
                      command=self._apply_settings).pack()

        self._apply_status = ctk.CTkLabel(body, text="", font=("Segoe UI", 9),
                                          text_color="#555555")
        self._apply_status.pack(pady=(8, 0))

    # ── Hotkey capture ────────────────────────────────────────────────────────
    def _capture_hotkey_start(self):
        global _capturing_hotkey
        # Drain any stale entries
        while not _capture_queue.empty():
            try:
                _capture_queue.get_nowait()
            except queue.Empty:
                break
        _capturing_hotkey = True
        self._hotkey_btn.configure(
            text="Press any key or mouse button...",
            fg_color="#1e2a3e",
        )
        self._hotkey_hint.configure(text="Listening for input...")
        self.after(100, self._capture_hotkey_poll)

    def _capture_hotkey_poll(self):
        global _capturing_hotkey, _current_hotkey_str, _trigger_down
        try:
            result = _capture_queue.get_nowait()
            _capturing_hotkey = False

            # Apply immediately — no Apply button needed for hotkey
            _current_hotkey_str = result
            _trigger_down = False

            self._settings["hotkey"] = result

            # Persist to disk (merge with current device/model)
            saved = {
                "device": self._settings.get("device", "auto"),
                "model":  self._settings.get("model",  "auto"),
                "hotkey": result,
            }
            save_settings(saved)

            name = hotkey_display_name(result)
            self._hotkey_btn.configure(
                text=f"[ {name} ]",
                fg_color="#1a1a1a",
            )
            self._hotkey_hint.configure(text=f"Saved. Click to reassign.")
            self._hold_label.configure(text=_hold_hint())
        except queue.Empty:
            if _capturing_hotkey:
                self.after(100, self._capture_hotkey_poll)

    # ── View switching ────────────────────────────────────────────────────────
    def _show_main(self):
        self._settings_frame.pack_forget()
        self._main_frame.pack(fill="both", expand=True)
        self._view = "main"

    def _show_settings(self):
        self._main_frame.pack_forget()
        self._settings_frame.pack(fill="both", expand=True)
        self._view = "settings"

    # ── Drag ──────────────────────────────────────────────────────────────────
    def _drag_start(self, e):
        self._drag_x, self._drag_y = e.x, e.y

    def _drag_move(self, e):
        self.geometry(f"+{self.winfo_x() + e.x - self._drag_x}"
                      f"+{self.winfo_y() + e.y - self._drag_y}")

    # ── Settings helpers ──────────────────────────────────────────────────────
    def _device_hint_text(self):
        v = self._device_var.get()
        if v == "auto":
            return "Detects GPU automatically. Falls back to CPU if unavailable."
        if v == "cpu":
            return "Forces CPU · int8. Slower but works on any machine."
        return "Forces GPU · float16. Will error if no CUDA GPU is present."

    def _fill_hw_info(self):
        lines = []
        try:
            import ctranslate2
            n = ctranslate2.get_cuda_device_count()
            lines.append(f"CUDA GPUs detected: {n}")
            if n > 0:
                vram = _get_vram_mb()
                lines.append(f"VRAM: {vram} MB")
        except Exception:
            lines.append("CUDA: not available")
        import psutil
        ram = psutil.virtual_memory().total // (1024 ** 3)
        lines.append(f"RAM: {ram} GB")
        self._hw_label.configure(text="\n".join(lines))

    def _apply_settings(self):
        global _trigger_down
        new_settings = {
            "device": self._device_var.get(),
            "model":  self._model_var.get(),
            "hotkey": self._settings.get("hotkey", "key:insert"),
        }
        save_settings(new_settings)
        self._settings = new_settings
        _trigger_down = False

        self._apply_status.configure(text="Reloading model...", text_color="#668866")
        self._show_main()
        ui_queue.put(("log", f"Reloading: device={new_settings['device']} model={new_settings['model']}", "dim"))
        ui_queue.put(("log", f"Hotkey set to: {hotkey_display_name(new_settings['hotkey'])}", "dim"))
        ui_queue.put(("status", "Reloading model...", "#556655"))
        threading.Thread(target=self._reload_model, daemon=True).start()

    def _reload_model(self):
        device, compute, model_size = resolve_config(self._settings)
        load_model(
            device, compute, model_size,
            on_done=lambda d, c, m: ui_queue.put(("model_ready", d, c, m)),
            on_error=lambda e:      ui_queue.put(("log", f"Load error: {e}", "error")),
        )

    # ── UI queue processor ────────────────────────────────────────────────────
    def _start_loading_spinner(self):
        self._spinner_active = True
        self._spinner_frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
        self._spinner_idx = 0
        self._tick_spinner()

    def _tick_spinner(self):
        if not getattr(self, "_spinner_active", False):
            return
        f = self._spinner_frames[self._spinner_idx % len(self._spinner_frames)]
        self._spinner_idx += 1
        self._bar.set((self._spinner_idx % 20) / 20)
        self._bar.configure(progress_color="#224422")
        self.after(120, self._tick_spinner)

    def _stop_spinner(self):
        self._spinner_active = False
        self._bar.set(0)
        self._bar.configure(progress_color="#2a2a2a")

    def _process_ui_queue(self):
        try:
            while True:
                msg = ui_queue.get_nowait()
                if msg[0] == "recording":
                    self._set_recording(msg[1])
                elif msg[0] == "status":
                    self._status.configure(text=msg[1], text_color=msg[2])
                elif msg[0] == "log":
                    self._append_log(msg[1], msg[2])
                elif msg[0] == "model_ready":
                    self._stop_spinner()
                    d, c, m = msg[1], msg[2], msg[3]
                    label = f"{m}  ·  {d}  ·  {c}"
                    self._model_info.configure(text=label, text_color="#334433")
                    self._status.configure(text=_hold_hint(), text_color="#555555")
                    self._append_log(f"Ready: {label}", "ok")
        except queue.Empty:
            pass
        self.after(80, self._process_ui_queue)

    def _set_recording(self, is_rec):
        if is_rec:
            self._dot.configure(text_color="#cc2222")
            self._status.configure(text="Recording...", text_color="#bb3333")
            self._bar.configure(progress_color="#771111")
            self._animate_bar(True, 0.0)
        else:
            self._dot.configure(text_color="#3a3a3a")
            self._status.configure(text="Processing...", text_color="#666666")
            self._bar.set(0)
            self._bar.configure(progress_color="#2a2a2a")

    def _animate_bar(self, up, val):
        if not recording:
            self._bar.set(0)
            return
        val += 0.06 if up else -0.06
        if val >= 1.0:  up = False
        elif val <= 0.0: up, val = True, 0.0
        self._bar.set(abs(val))
        self.after(55, lambda: self._animate_bar(up, val))

    def _append_log(self, text, tag):
        colors = {"ok": "#4a8a4a", "error": "#aa3333", "dim": "#3a3a3a"}
        self._log.configure(state="normal")
        self._log.insert("end", text + "\n")
        lines = int(self._log.index("end-1c").split(".")[0])
        if lines > 14:
            self._log.delete("1.0", f"{lines - 14}.0")
        self._log.configure(state="disabled")
        self._log.see("end")

    def set_model_ready(self, device, compute, model_size):
        ui_queue.put(("model_ready", device, compute, model_size))

    # ── Window controls ───────────────────────────────────────────────────────
    def _minimize(self):
        self.withdraw()

    def _on_close(self):
        os._exit(0)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global app_gui, _current_hotkey_str

    settings = load_settings()
    _current_hotkey_str = settings.get("hotkey", "key:insert")

    device, compute, model_size = resolve_config(settings)

    app_gui = BukiApp()
    ui_queue.put(("log", f"Loading {model_size} · {device} · {compute}...", "dim"))
    ui_queue.put(("log", "(first run downloads the model, please wait)", "dim"))
    ui_queue.put(("log", f"Hotkey: {hotkey_display_name(_current_hotkey_str)}", "dim"))
    ui_queue.put(("status", f"Loading {model_size}...", "#446644"))
    app_gui.update()
    app_gui.after(200, app_gui._start_loading_spinner)

    def _on_model_ready(d, c, m):
        ui_queue.put(("model_ready", d, c, m))

    def _on_model_error(e):
        ui_queue.put(("log", f"Load error: {e}", "error"))
        ui_queue.put(("status", "Model failed — check settings", "#883333"))

    threading.Thread(
        target=load_model,
        args=(device, compute, model_size, _on_model_ready, _on_model_error),
        daemon=True
    ).start()

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            callback=audio_callback, dtype="float32")
    stream.start()

    mouse_listener = MouseListener(on_click=on_mouse_click)
    mouse_listener.start()

    keyboard_listener = KeyboardListener(on_press=on_key_press, on_release=on_key_release)
    keyboard_listener.start()

    threading.Thread(target=run_tray, daemon=True).start()

    app_gui.mainloop()
    stream.stop()
    mouse_listener.stop()
    keyboard_listener.stop()


if __name__ == "__main__":
    main()
