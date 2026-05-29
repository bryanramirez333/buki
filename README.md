# Buki — Voice to Text

Push-to-talk dictation for Windows, powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper).  
Hold **Mouse 5** (front side button) → speak → release → text pastes wherever your cursor is.

---

## Download & Install

1. Go to [**Releases**](../../releases/latest)
2. Download `Buki-Setup.zip`
3. Extract and double-click **`Install Buki.bat`**
4. Wait ~2 min on first run (downloads the AI model, ~500 MB–1.6 GB depending on settings)
5. Buki appears in your system tray and bottom-right corner

No admin required. Installs to `%LOCALAPPDATA%\Buki\`.

---

## Requirements

| | Minimum | Recommended |
|---|---|---|
| OS | Windows 10 64-bit | Windows 11 |
| RAM | 8 GB | 16 GB |
| CPU | Intel i5 (8th gen+) | i5 11th gen+ |
| GPU | — (CPU mode works) | NVIDIA GPU 4GB+ VRAM |
| Disk | 2 GB free | 4 GB free |
| Internet | First run only (model download) | — |

---

## Usage

| Action | Result |
|---|---|
| Hold **Mouse 5** | Start recording (red dot) |
| Release **Mouse 5** | Transcribe and paste text |
| **⚙ Settings** | Change device (CPU/GPU) and model |
| **─** button | Minimize to tray |
| Tray → **Show** | Restore window |
| Tray → **Quit** | Exit (frees VRAM for gaming) |

---

## Models

| Model | Speed (GPU) | Speed (CPU i5) | Quality |
|---|---|---|---|
| tiny | ~0.3s | ~0.5s | Basic |
| small | ~0.5s | ~1.5s | Good |
| medium | ~1s | ~4–6s | Very good |
| large-v3-turbo | ~0.8s | ~8–12s | Best |

**Auto mode** picks the best model for your hardware automatically.

---

## Settings

Open with the **⚙** button in the top-right corner.

- **Device**: `Auto` / `CPU` / `GPU`  
  Use `CPU` to simulate the experience on a laptop without a GPU.
- **Model**: `Auto` / `tiny` / `small` / `medium` / `large-v3-turbo`

Changes apply after clicking **Apply & Reload Model**.

---

## How it works

1. You hold Mouse 5 → audio is captured from your default microphone
2. You release → audio is sent to [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (runs locally, no internet)
3. Transcription is copied to clipboard and pasted via `Ctrl+V` into the active window
4. Clipboard is restored immediately after

All processing is **100% local**. No audio leaves your machine.

---

## Uninstall

Delete `%LOCALAPPDATA%\Buki\` and the Desktop shortcut.  
To also remove Python packages: `pip uninstall faster-whisper sounddevice customtkinter pystray pynput`

---

## License

MIT
