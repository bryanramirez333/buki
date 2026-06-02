# Buki — Voice to Text

Push-to-talk dictation for Windows, powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper).  
Hold **Insert** (default, configurable) → speak → release → text pastes wherever your cursor is.

---

## Download & Install

1. Go to [**Releases**](../../releases/latest)
2. Download **`Buki-Setup.exe`**
3. Run it — no admin required, installs to `%LOCALAPPDATA%\Buki\`
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
| Hold **Insert** *(or your configured key)* | Start recording (red dot) |
| Release | Transcribe and paste text |
| **⚙ Settings** | Change hotkey, device (CPU/GPU), and model |
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

- **Hotkey**: any keyboard key or mouse button (default: `Insert`)  
  Click the button and press any key or mouse button to reassign.
- **Device**: `Auto` / `CPU` / `GPU`  
  Use `CPU` to simulate the experience on a laptop without a GPU.
- **Model**: `Auto` / `tiny` / `small` / `medium` / `large-v3-turbo`

Changes apply after clicking **Apply & Reload Model**.

---

## How it works

1. You hold the configured hotkey → audio is captured from your default microphone
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
