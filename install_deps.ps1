# install_deps.ps1 - runs silently during Inno Setup installation

$PYTHON_ID  = "Python.Python.3.11"
$PYTHON_EXE = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"

$PACKAGES = @(
    "faster-whisper",
    "sounddevice",
    "numpy",
    "pynput",
    "pyautogui",
    "pyperclip",
    "pystray",
    "Pillow",
    "customtkinter",
    "psutil",
    "nvidia-cublas-cu12",
    "nvidia-cuda-runtime-cu12"
)

# 1. Install Python 3.11 if missing
if (-not (Test-Path $PYTHON_EXE)) {
    & winget install $PYTHON_ID --silent --accept-package-agreements --accept-source-agreements | Out-Null
}

# 2. Install pip packages
foreach ($pkg in $PACKAGES) {
    & $PYTHON_EXE -m pip install $pkg --quiet 2>&1 | Out-Null
}
