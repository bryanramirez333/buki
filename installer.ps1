# Buki Installer
# Run via: Install Buki.bat

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$PYTHON_ID   = "Python.Python.3.11"
$PYTHON_EXE  = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
$PYTHONW_EXE = "$env:LOCALAPPDATA\Programs\Python\Python311\pythonw.exe"
$INSTALL_DIR = "$env:LOCALAPPDATA\Buki"
$DESKTOP     = [Environment]::GetFolderPath("Desktop")
$STARTMENU   = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Buki"
$SCRIPT_DIR  = Split-Path -Parent $MyInvocation.MyCommand.Path

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

# -- GUI ----------------------------------------------------------------------
$form = New-Object System.Windows.Forms.Form
$form.Text            = "Buki Installer"
$form.Size            = New-Object System.Drawing.Size(480, 320)
$form.StartPosition   = "CenterScreen"
$form.FormBorderStyle = "FixedDialog"
$form.MaximizeBox     = $false
$form.BackColor       = [System.Drawing.Color]::FromArgb(15, 15, 15)
$form.ForeColor       = [System.Drawing.Color]::FromArgb(200, 200, 200)

$title = New-Object System.Windows.Forms.Label
$title.Text      = "Buki - Voice to Text"
$title.Font      = New-Object System.Drawing.Font("Segoe UI Semibold", 16)
$title.ForeColor = [System.Drawing.Color]::FromArgb(220, 220, 220)
$title.Location  = New-Object System.Drawing.Point(24, 24)
$title.Size      = New-Object System.Drawing.Size(430, 36)
$form.Controls.Add($title)

$subtitle = New-Object System.Windows.Forms.Label
$subtitle.Text      = "Push-to-talk dictation powered by Whisper AI"
$subtitle.Font      = New-Object System.Drawing.Font("Segoe UI", 9)
$subtitle.ForeColor = [System.Drawing.Color]::FromArgb(90, 90, 90)
$subtitle.Location  = New-Object System.Drawing.Point(26, 60)
$subtitle.Size      = New-Object System.Drawing.Size(430, 20)
$form.Controls.Add($subtitle)

$sep = New-Object System.Windows.Forms.Panel
$sep.BackColor = [System.Drawing.Color]::FromArgb(35, 35, 35)
$sep.Location  = New-Object System.Drawing.Point(24, 88)
$sep.Size      = New-Object System.Drawing.Size(420, 1)
$form.Controls.Add($sep)

$statusLabel = New-Object System.Windows.Forms.Label
$statusLabel.Text      = "Ready to install."
$statusLabel.Font      = New-Object System.Drawing.Font("Segoe UI", 10)
$statusLabel.ForeColor = [System.Drawing.Color]::FromArgb(160, 160, 160)
$statusLabel.Location  = New-Object System.Drawing.Point(24, 104)
$statusLabel.Size      = New-Object System.Drawing.Size(430, 24)
$form.Controls.Add($statusLabel)

$bar = New-Object System.Windows.Forms.ProgressBar
$bar.Location = New-Object System.Drawing.Point(24, 136)
$bar.Size     = New-Object System.Drawing.Size(420, 10)
$bar.Style    = "Continuous"
$bar.Minimum  = 0
$bar.Maximum  = 100
$bar.Value    = 0
$form.Controls.Add($bar)

$logBox = New-Object System.Windows.Forms.RichTextBox
$logBox.Location    = New-Object System.Drawing.Point(24, 158)
$logBox.Size        = New-Object System.Drawing.Size(420, 100)
$logBox.BackColor   = [System.Drawing.Color]::FromArgb(10, 10, 10)
$logBox.ForeColor   = [System.Drawing.Color]::FromArgb(100, 100, 100)
$logBox.Font        = New-Object System.Drawing.Font("Consolas", 8.5)
$logBox.ReadOnly    = $true
$logBox.BorderStyle = "None"
$logBox.ScrollBars  = "Vertical"
$form.Controls.Add($logBox)

$btn = New-Object System.Windows.Forms.Button
$btn.Text      = "Install"
$btn.Font      = New-Object System.Drawing.Font("Segoe UI Semibold", 10)
$btn.Location  = New-Object System.Drawing.Point(338, 268)
$btn.Size      = New-Object System.Drawing.Size(106, 34)
$btn.FlatStyle = "Flat"
$btn.BackColor = [System.Drawing.Color]::FromArgb(30, 70, 30)
$btn.ForeColor = [System.Drawing.Color]::FromArgb(200, 220, 200)
$btn.FlatAppearance.BorderColor = [System.Drawing.Color]::FromArgb(50, 100, 50)
$form.Controls.Add($btn)

# -- Helpers ------------------------------------------------------------------
function Set-Status($text, $pct) {
    $statusLabel.Text = $text
    if ($pct -ge 0) { $bar.Value = [Math]::Min($pct, 100) }
    $form.Refresh()
}

function Add-Log($text) {
    $logBox.AppendText("$text`n")
    $logBox.ScrollToCaret()
    $form.Refresh()
}

function Show-Done {
    $statusLabel.Text      = "Installation complete."
    $statusLabel.ForeColor = [System.Drawing.Color]::FromArgb(100, 200, 100)
    $bar.Value             = 100
    $btn.Text              = "Launch Buki"
    $btn.BackColor         = [System.Drawing.Color]::FromArgb(20, 90, 20)
    $btn.Tag               = "done"
    $form.Refresh()
}

function Show-Error($msg) {
    $statusLabel.Text      = "Error: $msg"
    $statusLabel.ForeColor = [System.Drawing.Color]::FromArgb(200, 60, 60)
    $btn.Enabled           = $true
    $btn.Text              = "Retry"
    $form.Refresh()
}

# -- Install ------------------------------------------------------------------
$launcherPath = "$INSTALL_DIR\launch.bat"

function Start-Install {
    $btn.Enabled = $false

    # 1. Python
    Set-Status "Checking Python 3.11..." 5
    if (-not (Test-Path $PYTHON_EXE)) {
        Set-Status "Installing Python 3.11 (may take a minute)..." 10
        Add-Log "winget install Python.Python.3.11"
        $r = & winget install $PYTHON_ID --silent --accept-package-agreements --accept-source-agreements 2>&1
        Add-Log ($r | Out-String).Trim()
        if (-not (Test-Path $PYTHON_EXE)) {
            Show-Error "Python install failed."
            return
        }
    } else {
        Add-Log "Python 3.11 found."
    }

    # 2. pip packages
    $total    = $PACKAGES.Count
    $i        = 0
    $basePct  = 20
    $rangePct = 60
    foreach ($pkg in $PACKAGES) {
        $i++
        $pct = $basePct + [int](($i / $total) * $rangePct)
        Set-Status "Installing $pkg ($i/$total)..." $pct
        Add-Log "pip install $pkg"
        & $PYTHON_EXE -m pip install $pkg --quiet 2>&1 | Out-Null
    }

    # 3. Copy files
    Set-Status "Copying Buki files..." 85
    New-Item -ItemType Directory -Path $INSTALL_DIR -Force | Out-Null
    $src = Join-Path $SCRIPT_DIR "buki.py"
    if (-not (Test-Path $src)) {
        Show-Error "buki.py not found next to installer."
        return
    }
    Copy-Item $src -Destination $INSTALL_DIR -Force
    Add-Log "Copied buki.py to $INSTALL_DIR"

    # 4. Launcher
    $launcherContent = "@echo off`nstart `"`" /B `"$PYTHONW_EXE`" `"$INSTALL_DIR\buki.py`""
    [System.IO.File]::WriteAllText($launcherPath, $launcherContent, [System.Text.Encoding]::ASCII)
    Add-Log "Created launcher."

    # 5. Shortcuts
    Set-Status "Creating shortcuts..." 93
    $ws = New-Object -ComObject WScript.Shell

    $sc = $ws.CreateShortcut("$DESKTOP\Buki.lnk")
    $sc.TargetPath       = $launcherPath
    $sc.WorkingDirectory = $INSTALL_DIR
    $sc.WindowStyle      = 7
    $sc.Description      = "Buki - Voice to Text"
    $sc.Save()
    Add-Log "Desktop shortcut created."

    New-Item -ItemType Directory -Path $STARTMENU -Force | Out-Null
    $sm = $ws.CreateShortcut("$STARTMENU\Buki.lnk")
    $sm.TargetPath       = $launcherPath
    $sm.WorkingDirectory = $INSTALL_DIR
    $sm.WindowStyle      = 7
    $sm.Description      = "Buki - Voice to Text"
    $sm.Save()
    Add-Log "Start Menu shortcut created."

    Show-Done
}

# -- Button -------------------------------------------------------------------
$btn.Add_Click({
    if ($btn.Tag -eq "done") {
        Start-Process -FilePath $launcherPath
        $form.Close()
    } else {
        $thread = [System.Threading.Thread]::new([System.Threading.ThreadStart]{
            try   { Start-Install }
            catch { Show-Error $_.Exception.Message }
        })
        $thread.Start()
    }
})

$form.Add_Shown({ $form.Activate() })
[void]$form.ShowDialog()
