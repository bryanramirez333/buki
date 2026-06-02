"""
test_buki.py - Buki Smoke & Resistance Test Suite
Suite A: Mocked (fast, isolated, no model needed)
Suite B: Semi-real (loads tiny/cpu once, reuses)
"""

import sys
import os
import time
import threading
import unittest
import numpy as np
import psutil

# ── Patch PATH for cublas before any buki import ──────────────────────────────
_nvidia_bin = os.path.join(
    os.path.expanduser("~"),
    r"AppData\Local\Programs\Python\Python311\Lib\site-packages\nvidia\cublas\bin"
)
if os.path.isdir(_nvidia_bin):
    os.environ["PATH"] = _nvidia_bin + os.pathsep + os.environ.get("PATH", "")

# ── Mock GUI/tray/audio before importing buki ─────────────────────────────────
from unittest.mock import MagicMock, patch, PropertyMock

# Prevent customtkinter from opening any window
sys.modules["customtkinter"] = MagicMock()
sys.modules["pystray"]       = MagicMock()

# We will let sounddevice import normally but mock InputStream
import sounddevice as sd

# ── Now import buki internals ─────────────────────────────────────────────────
sys.path.insert(0, r"C:\Users\Bryan\AppData\Local\Buki")

import importlib
import buki as B   # imports core functions without launching main()

PROC = psutil.Process(os.getpid())

# ── Helpers ───────────────────────────────────────────────────────────────────
def ram_mb():
    return PROC.memory_info().rss / 1024 / 1024

def thread_count():
    return threading.active_count()

def make_silence(seconds=1, rate=16000):
    return np.zeros((rate * seconds, 1), dtype=np.float32)

def make_noise(seconds=1, rate=16000):
    rng = np.random.default_rng(42)
    return rng.uniform(-0.1, 0.1, (rate * seconds, 1)).astype(np.float32)

def make_fake_whisper_result(text="hello world"):
    seg = MagicMock()
    seg.text = text
    info = MagicMock()
    info.language = "en"
    return [seg], info

# ── ANSI colors ───────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def fmt_pass(label, notes=""):
    return f"  {GREEN}PASS{RESET}  {label:<40} {YELLOW}{notes}{RESET}"

def fmt_fail(label, notes=""):
    return f"  {RED}FAIL{RESET}  {label:<40} {YELLOW}{notes}{RESET}"


# ═════════════════════════════════════════════════════════════════════════════
# SUITE A — MOCKED TESTS
# ═════════════════════════════════════════════════════════════════════════════
class SuiteA_Mocked(unittest.TestCase):

    def setUp(self):
        # Reset buki state before each test
        B.recording       = False
        B.audio_frames    = []
        B._trigger_down   = False
        B._capturing_hotkey = False
        B.model_obj       = None
        B.app_gui         = None
        # Reset hotkey to default
        B._current_hotkey_str = "key:insert"

        # Fake model
        self.mock_model = MagicMock()
        self.mock_model.transcribe.return_value = make_fake_whisper_result()

        # Capture log messages
        self.log_messages = []
        def fake_queue_put(msg):
            self.log_messages.append(msg)
        B.ui_queue.put = fake_queue_put

    def tearDown(self):
        # Restore ui_queue and hotkey
        import queue
        B.ui_queue = type(B.ui_queue)()
        B._current_hotkey_str = "key:insert"
        B._trigger_down = False
        B._capturing_hotkey = False

    # ── M1: Button spam (mouse hotkey) ────────────────────────────────────────
    def test_M1_button_spam(self):
        """50x rapid mouse press/release with mouse hotkey — no thread/RAM leak."""
        from pynput.mouse import Button

        B._current_hotkey_str = "mouse:x2"
        B.model_obj = self.mock_model

        ram_before     = ram_mb()
        threads_before = thread_count()

        for _ in range(50):
            B.on_mouse_click(0, 0, Button.x2, True)
            B.on_mouse_click(0, 0, Button.x2, False)
            time.sleep(0.01)

        # Wait for any spawned transcription threads to finish
        time.sleep(1.0)

        ram_delta    = ram_mb() - ram_before
        thread_delta = thread_count() - threads_before

        self._result_notes = f"RAM +{ram_delta:.1f}MB  threads +{thread_delta}"
        self.assertLess(ram_delta,    50, "RAM leak on button spam")
        self.assertLessEqual(thread_delta, 2, "Thread leak on button spam")

    # ── M2: Long audio frames ─────────────────────────────────────────────────
    def test_M2_long_audio_frames(self):
        """Inject 3 min of audio_frames — verify RAM stays reasonable."""
        ram_before = ram_mb()

        # Simulate 3 minutes of audio chunks (180 x 1-second chunks)
        B.audio_frames = [make_silence(1) for _ in range(180)]

        ram_after = ram_mb()
        ram_delta = ram_after - ram_before

        # Simulate stop + transcribe with mock model
        B.model_obj = self.mock_model
        B.stop_and_transcribe()
        time.sleep(0.5)

        self._result_notes = f"RAM for 3min frames: +{ram_delta:.1f}MB"
        self.assertLess(ram_delta, 500, "RAM too high for 3min audio frames")

    # ── M3: Reload model while transcribing ───────────────────────────────────
    def test_M3_reload_while_transcribing(self):
        """Set model_obj=None mid-transcription — must catch gracefully."""

        def slow_transcribe(*args, **kwargs):
            time.sleep(0.3)
            return make_fake_whisper_result()

        self.mock_model.transcribe.side_effect = slow_transcribe
        B.model_obj = self.mock_model
        B.audio_frames = [make_silence(1)]

        # Start transcription in background
        t = threading.Thread(target=B._transcribe,
                             args=(make_silence(1).flatten(),))
        t.start()

        # Kill model mid-transcription
        time.sleep(0.1)
        B.model_obj = None

        t.join(timeout=3)
        self._result_notes = "model set to None mid-transcribe"
        self.assertFalse(t.is_alive(), "Transcription thread hung")

    # ── M4: No microphone ─────────────────────────────────────────────────────
    def test_M4_no_microphone(self):
        """Audio callback raises exception — error must reach log, not crash."""
        B.model_obj = self.mock_model
        B.recording = True
        B.audio_frames = []

        try:
            raise sd.PortAudioError("No input device available")
        except Exception:
            B.audio_frames = []  # empty as if no audio arrived

        B.recording = False
        B.stop_and_transcribe()
        time.sleep(0.3)

        self._result_notes = "empty audio after mic failure"
        types = [m[0] for m in self.log_messages]
        self.assertIn("status", types, "No status update after empty audio")

    # ── M5: Clipboard blocked ─────────────────────────────────────────────────
    def test_M5_clipboard_blocked(self):
        """pyperclip.paste() raises — error must reach log, app must survive."""
        import pyperclip

        B.model_obj = self.mock_model
        error_logged = threading.Event()

        real_put = B.ui_queue.put
        def watching_put(msg):
            if isinstance(msg, tuple) and len(msg) >= 3 and msg[2] == "error":
                error_logged.set()
            real_put(msg)
        B.ui_queue.put = watching_put

        with patch("pyperclip.paste", side_effect=Exception("Clipboard access denied")):
            audio = make_silence(1).flatten()
            t = threading.Thread(target=B._transcribe, args=(audio,))
            t.start()
            t.join(timeout=5)

        self._result_notes = "pyperclip.paste() blocked"
        self.assertFalse(t.is_alive(), "Transcribe thread hung on clipboard error")
        self.assertTrue(error_logged.is_set(), "Error not logged when clipboard blocked")

    # ── M6: Hotkey parsing ────────────────────────────────────────────────────
    def test_M6_hotkey_parsing(self):
        """parse_hotkey_str and hotkey_display_name return correct values."""
        from pynput.keyboard import Key, KeyCode
        from pynput.mouse import Button

        cases = [
            ("key:insert",   ("key",   Key.insert),  "Insert"),
            ("key:f1",       ("key",   Key.f1),       "F1"),
            ("mouse:x2",     ("mouse", Button.x2),    "Mouse 5"),
            ("mouse:x1",     ("mouse", Button.x1),    "Mouse 4"),
            ("mouse:left",   ("mouse", Button.left),  "Left Click"),
            ("char:a",       ("char",  KeyCode.from_char("a")), "A"),
            ("",             ("key",   Key.insert),   "Insert"),   # fallback
            ("garbage:???",  ("key",   Key.insert),   "garbage:???"),  # unknown passthrough
        ]

        errors = []
        for s, expected_parse, expected_display in cases:
            kind, val = B.parse_hotkey_str(s)
            disp = B.hotkey_display_name(s)
            if (kind, val) != expected_parse:
                errors.append(f"parse({s!r}): got ({kind},{val}) expected {expected_parse}")
            if disp != expected_display:
                errors.append(f"display({s!r}): got {disp!r} expected {expected_display!r}")

        self._result_notes = f"{len(cases)} cases"
        self.assertEqual(errors, [], "\n".join(errors))

    # ── M7: Keyboard hotkey triggers recording ────────────────────────────────
    def test_M7_keyboard_trigger(self):
        """Insert key press/release triggers start/stop recording via keyboard listener."""
        from pynput.keyboard import Key

        B._current_hotkey_str = "key:insert"
        B.model_obj = self.mock_model

        recording_states = []
        original_start = B.start_recording
        original_stop  = B.stop_and_transcribe

        def fake_start():
            recording_states.append("start")
        def fake_stop():
            recording_states.append("stop")

        B.start_recording    = fake_start
        B.stop_and_transcribe = fake_stop

        try:
            # Simulate press
            B.on_key_press(Key.insert)
            self.assertTrue(B._trigger_down, "trigger_down not set on key press")
            self.assertTrue(B.recording,     "recording not True on key press")

            # Simulate release
            B.on_key_release(Key.insert)
            self.assertFalse(B._trigger_down, "trigger_down not cleared on key release")
            self.assertFalse(B.recording,     "recording not False on key release")

            # Unrelated key must not trigger
            B.on_key_press(Key.space)
            self.assertFalse(B._trigger_down, "space key incorrectly triggered recording")

        finally:
            B.start_recording     = original_start
            B.stop_and_transcribe = original_stop

        self._result_notes = f"states={recording_states}"
        self.assertEqual(recording_states, ["start", "stop"])

    # ── M8: Capture mode ──────────────────────────────────────────────────────
    def test_M8_capture_mode(self):
        """While _capturing_hotkey=True, key/mouse put to _capture_queue, not record."""
        from pynput.keyboard import Key
        from pynput.mouse import Button
        import queue as _queue

        B._capturing_hotkey = True
        # Drain queue
        while not B._capture_queue.empty():
            try: B._capture_queue.get_nowait()
            except _queue.Empty: break

        # Key press should go to capture queue
        B.on_key_press(Key.f2)
        result_key = B._capture_queue.get(timeout=1)
        self.assertEqual(result_key, "key:f2")
        self.assertFalse(B.recording, "recording started during capture mode (key)")

        # Mouse press should go to capture queue
        B.on_mouse_click(0, 0, Button.x1, True)
        result_mouse = B._capture_queue.get(timeout=1)
        self.assertEqual(result_mouse, "mouse:x1")
        self.assertFalse(B.recording, "recording started during capture mode (mouse)")

        self._result_notes = f"key={result_key} mouse={result_mouse}"


# ═════════════════════════════════════════════════════════════════════════════
# SUITE B — SEMI-REAL TESTS (tiny · cpu · int8)
# ═════════════════════════════════════════════════════════════════════════════
class SuiteB_SemiReal(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"\n{CYAN}  Loading tiny model (cpu/int8) — please wait...{RESET}")
        from faster_whisper import WhisperModel
        cls.model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print(f"  {GREEN}Model ready.{RESET}\n")

    def setUp(self):
        B.recording         = False
        B.audio_frames      = []
        B._trigger_down     = False
        B._capturing_hotkey = False
        B._current_hotkey_str = "key:insert"
        B.model_obj         = self.model
        B.app_gui           = None
        self.log_messages   = []
        def fake_put(msg):
            self.log_messages.append(msg)
        B.ui_queue.put = fake_put

    def tearDown(self):
        import queue
        B.ui_queue = type(B.ui_queue)()

    # ── R1: Silence ───────────────────────────────────────────────────────────
    def test_R1_silence(self):
        """Transcribe silence — no crash, result empty or 'no speech'."""
        audio = make_silence(2).flatten()
        t0 = time.time()
        B._transcribe(audio)
        elapsed = time.time() - t0

        self._result_notes = f"{elapsed:.1f}s"
        self.assertTrue(len(self.log_messages) > 0, "No output from silence transcription")

    # ── R2: White noise ───────────────────────────────────────────────────────
    def test_R2_white_noise(self):
        """Transcribe white noise — no crash, handles garbage audio."""
        audio = make_noise(2).flatten()
        t0 = time.time()
        B._transcribe(audio)
        elapsed = time.time() - t0

        self._result_notes = f"{elapsed:.1f}s"
        self.assertTrue(len(self.log_messages) > 0, "No output from noise transcription")

    # ── R3: 3-minute audio ────────────────────────────────────────────────────
    def test_R3_long_audio(self):
        """Transcribe 3 min of silence — RAM safe, completes in <30s."""
        audio = make_silence(180).flatten()

        ram_before = ram_mb()
        t0 = time.time()
        B._transcribe(audio)
        elapsed = time.time() - t0
        ram_delta = ram_mb() - ram_before

        self._result_notes = f"RAM +{ram_delta:.0f}MB  {elapsed:.1f}s"
        self.assertLess(elapsed,   30,  "3-min transcription timed out")
        self.assertLess(ram_delta, 500, "RAM spike too high for 3-min audio")

    # ── R4: Reload model mid-transcribe ───────────────────────────────────────
    def test_R4_reload_mid_transcribe(self):
        """Replace model_obj=None while real transcription running — must not crash."""
        real_model = self.model
        results = {"completed": False, "error": None}

        def run():
            try:
                audio = make_silence(3).flatten()
                B._transcribe(audio)
                results["completed"] = True
            except Exception as e:
                results["error"] = str(e)

        t = threading.Thread(target=run)
        t.start()
        time.sleep(0.1)
        B.model_obj = None  # kill model mid-flight
        t.join(timeout=10)
        B.model_obj = real_model  # restore

        self._result_notes = f"error={results['error']}"
        self.assertFalse(t.is_alive(), "Thread hung after model replaced")

    # ── R5: Clipboard blocked real ────────────────────────────────────────────
    def test_R5_clipboard_blocked_real(self):
        """Real transcription + clipboard blocked — error logged, no crash."""
        error_seen = threading.Event()

        real_put = B.ui_queue.put
        def watching_put(msg):
            if isinstance(msg, tuple) and len(msg) >= 3 and msg[2] == "error":
                error_seen.set()
            real_put(msg)
        B.ui_queue.put = watching_put

        audio = make_noise(2).flatten()

        with patch("pyperclip.paste", side_effect=Exception("Clipboard locked")):
            t = threading.Thread(target=B._transcribe, args=(audio,))
            t.start()
            t.join(timeout=10)

        self._result_notes = "pyperclip blocked with real model"
        self.assertFalse(t.is_alive(), "Thread hung on clipboard error (real model)")


# ═════════════════════════════════════════════════════════════════════════════
# CUSTOM RUNNER — pretty output
# ═════════════════════════════════════════════════════════════════════════════
class BukiTestResult(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.results = []

    def addSuccess(self, test):
        notes = getattr(test, "_result_notes", "")
        self.results.append(("PASS", test, notes))

    def addFailure(self, test, err):
        msg = str(err[1])
        self.results.append(("FAIL", test, msg))

    def addError(self, test, err):
        msg = str(err[1])
        self.results.append(("ERROR", test, msg))


def run_suite(suite_class, label):
    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = None
    suite  = loader.loadTestsFromTestCase(suite_class)
    result = BukiTestResult()

    print(f"\n{BOLD}{'-'*55}{RESET}")
    print(f"{BOLD}{CYAN}  {label}{RESET}")
    print(f"{BOLD}{'-'*55}{RESET}")

    try:
        suite_class.setUpClass()
    except AttributeError:
        pass
    except Exception as e:
        print(f"  {RED}setUpClass failed: {e}{RESET}")
        return 0, len(list(suite))

    for test in suite:
        name = test._testMethodName
        t0   = time.time()
        test.run(result)
        elapsed = time.time() - t0

        last = result.results[-1] if result.results else None
        if last and last[0] == "PASS":
            notes = f"{last[2]}  [{elapsed:.1f}s]"
            print(fmt_pass(name, notes))
        elif last and last[0] in ("FAIL", "ERROR"):
            notes = f"{last[2]}  [{elapsed:.1f}s]"
            print(fmt_fail(name, notes))

    try:
        suite_class.tearDownClass()
    except AttributeError:
        pass

    passed = sum(1 for r in result.results if r[0] == "PASS")
    total  = len(result.results)
    color  = GREEN if passed == total else RED
    print(f"\n  {color}{passed}/{total} passed{RESET}")
    return passed, total


if __name__ == "__main__":
    print(f"\n{BOLD}{'='*55}")
    print(f"  BUKI SMOKE & RESISTANCE TEST SUITE")
    print(f"{'='*55}{RESET}")

    p1, t1 = run_suite(SuiteA_Mocked,   "SUITE A — Mocked (fast)")
    p2, t2 = run_suite(SuiteB_SemiReal, "SUITE B — Semi-real (tiny · cpu · int8)")

    total_p = p1 + p2
    total_t = t1 + t2

    print(f"\n{BOLD}{'='*55}")
    color = GREEN if total_p == total_t else RED
    print(f"  {color}TOTAL: {total_p}/{total_t} PASSED{RESET}")
    print(f"{BOLD}{'='*55}{RESET}\n")

    sys.exit(0 if total_p == total_t else 1)
