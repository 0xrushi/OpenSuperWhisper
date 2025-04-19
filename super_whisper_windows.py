import sys, os, socket, asyncio, json, numpy as np
import pyaudio, line_packet
from pynput.keyboard import Controller as KeyController
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, pyqtSignal, QThread
import pyqtgraph as pg

# ── audio constants ────────────────────────────────────────────────────────────
CHUNK    = 1024
FORMAT   = pyaudio.paInt16
CHANNELS = 1
RATE     = 16000          # server expects 16 kHz


# ── worker threads ─────────────────────────────────────────────────────────────
class AudioRecorder(QThread):
    audio_chunk = pyqtSignal(bytes)
    data_ready  = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._running = False
        self._pa      = pyaudio.PyAudio()

    def run(self):
        try:
            stream = self._pa.open(format=FORMAT, channels=CHANNELS,
                                   rate=RATE, input=True,
                                   frames_per_buffer=CHUNK)
        except Exception as e:
            print(f"Audio input error: {e}")
            return

        self._running = True
        while self._running:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
            except Exception:
                continue
            self.audio_chunk.emit(data)
            self.data_ready.emit(np.frombuffer(data, dtype=np.int16))

        stream.stop_stream()
        stream.close()

    def stop(self):
        self._running = False
        self.quit()          # tell Qt to end the thread loop
        self.wait(100)       # wait 100 ms max → closes instantly


class RemoteTranscriber(QThread):
    text_ready = pyqtSignal(str)

    def __init__(self, host, port):
        super().__init__()
        self.host, self.port = host, port
        self._running = True
        self.sock = None

    def run(self):
        try:
            self.sock = socket.create_connection((self.host, self.port))
            self.sock.setblocking(False)
        except Exception as e:
            print(f"Could not connect to server: {e}")
            return

        while self._running:
            QtCore.QThread.msleep(10)
            try:
                for line in line_packet.receive_lines(self.sock):
                    parts = line.strip().split(' ', 2)
                    if len(parts) == 3:
                        _, _, text = parts
                        self.text_ready.emit(text)
            except BlockingIOError:
                continue

    @QtCore.pyqtSlot(bytes)
    def send_chunk(self, chunk: bytes):
        try: self.sock.send(chunk)
        except (BlockingIOError, BrokenPipeError, OSError): pass

    def stop(self):
        self._running = False
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_WR)
                self.sock.close()
            except Exception: pass
        self.quit()
        self.wait(100)


# ── hot‑key listener ───────────────────────────────────────────────────────────
class HotkeyListener(QThread):
    """
    * Ctrl+Space  → start recording
    * Esc         → stop recording + exit app immediately
    """
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.ctrl   = False
        self.listener = None

    def run(self):
        from pynput import keyboard
        self.listener = keyboard.Listener(on_press=self.on_press,
                                          on_release=self.on_release,
                                          daemon=True)
        self.listener.start()
        self.exec_()

    def on_press(self, key):
        from pynput.keyboard import Key
        if key in (Key.ctrl_l, Key.ctrl_r):
            self.ctrl = True

        elif key == Key.space and self.ctrl:
            QtCore.QMetaObject.invokeMethod(self.window, '_start_recording',
                                            Qt.QueuedConnection)

        elif key == Key.esc:
            QtCore.QMetaObject.invokeMethod(self.window, '_exit_app',
                                            Qt.QueuedConnection)

    def on_release(self, key):
        from pynput.keyboard import Key
        if key in (Key.ctrl_l, Key.ctrl_r):
            self.ctrl = False

    def stop(self):
        if self.listener: self.listener.stop()
        self.quit()


# ── main window ────────────────────────────────────────────────────────────────
class SuperWhisperWindow(QtWidgets.QWidget):
    def __init__(self, host, port):
        super().__init__()
        self.kb  = KeyController()
        self.transcriber = None
        self.host, self.port = host, port
        self.replacements = self._load_text_replacements()
        self.deepseek_r1_enabled = True

        # ‑‑ UI chrome ‑‑
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool |
                            Qt.WindowStaysOnTopHint)
        self.setFixedSize(400, 180)
        self.setAttribute(Qt.WA_TranslucentBackground)
        effect = QtWidgets.QGraphicsDropShadowEffect(self)
        effect.setBlurRadius(20); effect.setOffset(0, 0)
        self.setGraphicsEffect(effect)

        root = QtWidgets.QVBoxLayout(self); root.setContentsMargins(0, 0, 0, 0)
        bg   = QtWidgets.QWidget(self)
        bg.setStyleSheet("background-color: rgba(255,255,255,230); border-radius:10px;")
        bg_layout = QtWidgets.QVBoxLayout(bg); bg_layout.setContentsMargins(10, 5, 10, 5)
        root.addWidget(bg)

        pg.setConfigOptions(antialias=True)
        self.plot = pg.PlotWidget(bg); self.plot.setBackground(None)
        self.plot.hideAxis('bottom'); self.plot.hideAxis('left')
        self.bars = pg.BarGraphItem(x=[], height=[], width=0.8)
        self.plot.addItem(self.bars)
        bg_layout.addWidget(self.plot)

        status = QtWidgets.QHBoxLayout()
        self.label = QtWidgets.QLabel('Press Ctrl+Space to Record • Esc to Close')
        self.label.setStyleSheet("color:gray; font:9pt 'Sans';")
        status.addWidget(self.label, alignment=Qt.AlignLeft)

        self.btn = QtWidgets.QPushButton('Record'); self.btn.setCheckable(True)
        self.btn.clicked.connect(self._toggle_recording)
        status.addWidget(self.btn, alignment=Qt.AlignCenter)
        status.addStretch(1)
        bg_layout.addLayout(status)

        self.recorder = AudioRecorder()
        self.recorder.data_ready.connect(self.update_waveform)

        self.hk = HotkeyListener(self); self.hk.start()

    # ── helpers ────────────────────────────────────────────────────────────
    def _load_text_replacements(self):
        path = os.path.join(os.path.dirname(__file__), "text_replacements.json")
        if not os.path.exists(path): return []
        with open(path, "r", encoding="utf-8") as f: return json.load(f)

    # ── recording control ───────────────────────────────────────────────────
    @QtCore.pyqtSlot()
    def _start_recording(self):
        if not self.isVisible():
            self.show()
            geom = QtWidgets.QApplication.primaryScreen().availableGeometry()
            self.move((geom.width()-self.width())//2, geom.height()-self.height())

        if not self.recorder.isRunning():
            self.label.setText('Recording… (Esc to Close)')
            self.label.setStyleSheet("color:red; font:9pt 'Sans';")
            self.btn.setChecked(True)

            self.transcriber = RemoteTranscriber(self.host, self.port)
            self.recorder.audio_chunk.connect(self.transcriber.send_chunk,
                                              QtCore.Qt.QueuedConnection)
            self.transcriber.text_ready.connect(self._display_and_type)
            self.transcriber.start(); self.recorder.start()

    @QtCore.pyqtSlot()
    def _stop_recording(self):
        if self.recorder.isRunning():
            self.recorder.stop()
            if self.transcriber:
                self.recorder.audio_chunk.disconnect(self.transcriber.send_chunk)
                self.transcriber.stop(); self.transcriber = None
        self.btn.setChecked(False)
        self.bars.setOpts(x=[], height=[], width=0.8)

    # called by the hot‑key on Esc
    @QtCore.pyqtSlot()
    def _exit_app(self):
        self._stop_recording()
        self.close()  # triggers closeEvent → hard exit

    def _toggle_recording(self, checked):
        self._start_recording() if checked else self._stop_recording()

    # ── text output ────────────────────────────────────────────────────────
    @QtCore.pyqtSlot(str)
    def _display_and_type(self, text):
        if self.deepseek_r1_enabled:
            for r in self.replacements: text = text.replace(r["from"], r["to"])
        print(f"[Typed] {text}"); self.kb.type(text)

    # ── waveform ───────────────────────────────────────────────────────────
    @QtCore.pyqtSlot(np.ndarray)
    def update_waveform(self, w):
        parts   = np.array_split(np.abs(w), 60)
        heights = np.array([np.mean(p) for p in parts])
        m       = np.max(heights) or 1
        new     = heights / m * 40
        if not hasattr(self, '_smooth'): self._smooth = new
        else: self._smooth = 0.6*self._smooth + 0.4*new
        self.bars.setOpts(x=np.arange(len(self._smooth)),
                          height=self._smooth, width=0.8)

    # ── shutdown ───────────────────────────────────────────────────────────
    def closeEvent(self, event):
        try:
            self._stop_recording()
            self.hk.stop()
        finally:
            QtWidgets.QApplication.quit()
            event.accept()


# ── launcher ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=43007)
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    win = SuperWhisperWindow(args.host, args.port)
    sys.exit(app.exec_())
