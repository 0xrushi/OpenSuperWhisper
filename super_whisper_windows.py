import sys, os, socket, asyncio, json, numpy as np
import pyaudio, line_packet
from pynput.keyboard import Controller as KeyController
from PyQt5 import QtWidgets, QtCore, QtGui
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
        self.lock = QtCore.QMutex()  # Add a lock for thread-safe socket access

    def run(self):
        try:
            self.lock.lock()
            self.sock = socket.create_connection((self.host, self.port))
            self.sock.setblocking(False)
            self.lock.unlock()
        except Exception as e:
            print(f"Could not connect to server: {e}")
            self.lock.unlock()
            return

        while self._running:
            QtCore.QThread.msleep(10)
            try:
                self.lock.lock()
                if self.sock is None:
                    self.lock.unlock()
                    continue
                    
                for line in line_packet.receive_lines(self.sock):
                    parts = line.strip().split(' ', 2)
                    if len(parts) == 3:
                        _, _, text = parts
                        self.text_ready.emit(text)
                self.lock.unlock()
            except (BlockingIOError, ConnectionError) as e:
                self.lock.unlock()
                continue
            except Exception as e:
                print(f"Unexpected error in transcriber: {e}")
                self.lock.unlock()
                self._cleanup_socket()
                break

    @QtCore.pyqtSlot(bytes)
    def send_chunk(self, chunk: bytes):
        self.lock.lock()
        try:
            if self.sock is not None:
                self.sock.send(chunk)
        except (BlockingIOError, BrokenPipeError, OSError, ConnectionError) as e:
            print(f"Error sending audio chunk: {e}")
            self._cleanup_socket()
        finally:
            self.lock.unlock()

    def _cleanup_socket(self):
        """Safely close the socket"""
        self.lock.lock()
        try:
            if self.sock:
                try:
                    self.sock.shutdown(socket.SHUT_WR)
                    self.sock.close()
                except Exception:
                    pass
                self.sock = None
        finally:
            self.lock.unlock()

    def stop(self):
        self._running = False
        self._cleanup_socket()
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
# ... [keep all your existing imports and constants] ...

class SuperWhisperWindow(QtWidgets.QWidget):
    def __init__(self, host, port):
        super().__init__()
        self.kb = KeyController()
        self.transcriber = None
        self.host, self.port = host, port
        self.replacements = self._load_text_replacements()
        self.deepseek_r1_enabled = True

        # ── UI Setup with Improved Visuals ──────────────────────────────────
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool |
                           Qt.WindowStaysOnTopHint)
        self.setFixedSize(500, 220)  # Slightly larger for better visibility
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Improved shadow effect
        effect = QtWidgets.QGraphicsDropShadowEffect(self)
        effect.setBlurRadius(25)
        effect.setColor(QtGui.QColor(0, 0, 0, 150))
        effect.setOffset(0, 5)
        self.setGraphicsEffect(effect)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(15, 15, 15, 15)
        
        # Main container with better styling
        bg = QtWidgets.QWidget(self)
        bg.setStyleSheet("""
            background-color: rgba(30, 30, 35, 240);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 30);
        """)
        bg_layout = QtWidgets.QVBoxLayout(bg)
        bg_layout.setContentsMargins(15, 15, 15, 15)
        bg_layout.setSpacing(15)
        root.addWidget(bg)

        # ── Enhanced Waveform Display ───────────────────────────────────────
        pg.setConfigOptions(antialias=True)
        self.plot = pg.PlotWidget(bg)
        self.plot.setBackground(None)
        self.plot.hideAxis('bottom')
        self.plot.hideAxis('left')
        self.plot.setMinimumHeight(100)
        self.plot.setMaximumHeight(120)
        
        # Gradient-colored bars
        self.bars = pg.BarGraphItem(
            x=[], height=[], width=0.8,
            pen=pg.mkPen(width=0),  # No border
            brush=pg.mkBrush((100, 150, 255, 200)))
        
        # Add a subtle background grid
        self.plot.showGrid(x=False, y=True, alpha=0.1)
        self.plot.addItem(self.bars)
        bg_layout.addWidget(self.plot)

        # ── Improved Status Area ───────────────────────────────────────────
        status = QtWidgets.QHBoxLayout()
        status.setContentsMargins(5, 0, 5, 0)
        
        # Status label with better styling
        self.label = QtWidgets.QLabel('Press Ctrl+Space to Record • Esc to Close')
        self.label.setStyleSheet("""
            color: rgba(200, 200, 210, 180);
            font: 10pt 'Segoe UI', 'Arial';
            padding: 3px;
        """)
        status.addWidget(self.label, alignment=Qt.AlignLeft)
        
        # Record button with modern styling
        self.btn = QtWidgets.QPushButton('Record')
        self.btn.setCheckable(True)
        self.btn.setFixedWidth(100)
        self.btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 70, 70, 0);
                color: rgba(240, 80, 80, 220);
                border: 1px solid rgba(240, 80, 80, 120);
                border-radius: 4px;
                padding: 5px;
                font: 10pt 'Segoe UI', 'Arial';
            }
            QPushButton:checked {
                background-color: rgba(240, 80, 80, 180);
                color: white;
            }
            QPushButton:hover {
                background-color: rgba(240, 80, 80, 60);
            }
        """)
        self.btn.clicked.connect(self._toggle_recording)
        status.addWidget(self.btn, alignment=Qt.AlignCenter)
        status.addStretch(1)
        bg_layout.addLayout(status)

        # Initialize audio components
        self.recorder = AudioRecorder()
        self.recorder.data_ready.connect(self.update_waveform)
        self.hk = HotkeyListener(self)
        self.hk.start()

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
                # Disconnect the signal before stopping
                try:
                    self.recorder.audio_chunk.disconnect(self.transcriber.send_chunk)
                except Exception:
                    pass
                self.transcriber.stop()
                self.transcriber = None
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
        parts = np.array_split(np.abs(w), 80)  # More bars for smoother appearance
        heights = np.array([np.mean(p) for p in parts])
        m = np.max(heights) or 1
        
        # Apply logarithmic scaling for better visual dynamics
        new = np.log1p(heights / m * 50) * 15
        
        if not hasattr(self, '_smooth'):
            self._smooth = new
        else:
            # Smoother animation with dynamic damping
            damping = 0.3 if np.max(new) > np.max(self._smooth) else 0.6
            self._smooth = damping * self._smooth + (1-damping) * new
        
        # Dynamic color based on amplitude
        peak = np.max(self._smooth)
        hue = 210 - min(50, peak * 2)  # Blue shifts toward purple with intensity
        self.bars.setOpts(
            x=np.arange(len(self._smooth)),
            height=self._smooth,
            width=0.8,
            brush=pg.mkBrush((hue, 200, 255, 200 - min(100, peak * 3))))

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
