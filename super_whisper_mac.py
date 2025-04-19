#!/usr/bin/env python3
"""
Super‑Whisper overlay
 • Ctrl+Space ‑ start/stop recording   • Esc ‑ quit
 • Never steals focus (WindowDoesNotAcceptFocus + WA_ShowWithoutActivating)
 • Stays visible across spaces / full‑screen on macOS
"""
import sys, os, socket, json, audioop, numpy as np
from   ctypes import c_void_p
import pyaudio, line_packet
from   pynput.keyboard   import Controller as KeyController
from   PyQt5             import QtWidgets, QtCore
from   PyQt5.QtCore      import Qt, pyqtSignal, QThread
import pyqtgraph as pg
import subprocess
import shutil

if sys.platform == "darwin":
    import objc
    from Cocoa import (
        NSWindowCollectionBehaviorCanJoinAllSpaces,
        NSWindowCollectionBehaviorStationary,
        NSWindowCollectionBehaviorFullScreenAuxiliary,
        NSWindowCollectionBehaviorIgnoresCycle,
        NSFloatingWindowLevel,
    )

TARGET_RATE = 16_000          # Whisper server expects 16 kHz
FORMAT      = pyaudio.paInt16
CHANNELS    = 1
CHUNK       = 1024            # frames @ TARGET_RATE

class AudioRecorder(QThread):
    audio_chunk = pyqtSignal(bytes)
    data_ready  = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._pa       = pyaudio.PyAudio()
        self._running  = False
        self._rate     = TARGET_RATE     # actual device rate
        self._need_rs  = False           # resample required?

    # try 16 kHz; if device refuses, open at native rate & flag resample
    def _open_stream(self):
        try:
            return self._pa.open(format=FORMAT, channels=CHANNELS,
                                 rate=TARGET_RATE, input=True,
                                 frames_per_buffer=CHUNK)
        except Exception:
            info = self._pa.get_default_input_device_info()
            self._rate = int(info['defaultSampleRate'])
            self._need_rs = (self._rate != TARGET_RATE)
            chunk_dev = int(CHUNK * self._rate / TARGET_RATE)
            return self._pa.open(format=FORMAT, channels=CHANNELS,
                                 rate=self._rate, input=True,
                                 frames_per_buffer=chunk_dev)

    def run(self):
        try:
            stream = self._open_stream()
        except Exception as e:
            print("[audio] cannot open mic:", e)
            return

        self._running = True
        while self._running:
            try:
                raw = stream.read(stream._frames_per_buffer,
                                  exception_on_overflow=False)
            except Exception:
                continue

            if self._need_rs:
                raw, _ = audioop.ratecv(raw, 2, 1,
                                        self._rate, TARGET_RATE, None)

            self.audio_chunk.emit(raw)
            self.data_ready.emit(np.frombuffer(raw, dtype=np.int16))

        stream.stop_stream()
        stream.close()

    def stop(self):
        self._running = False
        self.quit(); self.wait(100)

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
            print("[net] connect error:", e); return
        while self._running:
            QtCore.QThread.msleep(10)
            try:
                for line in line_packet.receive_lines(self.sock):
                    parts = line.strip().split(' ', 2)
                    if len(parts) == 3:
                        self.text_ready.emit(parts[2])
            except BlockingIOError:
                continue
    @QtCore.pyqtSlot(bytes)
    def send_chunk(self, b):
        try: self.sock.send(b)
        except (BlockingIOError, BrokenPipeError, OSError): pass
    def stop(self):
        self._running = False
        if self.sock:
            try: self.sock.shutdown(socket.SHUT_WR); self.sock.close()
            except Exception: pass
        self.quit(); self.wait(100)

class HotkeyListener(QThread):
    """Ctrl+Space → start/stop, Esc → quit"""
    def __init__(self, window):
        super().__init__(); self.window = window; self.ctrl = False
    def run(self):
        from pynput import keyboard
        self.listener = keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release, daemon=True)
        self.listener.start(); self.exec_()
    def on_press(self, key):
        from pynput.keyboard import Key
        if key in (Key.ctrl_l, Key.ctrl_r): self.ctrl = True
        elif key == Key.space and self.ctrl:
            QtCore.QMetaObject.invokeMethod(self.window, '_toggle_recording',
                                            Qt.QueuedConnection)
        elif key == Key.esc:
            QtCore.QMetaObject.invokeMethod(self.window, '_exit_app',
                                            Qt.QueuedConnection)
    def on_release(self, key):
        from pynput.keyboard import Key
        if key in (Key.ctrl_l, Key.ctrl_r): self.ctrl = False
    def stop(self):
        if hasattr(self, 'listener'): self.listener.stop()
        self.quit()

class SuperWhisperWindow(QtWidgets.QWidget):
    def __init__(self, host, port):
        super().__init__()
        self.host, self.port = host, port
        self.kb  = KeyController()
        self.transcriber = None
        self.deepseek = True
        self.replacements = self._load_replacements()
        self._floating_patched = False

        self.waveform_proc = None
        # locate the app‑binary once:
        self.waveform_exec = "/Users/bread/Documents/OpenSuperWhisper/test.app/Contents/MacOS/test"
        if not shutil.which(self.waveform_exec):
            print(f"[warn] cannot find waveform app at {self.waveform_exec}")

        # window flags: on mac never accept focus
        flags = Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint
        if sys.platform == "darwin":
            flags |= Qt.WindowDoesNotAcceptFocus
            self.setAttribute(Qt.WA_ShowWithoutActivating, True)
            self.setFocusPolicy(Qt.NoFocus)
        self.setWindowFlags(flags)

        # tiny UI
        self.setFixedSize(400, 180)
        self.setAttribute(Qt.WA_TranslucentBackground)
        fx = QtWidgets.QGraphicsDropShadowEffect(self)
        fx.setBlurRadius(20); fx.setOffset(0, 0); self.setGraphicsEffect(fx)
        root = QtWidgets.QVBoxLayout(self); root.setContentsMargins(0,0,0,0)
        bg   = QtWidgets.QWidget(self); root.addWidget(bg)
        bg.setStyleSheet("background-color:rgba(255,255,255,230);border-radius:10px;")
        lay  = QtWidgets.QVBoxLayout(bg); lay.setContentsMargins(10,5,10,5)
        pg.setConfigOptions(antialias=True)
        self.plot = pg.PlotWidget(bg); self.plot.setBackground(None)
        self.plot.hideAxis('bottom'); self.plot.hideAxis('left')
        self.bars = pg.BarGraphItem(x=[],height=[],width=0.8)
        self.plot.addItem(self.bars); lay.addWidget(self.plot)
        bar = QtWidgets.QHBoxLayout()
        self.label = QtWidgets.QLabel('Ctrl+Space record • Esc quit')
        self.label.setStyleSheet("color:gray;font:9pt 'Sans';")
        bar.addWidget(self.label)
        self.btn = QtWidgets.QPushButton('Record'); self.btn.setCheckable(True)
        self.btn.clicked.connect(self._toggle_recording)
        bar.addWidget(self.btn); bar.addStretch(1); lay.addLayout(bar)

        self.recorder = AudioRecorder()
        self.recorder.data_ready.connect(self._waveform)
        self.hk = HotkeyListener(self); self.hk.start()

    def _make_floating(self):
        if self._floating_patched or sys.platform != "darwin":
            return
        ns_view = objc.objc_object(c_void_p=int(self.winId()))
        ns_win  = ns_view.window()
        if ns_win is None:                       # shouldn't happen
            return
        ns_win.setLevel_(NSFloatingWindowLevel)
        ns_win.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary
            | NSWindowCollectionBehaviorIgnoresCycle
        )
        self._floating_patched = True

    def _load_replacements(self):
        path = os.path.join(os.path.dirname(__file__), "text_replacements.json")
        if not os.path.exists(path): return []
        with open(path,"r",encoding="utf-8") as f: return json.load(f)

    @QtCore.pyqtSlot()
    def _toggle_recording(self):
        if self.recorder.isRunning():
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        if not self.isVisible():
            self.show()
            self._make_floating()
            g = QtWidgets.QApplication.primaryScreen().availableGeometry()
            self.move((g.width()-self.width())//2, g.height()-self.height())
        if not self.recorder.isRunning():
            self.label.setText('Recording…'); self.label.setStyleSheet("color:red;font:9pt 'Sans';")
            self.btn.setChecked(True)
            self.transcriber = RemoteTranscriber(self.host, self.port)
            self.recorder.audio_chunk.connect(self.transcriber.send_chunk,
                                            QtCore.Qt.QueuedConnection)
            self.transcriber.text_ready.connect(self._type_text)
            self.transcriber.start(); self.recorder.start()
        # launch Swift overlay if not already running
        if self.waveform_proc is None:
            try:
                self.waveform_proc = subprocess.Popen([self.waveform_exec])
            except Exception as e:
                print("[waveform] failed to launch overlay:", e)


    def _stop_recording(self):
        if self.recorder.isRunning(): self.recorder.stop()
        if self.transcriber:
            self.recorder.audio_chunk.disconnect(self.transcriber.send_chunk)
            self.transcriber.stop(); self.transcriber = None
        self.btn.setChecked(False); self.bars.setOpts(x=[],height=[],width=0.8)
        self.label.setText('Ctrl+Space record • Esc quit')
        self.label.setStyleSheet("color:gray;font:9pt 'Sans';")
        # ── kill Swift overlay ────────────────────────────
        if self.waveform_proc:
            try:
                self.waveform_proc.terminate()
                self.waveform_proc.wait(timeout=1)
            except Exception:
                self.waveform_proc.kill()
            finally:
                self.waveform_proc = None

    @QtCore.pyqtSlot()
    def _exit_app(self): 
        self._stop_recording()
        self.hide()

    @QtCore.pyqtSlot(str)
    def _type_text(self, text):
        if self.deepseek:
            for r in self.replacements: text = text.replace(r["from"], r["to"])
        print("[Typed]", text); self.kb.type(text)

    @QtCore.pyqtSlot(np.ndarray)
    def _waveform(self, w):
        h = np.array([np.mean(p) for p in np.array_split(np.abs(w), 60)])
        m = np.max(h) or 1; new = h/m*40
        if not hasattr(self,'_smooth'): self._smooth = new
        else: self._smooth = 0.6*self._smooth + 0.4*new
        self.bars.setOpts(x=np.arange(len(self._smooth)),
                          height=self._smooth, width=0.8)

    def closeEvent(self, ev):
        try: self._stop_recording(); self.hk.stop()
        finally: QtWidgets.QApplication.quit(); ev.accept()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=43007)
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    win = SuperWhisperWindow(args.host, args.port)
    sys.exit(app.exec_())
