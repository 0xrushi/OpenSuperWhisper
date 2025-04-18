import sys
import numpy as np
import pyaudio
import socket
import line_packet
from pynput.keyboard import Controller as KeyController
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, pyqtSignal, QThread
import pyqtgraph as pg

# No local ASR: use remote whisper_online_server
# Constants for audio
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
# server expects 16kHz
RATE = 16000 

class AudioRecorder(QThread):
    audio_chunk = pyqtSignal(bytes)
    data_ready = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._running = False
        self._pa = pyaudio.PyAudio()

    def run(self):
        try:
            stream = self._pa.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
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
            wave = np.frombuffer(data, dtype=np.int16)
            self.data_ready.emit(wave)
        stream.stop_stream()
        stream.close()

    def stop(self):
        self._running = False
        self.wait()

class RemoteTranscriber(QThread):
    text_ready = pyqtSignal(str)

    def __init__(self, host, port):
        super().__init__()
        self.host = host
        self.port = port
        self._running = True
        self.sock = None

    def run(self):
        # connect to whisper_online_server
        try:
            self.sock = socket.create_connection((self.host, self.port))
            self.sock.setblocking(False)
        except Exception as e:
            print(f"Could not connect to server: {e}")
            return
        while self._running:
            QtCore.QThread.msleep(10)
            try:
                lines = line_packet.receive_lines(self.sock)
                for line in lines:
                    parts = line.strip().split(' ', 2)
                    if len(parts) == 3:
                        _, _, text = parts
                        self.text_ready.emit(text)
            except BlockingIOError:
                continue

    @QtCore.pyqtSlot(bytes)
    def send_chunk(self, chunk: bytes):
        # non-blocking send; ignore errors
        try:
            self.sock.send(chunk)
        except (BlockingIOError, BrokenPipeError, OSError):
            pass
    
    @QtCore.pyqtSlot(str)
    def _display_and_type(self, text):
        print(f"Transcribed: {text}")
        self.kb.type(text)


    def stop(self):
        self._running = False
        if self.sock:
            try:
                # signal EOF by shutting down write
                self.sock.shutdown(socket.SHUT_WR)
                self.sock.close()
            except Exception:
                pass
        self.wait()

class HotkeyListener(QThread):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.ctrl = False
        self.listener = None

    def run(self):
        from pynput import keyboard
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release,
            daemon=True
        )
        self.listener.start()
        self.exec_()

    def on_press(self, key):
        from pynput.keyboard import Key
        if key in (Key.ctrl_l, Key.ctrl_r):
            self.ctrl = True
        elif key == Key.space and self.ctrl:
            QtCore.QMetaObject.invokeMethod(self.window, '_start_recording', Qt.QueuedConnection)
        elif key == Key.esc:
            QtCore.QMetaObject.invokeMethod(self.window, 'close', Qt.QueuedConnection)
        elif key == Key.esc:
            QtCore.QMetaObject.invokeMethod(self.window, '_close_window', Qt.QueuedConnection)


    def on_release(self, key):
        from pynput.keyboard import Key
        if key in (Key.ctrl_l, Key.ctrl_r):
            self.ctrl = False
        elif key == Key.space:
            QtCore.QMetaObject.invokeMethod(self.window, '_stop_recording', Qt.QueuedConnection)

    def stop(self):
        if self.listener:
            self.listener.stop()
        self.quit()

class SuperWhisperWindow(QtWidgets.QWidget):
    def __init__(self, host, port):
        super().__init__()
        self.kb = KeyController()
        self.transcriber = None
        self.host = host
        self.port = port

        self.setWindowFlags(
        Qt.FramelessWindowHint |
        Qt.Tool |  # prevents stealing focus
        Qt.WindowStaysOnTopHint
        )

        self.setFixedSize(400, 180)
        self.setAttribute(Qt.WA_TranslucentBackground)
        effect = QtWidgets.QGraphicsDropShadowEffect(self)
        effect.setBlurRadius(20)
        effect.setOffset(0, 0)
        self.setGraphicsEffect(effect)

        main = QtWidgets.QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        bg = QtWidgets.QWidget(self)
        bg.setStyleSheet("background-color: rgba(255,255,255,230); border-radius:10px;")
        bg_layout = QtWidgets.QVBoxLayout(bg)
        bg_layout.setContentsMargins(10, 5, 10, 5)
        main.addWidget(bg)

        pg.setConfigOptions(antialias=True)
        self.plot = pg.PlotWidget(bg)
        self.plot.setBackground(None)
        self.plot.hideAxis('bottom')
        self.plot.hideAxis('left')
        self.bars = pg.BarGraphItem(x=[], height=[], width=0.8)
        self.plot.addItem(self.bars)
        bg_layout.addWidget(self.plot)

        status = QtWidgets.QHBoxLayout()
        self.label = QtWidgets.QLabel('Press Ctrl+Space to Record')
        self.label.setStyleSheet("color:gray; font:9pt 'Sans';")
        status.addWidget(self.label, alignment=Qt.AlignLeft)
        self.btn = QtWidgets.QPushButton('Record')
        self.btn.setCheckable(True)
        self.btn.clicked.connect(self._toggle_recording)
        status.addWidget(self.btn, alignment=Qt.AlignCenter)
        status.addWidget(QtWidgets.QLabel('Esc to Close'), alignment=Qt.AlignRight)
        bg_layout.addLayout(status)

        self.recorder = AudioRecorder()
        self.recorder.data_ready.connect(self.update_waveform)

        self.hk = HotkeyListener(self)
        self.hk.start()
    
    @QtCore.pyqtSlot()
    def _stop_recording(self):
        if self.recorder.isRunning():
            self.recorder.stop()
            self.label.setText('Stopped')
            self.label.setStyleSheet("color:gray; font:9pt 'Sans';")
            if self.transcriber:
                self.recorder.audio_chunk.disconnect(self.transcriber.send_chunk)
                self.transcriber.stop()
                self.transcriber = None
            self.btn.setChecked(False)
            self.bars.setOpts(x=[], height=[], width=0.8)
            CLOSE = False
            if CLOSE:
                QtCore.QTimer.singleShot(500, self._close_window)
                exit(0)



    def _toggle_recording(self, checked):
        if checked:
            self._start_recording()
        else:
            self._stop_recording()

    @QtCore.pyqtSlot()
    def _start_recording(self):
        if not self.isVisible():
            self.show()
            screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
            self.move((screen.width() - self.width()) // 2, screen.height() - self.height())

        if not self.recorder.isRunning():
            self.label.setText('Recording...')
            self.label.setStyleSheet("color:red; font:9pt 'Sans';")
            self.btn.setChecked(True)
            self.transcriber = RemoteTranscriber(self.host, self.port)
            self.recorder.audio_chunk.connect(self.transcriber.send_chunk, QtCore.Qt.QueuedConnection)
            self.transcriber.text_ready.connect(self._display_and_type)
            self.transcriber.start()
            self.recorder.start()

    @QtCore.pyqtSlot(str)
    def _display_and_type(self, text):
        self.kb.type(text)
    
    @QtCore.pyqtSlot()
    def _close_window(self):
        print("[INFO] Closing application")
        try:
            if self.transcriber and self.transcriber.isRunning():
                self.recorder.audio_chunk.disconnect(self.transcriber.send_chunk)
                self.transcriber.stop()
                self.transcriber = None

            if self.recorder and self.recorder.isRunning():
                self.recorder.stop()

            self.hk.stop()
        except Exception as e:
            print(f"[ERROR] Exception during shutdown: {e}")
        
        QtWidgets.QApplication.quit()


 
    @QtCore.pyqtSlot(np.ndarray)
    def update_waveform(self, w):
        parts = np.array_split(np.abs(w), 60)
        heights = [np.mean(p) for p in parts]
        m = max(heights) or 1
        heights = [h / m * 40 for h in heights]
        x = np.arange(len(heights))
        self.bars.setOpts(x=x, height=heights, width=0.8)

    def closeEvent(self, event):
        self._close_window()
        event.accept()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=43007)
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    win = SuperWhisperWindow(args.host, args.port)
    # geom = app.primaryScreen().availableGeometry()
    # win.move((geom.width() - win.width()) // 2, geom.height() - win.height())
    sys.exit(app.exec_())
