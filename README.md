pip install numpy librosa pyinput keyboard pyaudio pynput pyqt5 pyqtgraph faster_whisper

windows 
install cudnn
add cudnn path C:\Program Files\NVIDIA\CUDNN\v9.8\bin\12.8 and C:\Program Files\NVIDIA\CUDNN\v9.8\bin to environment variables

python3 whisper_online_server.py --host localhost --port 43007 --model tiny --lan en --backend faster-whisper
ollama pull phi4-mini
python super_whisper.py