"""
Real-time Impact Classifier Monitor (階段1)
支援新的雙協定：RSLT (判斷結果) + EVT0 (完整音訊)
20Hz 判斷頻率，即時可視化
"""

import os
import threading
import time
import csv
import wave
import struct
from queue import Queue
from datetime import datetime

import numpy as np
import serial
from serial.tools import list_ports

import tkinter as tk
from tkinter import ttk, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


DEFAULT_BAUD = 921600
STREAM_SECONDS = 2.0  # 波形顯示窗口


def extract_features(x: np.ndarray, fs: int):
    """計算音訊特徵（用於 EVT0 封包）"""
    xf = x.astype(np.float32) / 32768.0
    peak = float(np.max(np.abs(xf)) + 1e-12)
    rms = float(np.sqrt(np.mean(xf * xf)) + 1e-12)
    crest = peak / rms
    zcr = float(np.mean((xf[:-1] * xf[1:]) < 0)) if len(xf) > 1 else 0.0

    thr = peak * (10 ** (-20 / 20))
    idx = np.where(np.abs(xf) >= thr)[0]
    duration_ms = float((idx[-1] - idx[0] + 1) / fs * 1000.0) if idx.size > 0 else 0.0

    win = np.hanning(len(xf)) if len(xf) > 2 else np.ones_like(xf)
    X = np.fft.rfft(xf * win)
    mag = np.abs(X) + 1e-12
    freqs = np.fft.rfftfreq(len(xf), d=1.0 / fs)
    centroid = float(np.sum(freqs * mag) / np.sum(mag))
    cumsum = np.cumsum(mag)
    roll_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
    rolloff85 = float(freqs[min(roll_idx, len(freqs) - 1)])
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / np.sum(mag)))
    dominant = float(freqs[int(np.argmax(mag))])
    return peak, rms, crest, zcr, duration_ms, centroid, rolloff85, bandwidth, dominant


def save_wav(path: str, x: np.ndarray, fs: int):
    """儲存 WAV 檔案"""
    with wave.open(path, 'wb') as wv:
        wv.setnchannels(1)
        wv.setsampwidth(2)
        wv.setframerate(fs)
        wv.writeframes(x.tobytes())


def ensure_result_csv(csv_path: str):
    """建立判斷結果 CSV（RSLT 封包）"""
    need_header = not os.path.exists(csv_path)
    if need_header:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                'timestamp_ms', 'result', 'rms', 'peak'
            ])


def ensure_feature_csv(csv_path: str):
    """建立完整特徵 CSV（EVT0 封包）"""
    need_header = not os.path.exists(csv_path)
    if need_header:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                'timestamp_ms', 'wav', 'fs', 'nsamples', 'result',
                'peak', 'rms', 'crest', 'zcr', 'duration_ms',
                'centroid_hz', 'rolloff85_hz', 'bandwidth_hz', 'dominant_hz'
            ])


class SerialWorker(threading.Thread):
    """串口工作執行緒：解析 RSLT 和 EVT0 封包"""

    def __init__(
        self,
        port: str,
        baud: int,
        outdir: str,
        result_q: Queue,  # RSLT 封包佇列
        audio_q: Queue,   # EVT0 封包佇列
        stop_event: threading.Event,
    ):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.outdir = outdir
        self.result_q = result_q
        self.audio_q = audio_q
        self.stop_event = stop_event
        self.ser = None

        os.makedirs(self.outdir, exist_ok=True)
        ensure_result_csv(os.path.join(self.outdir, 'results.csv'))
        ensure_feature_csv(os.path.join(self.outdir, 'features.csv'))

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(0.5)

            while not self.stop_event.is_set():
                # 尋找封包標頭
                magic = self.ser.read(4)
                if len(magic) != 4:
                    continue

                if magic == b'RSLT':
                    self.parse_rslt()
                elif magic == b'EVT0':
                    self.parse_evt0()

        except Exception as e:
            print(f"SerialWorker error: {e}")
        finally:
            if self.ser:
                self.ser.close()

    def parse_rslt(self):
        """解析 RSLT 封包：timestamp(4) + result(1) + rms(4) + peak(4)"""
        try:
            data = self.ser.read(13)
            if len(data) != 13:
                return

            timestamp_ms, = struct.unpack('<I', data[0:4])
            result, = struct.unpack('<B', data[4:5])
            rms, = struct.unpack('<f', data[5:9])
            peak, = struct.unpack('<f', data[9:13])

            # 寫入 CSV
            csv_path = os.path.join(self.outdir, 'results.csv')
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([timestamp_ms, result, rms, peak])

            # 送入佇列供 GUI 顯示
            self.result_q.put({
                'timestamp_ms': timestamp_ms,
                'result': result,
                'rms': rms,
                'peak': peak
            })

        except Exception as e:
            print(f"RSLT parse error: {e}")

    def parse_evt0(self):
        """解析 EVT0 封包：fs(4) + count(4) + result(1) + int16[]"""
        try:
            header = self.ser.read(9)
            if len(header) != 9:
                return

            fs, = struct.unpack('<I', header[0:4])
            count, = struct.unpack('<I', header[4:8])
            result, = struct.unpack('<B', header[8:9])

            audio_bytes = self.ser.read(count * 2)
            if len(audio_bytes) != count * 2:
                return

            audio = np.frombuffer(audio_bytes, dtype=np.int16)

            # 儲存 WAV
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            wav_name = f"chunk_{timestamp_str}_r{result}.wav"
            wav_path = os.path.join(self.outdir, wav_name)
            save_wav(wav_path, audio, fs)

            # 計算完整特徵
            features = extract_features(audio, fs)

            # 寫入 features.csv
            csv_path = os.path.join(self.outdir, 'features.csv')
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([
                    timestamp_str, wav_name, fs, count, result, *features
                ])

            # 送入佇列供 GUI 顯示
            self.audio_q.put({
                'audio': audio,
                'fs': fs,
                'result': result,
                'wav_name': wav_name
            })

        except Exception as e:
            print(f"EVT0 parse error: {e}")


class ClassifierMonitorGUI:
    """即時分類監控 GUI"""

    def __init__(self, root):
        self.root = root
        self.root.title("即時撞擊分類監控 (20Hz)")
        self.root.geometry("1200x800")

        self.worker = None
        self.stop_event = threading.Event()
        self.result_q = Queue()
        self.audio_q = Queue()

        self.build_ui()
        self.init_plot()

        # 定時更新
        self.root.after(50, self.update_display)

    def build_ui(self):
        """建立 UI"""
        # 控制面板
        ctrl_frame = ttk.Frame(self.root)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        ttk.Label(ctrl_frame, text="Port:").pack(side=tk.LEFT, padx=5)
        self.port_combo = ttk.Combobox(ctrl_frame, width=15, state='readonly')
        self.port_combo['values'] = [p.device for p in list_ports.comports()]
        if self.port_combo['values']:
            self.port_combo.current(0)
        self.port_combo.pack(side=tk.LEFT, padx=5)

        ttk.Label(ctrl_frame, text="Baud:").pack(side=tk.LEFT, padx=5)
        self.baud_var = tk.StringVar(value=str(DEFAULT_BAUD))
        ttk.Entry(ctrl_frame, textvariable=self.baud_var, width=10).pack(side=tk.LEFT, padx=5)

        ttk.Label(ctrl_frame, text="Out Dir:").pack(side=tk.LEFT, padx=5)
        self.outdir_var = tk.StringVar(value="events")
        ttk.Entry(ctrl_frame, textvariable=self.outdir_var, width=20).pack(side=tk.LEFT, padx=5)

        self.start_btn = ttk.Button(ctrl_frame, text="Start", command=self.start_capture)
        self.start_btn.pack(side=tk.LEFT, padx=10)

        self.stop_btn = ttk.Button(ctrl_frame, text="Stop", command=self.stop_capture, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # 狀態顯示
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.status_label = ttk.Label(status_frame, text="狀態: 未啟動", font=('Arial', 12, 'bold'))
        self.status_label.pack(side=tk.LEFT, padx=10)

        self.result_label = ttk.Label(status_frame, text="判斷: -", font=('Arial', 14, 'bold'), foreground='black')
        self.result_label.pack(side=tk.LEFT, padx=20)

        self.rms_label = ttk.Label(status_frame, text="RMS: -", font=('Arial', 10))
        self.rms_label.pack(side=tk.LEFT, padx=10)

        # 繪圖區域
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig = Figure(figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def init_plot(self):
        """初始化繪圖"""
        self.ax_wave = self.fig.add_subplot(2, 2, 1)
        self.ax_fft = self.fig.add_subplot(2, 2, 2)
        self.ax_spec = self.fig.add_subplot(2, 1, 2)

        self.ax_wave.set_title('波形 (最新音訊 chunk)')
        self.ax_wave.set_xlabel('Time [s]')
        self.ax_wave.set_ylabel('Amplitude')

        self.ax_fft.set_title('頻譜 (FFT)')
        self.ax_fft.set_xlabel('Frequency [Hz]')
        self.ax_fft.set_ylabel('Magnitude [dB]')

        self.ax_spec.set_title('頻譜圖 (Spectrogram)')
        self.ax_spec.set_xlabel('Time [s]')
        self.ax_spec.set_ylabel('Frequency [Hz]')

        self.fig.tight_layout()

    def start_capture(self):
        """啟動擷取"""
        port = self.port_combo.get()
        if not port:
            messagebox.showerror("錯誤", "請選擇序列埠")
            return

        baud = int(self.baud_var.get())
        outdir_base = self.outdir_var.get()

        # 建立時間戳資料夾
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        outdir = os.path.join(outdir_base, timestamp)

        self.stop_event.clear()
        self.worker = SerialWorker(port, baud, outdir, self.result_q, self.audio_q, self.stop_event)
        self.worker.start()

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text=f"狀態: 執行中 (輸出至 {outdir})")

    def stop_capture(self):
        """停止擷取"""
        self.stop_event.set()
        if self.worker:
            self.worker.join(timeout=2)

        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="狀態: 已停止")

    def update_display(self):
        """更新顯示（定時呼叫）"""
        # 處理 RSLT 封包
        while not self.result_q.empty():
            result_data = self.result_q.get()
            result = result_data['result']
            rms = result_data['rms']

            # 更新狀態標籤
            if result == 1:
                self.result_label.config(text="判斷: ✓ (1)", foreground='green')
            else:
                self.result_label.config(text="判斷: ✗ (0)", foreground='red')

            self.rms_label.config(text=f"RMS: {rms:.2f}")

        # 處理 EVT0 封包（更新波形/頻譜）
        while not self.audio_q.empty():
            audio_data = self.audio_q.get()
            audio = audio_data['audio']
            fs = audio_data['fs']
            result = audio_data['result']

            self.plot_audio(audio, fs, result)

        # 繼續定時更新
        self.root.after(50, self.update_display)

    def plot_audio(self, audio: np.ndarray, fs: int, result: int):
        """繪製音訊波形和頻譜"""
        xf = audio.astype(np.float32) / 32768.0
        t = np.arange(len(xf)) / fs

        # 波形
        self.ax_wave.clear()
        self.ax_wave.plot(t, xf, lw=0.8)
        self.ax_wave.set_title(f'波形 (result={result})')
        self.ax_wave.set_xlabel('Time [s]')
        self.ax_wave.set_ylabel('Amplitude')
        self.ax_wave.set_xlim(0, t[-1])
        self.ax_wave.set_ylim(-1, 1)

        # FFT
        win = np.hanning(len(xf))
        X = np.fft.rfft(xf * win)
        mag = 20 * np.log10(np.abs(X) + 1e-12)
        freqs = np.fft.rfftfreq(len(xf), 1/fs)

        self.ax_fft.clear()
        self.ax_fft.plot(freqs, mag, lw=0.8)
        self.ax_fft.set_title('頻譜 (FFT)')
        self.ax_fft.set_xlabel('Frequency [Hz]')
        self.ax_fft.set_ylabel('Magnitude [dB]')
        self.ax_fft.set_xlim(0, 8000)
        self.ax_fft.set_ylim(-100, 0)

        # Spectrogram
        self.ax_spec.clear()
        nfft = 256
        hop = nfft // 4
        S = []
        for i in range(0, len(xf) - nfft, hop):
            X_seg = np.fft.rfft(np.hanning(nfft) * xf[i:i+nfft])
            S.append(20 * np.log10(np.abs(X_seg) + 1e-12))

        if S:
            S = np.array(S).T
            freqs_spec = np.fft.rfftfreq(nfft, 1/fs)
            times_spec = np.arange(S.shape[1]) * hop / fs

            self.ax_spec.pcolormesh(times_spec, freqs_spec, S, shading='auto', cmap='magma', vmin=-80, vmax=0)
            self.ax_spec.set_ylim(0, 8000)
            self.ax_spec.set_title('頻譜圖 (Spectrogram)')
            self.ax_spec.set_xlabel('Time [s]')
            self.ax_spec.set_ylabel('Frequency [Hz]')

        self.canvas.draw()


def main():
    root = tk.Tk()
    app = ClassifierMonitorGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
