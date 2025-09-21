import os
import threading
import time
import csv
import wave
import struct
from queue import Queue

import numpy as np
import serial
from serial.tools import list_ports

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


DEFAULT_BAUD = 921600
STREAM_SECONDS = 2.0  # rolling window length for live waveform


def extract_features(x: np.ndarray, fs: int):
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
    with wave.open(path, 'wb') as wv:
        wv.setnchannels(1)
        wv.setsampwidth(2)
        wv.setframerate(fs)
        wv.writeframes(x.tobytes())


def ensure_csv(csv_path: str):
    need_header = not os.path.exists(csv_path)
    if need_header:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                'ts', 'wav', 'fs', 'nsamples', 'peak', 'rms', 'crest', 'zcr', 'duration_ms',
                'centroid_hz', 'rolloff85_hz', 'bandwidth_hz', 'dominant_hz'
            ])


class SerialWorker(threading.Thread):
    def __init__(self, port: str, baud: int, outdir: str, frame_q: Queue, event_info_q: Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.outdir = outdir
        self.frame_q = frame_q
        self.event_info_q = event_info_q
        self.stop_event = stop_event
        self.ser = None
        os.makedirs(self.outdir, exist_ok=True)
        ensure_csv(os.path.join(self.outdir, 'features.csv'))

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
        except Exception as e:
            self.event_info_q.put(("error", f"無法開啟 {self.port}: {e}"))
            return

        buf = b''
        self.event_info_q.put(("status", f"已連線 {self.port} @ {self.baud}"))
        while not self.stop_event.is_set():
            try:
                chunk = self.ser.read(4096)
            except Exception as e:
                self.event_info_q.put(("error", f"讀取錯誤: {e}"))
                break
            if chunk:
                buf += chunk
            else:
                continue

            while True:
                i_evt = buf.find(b'EVT0')
                i_frm = buf.find(b'FRM0')
                cand = [i for i in [i_evt, i_frm] if i >= 0]
                if not cand:
                    buf = buf[-8:]
                    break
                idx = min(cand)
                tag = buf[idx:idx + 4]
                if idx > 0:
                    buf = buf[idx:]
                if len(buf) < 12:
                    break
                try:
                    sr, count = struct.unpack_from('<II', buf, 4)
                except struct.error:
                    # incomplete header
                    break
                need = 12 + count * 2
                if len(buf) < need:
                    break
                data = buf[12:need]
                buf = buf[need:]

                if tag == b'FRM0':
                    frame = np.frombuffer(data, dtype=np.int16).copy()
                    # push with drop-old policy
                    if self.frame_q.full():
                        try:
                            self.frame_q.get_nowait()
                        except Exception:
                            pass
                    self.frame_q.put((sr, frame))
                elif tag == b'EVT0':
                    x = np.frombuffer(data, dtype=np.int16).copy()
                    ts = time.strftime('%Y%m%d_%H%M%S')
                    wav_name = f'evt_{ts}_{len(x)}.wav'
                    wav_path = os.path.join(self.outdir, wav_name)

                    try:
                        save_wav(wav_path, x, sr)
                        feat = extract_features(x, sr)
                        with open(os.path.join(self.outdir, 'features.csv'), 'a', newline='', encoding='utf-8') as f:
                            csv.writer(f).writerow([ts, wav_name, sr, len(x)] + list(feat))
                        txt = (
                            f"事件 {wav_name} | peak={feat[0]:.3f} rms={feat[1]:.3f} "
                            f"dur={feat[4]:.1f}ms cent={feat[5]:.0f}Hz dom={feat[8]:.0f}Hz"
                        )
                        self.event_info_q.put(("event", txt))
                    except Exception as e:
                        self.event_info_q.put(("error", f"存檔/特徵失敗: {e}"))

        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass
        self.event_info_q.put(("status", "已停止"))


class ImpactMonitorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("ESP32 Impact Monitor")

        # Controls frame
        ctrl = ttk.Frame(root)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        ttk.Label(ctrl, text="Port:").pack(side=tk.LEFT)
        self.port_var = tk.StringVar()
        self.port_box = ttk.Combobox(ctrl, textvariable=self.port_var, width=18, state="readonly")
        self.port_box.pack(side=tk.LEFT, padx=4)
        self.btn_refresh = ttk.Button(ctrl, text="Refresh", command=self.refresh_ports)
        self.btn_refresh.pack(side=tk.LEFT, padx=2)

        ttk.Label(ctrl, text="Baud:").pack(side=tk.LEFT, padx=(10, 0))
        self.baud_var = tk.StringVar(value=str(DEFAULT_BAUD))
        self.baud_entry = ttk.Entry(ctrl, textvariable=self.baud_var, width=10)
        self.baud_entry.pack(side=tk.LEFT, padx=4)

        ttk.Label(ctrl, text="Out Dir:").pack(side=tk.LEFT, padx=(10, 0))
        self.outdir_var = tk.StringVar(value="events")
        self.outdir_entry = ttk.Entry(ctrl, textvariable=self.outdir_var, width=24)
        self.outdir_entry.pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Browse", command=self.browse_outdir).pack(side=tk.LEFT, padx=2)

        ttk.Label(ctrl, text="fmax(Hz):").pack(side=tk.LEFT, padx=(10, 0))
        self.fmax_var = tk.StringVar(value="8000")
        self.fmax_entry = ttk.Entry(ctrl, textvariable=self.fmax_var, width=8)
        self.fmax_entry.pack(side=tk.LEFT, padx=4)

        # Keep primary controls visible
        self.btn_start = ttk.Button(ctrl, text="Start", command=self.start)
        self.btn_start.pack(side=tk.RIGHT, padx=2)
        self.btn_stop = ttk.Button(ctrl, text="Stop", command=self.stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.RIGHT, padx=2)
        self.btn_analysis = ttk.Button(ctrl, text="Analysis", command=self.open_analysis)
        self.btn_analysis.pack(side=tk.RIGHT, padx=4)

        # Status label
        self.status_var = tk.StringVar(value="未連線")
        ttk.Label(root, textvariable=self.status_var).pack(side=tk.TOP, anchor=tk.W, padx=8)

        # Matplotlib Figure
        fig = Figure(figsize=(10, 8), constrained_layout=True)
        self.ax_t = fig.add_subplot(3, 1, 1)
        self.ax_f = fig.add_subplot(3, 1, 2)
        self.ax_spec = fig.add_subplot(3, 1, 3)
        self.ax_t.set_title('Waveform (last seconds)', pad=6)
        self.ax_t.set_xlabel('Time [s]')
        self.ax_t.set_ylabel('Amplitude')
        self.ax_f.set_title('Magnitude Spectrum (Hann)', pad=6)
        self.ax_f.set_xlabel('Frequency [Hz]')
        self.ax_f.set_ylabel('dB')
        self.t_line, = self.ax_t.plot([], [], lw=0.8)
        self.f_line, = self.ax_f.plot([], [], lw=0.8)
        self.ax_f.set_ylim(-100, 5)
        # initialize spectrogram image
        self.spec_im = self.ax_spec.imshow(
            np.zeros((128, 10)),
            aspect='auto', origin='lower', cmap='magma',
            extent=[0, 1, 0, 8000], vmin=-100, vmax=0
        )
        # Colorbar for spectrogram (dB)
        self.spec_cbar = fig.colorbar(self.spec_im, ax=self.ax_spec, orientation='vertical', pad=0.02)
        self.spec_cbar.set_label('dB')
        self.ax_spec.set_title('Spectrogram', pad=6)
        self.ax_spec.set_xlabel('Time [s]')
        self.ax_spec.set_ylabel('Frequency [Hz]')
        # add label padding and tick size to avoid overlap
        for ax in (self.ax_t, self.ax_f, self.ax_spec):
            ax.xaxis.labelpad = 6
            ax.yaxis.labelpad = 6
            ax.tick_params(labelsize=9)

        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Data buffers
        self.stream_fs = None
        self.stream_buf = None

        # Worker thread infra
        self.frame_q = Queue(maxsize=50)
        self.event_info_q = Queue(maxsize=100)
        self.stop_event = threading.Event()
        self.worker = None

        self.refresh_ports()
        self.schedule_ui_update()

        # Menubar to reduce toolbar clutter
        menubar = tk.Menu(root)
        # File menu
        m_file = tk.Menu(menubar, tearoff=0)
        m_file.add_command(label="Analysis", command=self.open_analysis, accelerator="Ctrl+Shift+A")
        m_file.add_separator()
        m_file.add_command(label="Exit", command=self.on_close, accelerator="Ctrl+Q")
        menubar.add_cascade(label="File", menu=m_file)
        # View menu
        m_view = tk.Menu(menubar, tearoff=0)
        m_view.add_command(label="Toggle Fullscreen", command=self.toggle_fullscreen, accelerator="F11")
        menubar.add_cascade(label="View", menu=m_view)
        # Help menu
        m_help = tk.Menu(menubar, tearoff=0)
        m_help.add_command(label="Plot Help", command=self.show_help, accelerator="F1")
        menubar.add_cascade(label="Help", menu=m_help)
        root.config(menu=menubar)

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        # Start in fullscreen and add key bindings
        self.is_fullscreen = False
        self.enter_fullscreen()
        root.bind('<F11>', lambda e: self.toggle_fullscreen())
        root.bind('<Escape>', lambda e: self.exit_fullscreen())
        root.bind('<F1>', lambda e: self.show_help())
        root.bind('<Control-q>', lambda e: self.on_close())
        root.bind('<Control-Q>', lambda e: self.on_close())
        root.bind('<Control-Shift-A>', lambda e: self.open_analysis())

    def show_help(self):
        text = (
            "1、波形 (Waveform)\n\n"
            "圖示內容：顯示訊號在時間上的變化（振幅隨時間的波動）。\n"
            "分析方法：\n\n"
            "從圖中可以看到在 0.10.25 秒之間能量最強，這代表訊號在這段時間內有主要活動。\n\n"
            "波形能幫助我們初步判斷訊號持續時間、強弱變化，以及是否有突發的能量峰值。\n\n"
            "2. 頻譜 (Magnitude Spectrum)\n\n"
            "圖示內容：顯示訊號的整體頻率分布（頻率成分的強度）。\n\n"
            "分析方法：\n\n"
            "橫軸是頻率，縱軸是分貝 (dB)，表示各頻率成分的能量強弱。\n\n"
            "從圖中可以看到訊號能量分布在 08000 Hz 範圍內，但某些頻率帶的能量更明顯。\n\n"
            "這有助於判斷訊號是否包含低頻或高頻特徵，以及可能對應的聲音或現象。\n\n"
            "3. 聲譜圖 (Spectrogram)\n\n"
            "圖示內容：時間與頻率的能量分布（時頻分析）。\n\n"
            "分析方法：\n\n"
            "聲譜圖能同時看到訊號在不同時間點的頻率能量。\n\n"
            "在 0.10.25 秒的區間，頻率範圍大約從 06000 Hz都有明顯能量，和波形的強能量區一致。"
        )
        messagebox.showinfo("圖表說明", text)

    def enter_fullscreen(self):
        try:
            # Fullscreen across platforms
            self.root.attributes('-fullscreen', True)
            self.is_fullscreen = True
        except Exception:
            # Fallback: maximize
            try:
                self.root.state('zoomed')
            except Exception:
                pass

    def exit_fullscreen(self):
        try:
            self.root.attributes('-fullscreen', False)
        except Exception:
            pass
        self.is_fullscreen = False

    def toggle_fullscreen(self):
        if self.is_fullscreen:
            self.exit_fullscreen()
        else:
            self.enter_fullscreen()

    def open_analysis(self):
        # Open a separate window for file-based analysis
        FileAnalysisWindow(self.root, base_dir=self.outdir_var.get(), fmax_getter=lambda: self.fmax_var.get())

    def refresh_ports(self):
        ports = list_ports.comports()
        items = [p.device for p in ports]
        self.port_box['values'] = items
        if items and not self.port_var.get():
            self.port_var.set(items[0])

    def browse_outdir(self):
        d = filedialog.askdirectory()
        if d:
            self.outdir_var.set(d)

    def start(self):
        port = self.port_var.get()
        if not port:
            messagebox.showwarning("提示", "請先選擇序列埠 Port")
            return
        try:
            baud = int(self.baud_var.get())
        except ValueError:
            messagebox.showwarning("提示", "Baud 需為數字")
            return

        # create session subfolder based on start time
        base_out = self.outdir_var.get().strip() or "events"
        session_name = time.strftime('%Y%m%d_%H%M%S')
        session_dir = os.path.join(base_out, session_name)
        os.makedirs(session_dir, exist_ok=True)

        self.stop_event.clear()
        self.worker = SerialWorker(port, baud, session_dir, self.frame_q, self.event_info_q, self.stop_event)
        self.worker.start()
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.port_box.config(state='disabled')
        self.baud_entry.config(state='disabled')
        self.outdir_entry.config(state='disabled')
        self.status_var.set(f"連線中... 儲存至 {session_dir}")

    def stop(self):
        self.stop_event.set()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=2.0)
        self.worker = None
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.port_box.config(state='readonly')
        self.baud_entry.config(state='normal')
        self.outdir_entry.config(state='normal')
        self.status_var.set("未連線")

    def schedule_ui_update(self):
        self.update_ui()
        self.root.after(50, self.schedule_ui_update)

    def update_ui(self):
        # handle messages
        while not self.event_info_q.empty():
            kind, msg = self.event_info_q.get_nowait()
            if kind == "status":
                self.status_var.set(msg)
            elif kind == "event":
                self.status_var.set(msg)
            elif kind == "error":
                self.status_var.set(msg)

        # drain frames and update plots
        updated = False
        while not self.frame_q.empty():
            sr, frame = self.frame_q.get_nowait()
            self.feed_stream(frame, sr)
            updated = True
        if updated:
            self.redraw()

    def feed_stream(self, frame_i16: np.ndarray, fs: int):
        if self.stream_fs != fs or self.stream_buf is None:
            self.stream_fs = fs
            nbuf = int(STREAM_SECONDS * fs)
            nbuf = max(2048, nbuf)
            self.stream_buf = np.zeros(nbuf, dtype=np.float32)
        xf = frame_i16.astype(np.float32) / 32768.0
        n = len(xf)
        if n >= len(self.stream_buf):
            self.stream_buf[:] = xf[-len(self.stream_buf):]
        else:
            self.stream_buf[:-n] = self.stream_buf[n:]
            self.stream_buf[-n:] = xf

    def redraw(self):
        if self.stream_buf is None or self.stream_fs is None:
            return
        # time-domain
        t = np.arange(len(self.stream_buf)) / self.stream_fs
        self.t_line.set_data(t, self.stream_buf)
        self.ax_t.set_xlim(t[0], t[-1])
        ymin, ymax = float(np.min(self.stream_buf)), float(np.max(self.stream_buf))
        pad = max(0.05, 0.1 * (ymax - ymin))
        self.ax_t.set_ylim(ymin - pad, ymax + pad)

        # frequency-domain
        try:
            fmax = float(self.fmax_var.get()) if self.fmax_var.get() else None
        except ValueError:
            fmax = None
        nwin = 1 << (len(self.stream_buf) - 1).bit_length()
        nfft = min(8192, nwin)
        win = np.hanning(nfft)
        X = np.fft.rfft(self.stream_buf[-nfft:] * win)
        mag = 20 * np.log10(np.abs(X) + 1e-12)
        freqs = np.fft.rfftfreq(nfft, 1 / self.stream_fs)
        self.f_line.set_data(freqs, mag)
        self.ax_f.set_xlim(0, fmax if fmax else freqs[-1])
        ymax_db = float(np.max(mag))
        self.ax_f.set_ylim(max(-120, ymax_db - 80), min(5, ymax_db + 5))

        # spectrogram over last seconds
        nfft_s = 1024
        hop = nfft_s // 4
        if len(self.stream_buf) >= nfft_s + hop:
            win_s = np.hanning(nfft_s)
            segs = []
            for i in range(0, len(self.stream_buf) - nfft_s + 1, hop):
                Xs = np.fft.rfft(self.stream_buf[i:i + nfft_s] * win_s)
                segs.append(20 * np.log10(np.abs(Xs) + 1e-12))
            if segs:
                S = np.array(segs).T  # [freq, time]
                freqs_s = np.fft.rfftfreq(nfft_s, 1 / self.stream_fs)
                times_s = np.arange(S.shape[1]) * hop / self.stream_fs
                # limit to fmax if provided
                if fmax:
                    kmax = int(np.searchsorted(freqs_s, fmax))
                    kmax = max(1, min(kmax, S.shape[0]-1))
                else:
                    kmax = S.shape[0]-1
                Splot = S[:kmax+1, :]
                f_top = freqs_s[kmax]
                t0 = times_s[0] if times_s.size else 0.0
                t1 = times_s[-1] if times_s.size else (len(self.stream_buf)/self.stream_fs)
                vmax = float(np.max(Splot))
                vmin = max(-120.0, vmax - 80.0)
                self.spec_im.set_data(Splot)
                self.spec_im.set_extent([t0, t1, 0.0, f_top])
                self.spec_im.set_clim(vmin=vmin, vmax=vmax)
                self.ax_spec.set_ylim(0.0, f_top)
                self.ax_spec.set_xlim(t0, t1)
                if getattr(self, 'spec_cbar', None) is not None:
                    self.spec_cbar.update_normal(self.spec_im)

        self.canvas.draw_idle()

    def on_close(self):
        self.stop()
        self.root.destroy()

class FileAnalysisWindow(tk.Toplevel):
    def __init__(self, master, base_dir: str, fmax_getter):
        super().__init__(master)
        self.title('File Analysis')
        # Try to start maximized; fallback handled after widgets are created
        try:
            self.state('zoomed')
        except Exception:
            pass
        self.fmax_getter = fmax_getter

        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)
        ttk.Label(top, text='Base Dir:').pack(side=tk.LEFT)
        self.base_dir_var = tk.StringVar(value=base_dir or 'events')
        ttk.Entry(top, textvariable=self.base_dir_var, width=40).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text='Browse', command=self.browse_base_dir).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text='Refresh', command=self.refresh_tree).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text='Help', command=self.show_help).pack(side=tk.RIGHT, padx=2)

        body = ttk.Frame(self)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        left = ttk.Frame(body, width=300)
        left.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(body)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(left, columns=("path",), show='tree')
        yscroll = ttk.Scrollbar(left, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=yscroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.Y, expand=True)
        yscroll.pack(side=tk.LEFT, fill=tk.Y)
        self.tree.bind('<<TreeviewSelect>>', self.on_tree_select)

        fig = Figure(figsize=(10, 8), constrained_layout=True)
        self.ax_t = fig.add_subplot(3, 1, 1)
        self.ax_f = fig.add_subplot(3, 1, 2)
        self.ax_s = fig.add_subplot(3, 1, 3)
        self.ax_t.set_title('Waveform (file)', pad=6)
        self.ax_t.set_xlabel('Time [s]')
        self.ax_t.set_ylabel('Amplitude')
        self.ax_f.set_title('Magnitude Spectrum (file)', pad=6)
        self.ax_f.set_xlabel('Frequency [Hz]')
        self.ax_f.set_ylabel('dB')
        for ax in (self.ax_t, self.ax_f, self.ax_s):
            ax.xaxis.labelpad = 6
            ax.yaxis.labelpad = 6
            ax.tick_params(labelsize=9)
        self.t_line, = self.ax_t.plot([], [], lw=0.8)
        self.f_line, = self.ax_f.plot([], [], lw=0.8)
        self.ax_f.set_ylim(-100, 5)
        self.spec_im = self.ax_s.imshow(
            np.zeros((128, 10)), aspect='auto', origin='lower', cmap='magma',
            extent=[0, 1, 0, 8000], vmin=-100, vmax=0
        )
        self.ax_s.set_title('Spectrogram (file)', pad=6)
        self.ax_s.set_xlabel('Time [s]')
        self.ax_s.set_ylabel('Frequency [Hz]')
        # Colorbar for file spectrogram (dB)
        self.spec_cbar = fig.colorbar(self.spec_im, ax=self.ax_s, orientation='vertical', pad=0.02)
        self.spec_cbar.set_label('dB')

        self.canvas = FigureCanvasTkAgg(fig, master=right)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Menubar similar to main window to reduce toolbar clutter
        menubar = tk.Menu(self)
        # File menu
        m_file = tk.Menu(menubar, tearoff=0)
        m_file.add_command(label='Browse Base Dir...', command=self.browse_base_dir, accelerator='Ctrl+O')
        m_file.add_command(label='Refresh', command=self.refresh_tree, accelerator='F5')
        m_file.add_separator()
        m_file.add_command(label='Close', command=self.on_close, accelerator='Ctrl+W')
        menubar.add_cascade(label='File', menu=m_file)
        # View menu
        m_view = tk.Menu(menubar, tearoff=0)
        m_view.add_command(label='Toggle Fullscreen', command=self.toggle_fullscreen, accelerator='F11')
        menubar.add_cascade(label='View', menu=m_view)
        # Help menu
        m_help = tk.Menu(menubar, tearoff=0)
        m_help.add_command(label='Plot Help', command=self.show_help, accelerator='F1')
        menubar.add_cascade(label='Help', menu=m_help)
        self.config(menu=menubar)

        # Key bindings
        self.is_fullscreen = False
        self.bind('<F11>', lambda e: self.toggle_fullscreen())
        self.bind('<Escape>', lambda e: self.exit_fullscreen())
        self.bind('<F1>', lambda e: self.show_help())
        self.bind('<Control-o>', lambda e: self.browse_base_dir())
        self.bind('<F5>', lambda e: self.refresh_tree())
        self.bind('<Control-w>', lambda e: self.on_close())
        self.protocol('WM_DELETE_WINDOW', self.on_close)

        self.refresh_tree()
        # Ensure the window fits the screen once laid out
        self.after(50, self._fit_to_screen)

    def browse_base_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.base_dir_var.set(d)
            self.refresh_tree()

    def show_help(self):
        text = (
            "、波形 (Waveform)\n\n"
            "圖示內容：顯示訊號在時間上的變化（振幅隨時間的波動）。\n"
            "分析方法：\n\n"
            "從圖中可以看到在 0.10.25 秒之間能量最強，這代表訊號在這段時間內有主要活動。\n\n"
            "波形能幫助我們初步判斷訊號持續時間、強弱變化，以及是否有突發的能量峰值。\n\n"
            "2. 頻譜 (Magnitude Spectrum)\n\n"
            "圖示內容：顯示訊號的整體頻率分布（頻率成分的強度）。\n\n"
            "分析方法：\n\n"
            "橫軸是頻率，縱軸是分貝 (dB)，表示各頻率成分的能量強弱。\n\n"
            "從圖中可以看到訊號能量分布在 08000 Hz 範圍內，但某些頻率帶的能量更明顯。\n\n"
            "這有助於判斷訊號是否包含低頻或高頻特徵，以及可能對應的聲音或現象。\n\n"
            "3. 聲譜圖 (Spectrogram)\n\n"
            "圖示內容：時間與頻率的能量分布（時頻分析）。\n\n"
            "分析方法：\n\n"
            "聲譜圖能同時看到訊號在不同時間點的頻率能量。\n\n"
            "在 0.10.25 秒的區間，頻率範圍大約從 06000 Hz都有明顯能量，和波形的強能量區一致。"
        )
        messagebox.showinfo('圖表說明', text)

    # Fullscreen helpers for analysis window
    def enter_fullscreen(self):
        try:
            self.attributes('-fullscreen', True)
            self.is_fullscreen = True
        except Exception:
            try:
                self.state('zoomed')
                self.is_fullscreen = True
            except Exception:
                pass

    def exit_fullscreen(self):
        try:
            self.attributes('-fullscreen', False)
        except Exception:
            pass
        self.is_fullscreen = False

    def toggle_fullscreen(self):
        if self.is_fullscreen:
            self.exit_fullscreen()
        else:
            self.enter_fullscreen()

    def _fit_to_screen(self):
        # Robust maximize across platforms
        ok = False
        try:
            self.state('zoomed')
            ok = True
        except Exception:
            pass
        if not ok:
            try:
                self.update_idletasks()
                sw = self.winfo_screenwidth()
                sh = self.winfo_screenheight()
                self.geometry(f"{sw}x{sh}+0+0")
            except Exception:
                pass

    def refresh_tree(self):
        base = self.base_dir_var.get().strip() or 'events'
        self.tree.delete(*self.tree.get_children())
        if not os.path.isdir(base):
            return
        sessions = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        sessions.sort(reverse=True)
        for s in sessions:
            sdir = os.path.join(base, s)
            sid = self.tree.insert('', 'end', text=s, values=(sdir,))
            wavs = [f for f in os.listdir(sdir) if f.lower().endswith('.wav')]
            wavs.sort()
            for wname in wavs:
                wpath = os.path.join(sdir, wname)
                self.tree.insert(sid, 'end', text=wname, values=(wpath,))

    def on_tree_select(self, event=None):
        sel = self.tree.focus()
        if not sel:
            return
        vals = self.tree.item(sel, 'values')
        if not vals:
            return
        path = vals[0]
        if os.path.isfile(path) and path.lower().endswith('.wav'):
            self.load_and_plot(path)

    def load_and_plot(self, wav_path: str):
        try:
            with wave.open(wav_path, 'rb') as w:
                fs = w.getframerate()
                n = w.getnframes()
                x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as e:
            messagebox.showerror('Error', f'讀取失敗: {e}')
            return

        # waveform
        t = np.arange(len(x)) / fs
        self.t_line.set_data(t, x)
        self.ax_t.set_xlim(t[0] if len(t) else 0, t[-1] if len(t) else 1)
        ymin, ymax = float(np.min(x) if len(x) else -1), float(np.max(x) if len(x) else 1)
        pad = max(0.05, 0.1 * (ymax - ymin))
        self.ax_t.set_ylim(ymin - pad, ymax + pad)

        # FFT
        try:
            fmax = float(self.fmax_getter()) if self.fmax_getter() else None
        except Exception:
            fmax = None
        nfft = 1 << (len(x) - 1).bit_length()
        nfft = min(16384, max(1024, nfft))
        win = np.hanning(min(len(x), nfft))
        xx = x[:len(win)]
        X = np.fft.rfft(xx * win, n=nfft)
        mag = 20 * np.log10(np.abs(X) + 1e-12)
        freqs = np.fft.rfftfreq(nfft, 1 / fs)
        self.f_line.set_data(freqs, mag)
        self.ax_f.set_xlim(0, fmax if fmax else freqs[-1])
        ymax_db = float(np.max(mag))
        self.ax_f.set_ylim(max(-120, ymax_db - 80), min(5, ymax_db + 5))

        # Spectrogram
        nfft_s = 1024
        hop = nfft_s // 4
        if len(x) >= nfft_s + hop:
            win_s = np.hanning(nfft_s)
            segs = []
            for i in range(0, len(x) - nfft_s + 1, hop):
                Xs = np.fft.rfft(x[i:i + nfft_s] * win_s)
                segs.append(20 * np.log10(np.abs(Xs) + 1e-12))
            if segs:
                S = np.array(segs).T
                freqs_s = np.fft.rfftfreq(nfft_s, 1 / fs)
                times_s = np.arange(S.shape[1]) * hop / fs
                if fmax:
                    kmax = int(np.searchsorted(freqs_s, fmax))
                    kmax = max(1, min(kmax, S.shape[0]-1))
                else:
                    kmax = S.shape[0]-1
                Splot = S[:kmax+1, :]
                f_top = freqs_s[kmax]
                t0 = times_s[0] if times_s.size else 0.0
                t1 = times_s[-1] if times_s.size else (len(x)/fs)
                vmax = float(np.max(Splot))
                vmin = max(-120.0, vmax - 80.0)
                self.spec_im.set_data(Splot)
                self.spec_im.set_extent([t0, t1, 0.0, f_top])
                self.spec_im.set_clim(vmin=vmin, vmax=vmax)
                self.ax_s.set_ylim(0.0, f_top)
                self.ax_s.set_xlim(t0, t1)
                if getattr(self, 'spec_cbar', None) is not None:
                    self.spec_cbar.update_normal(self.spec_im)

        self.canvas.draw_idle()


def main():
    root = tk.Tk()
    ImpactMonitorApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
