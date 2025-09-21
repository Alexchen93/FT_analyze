import argparse, os, time, csv, wave, struct
import serial
import numpy as np
import matplotlib.pyplot as plt


def extract_features(x: np.ndarray, fs: int):
    xf = x.astype(np.float32) / 32768.0
    peak = float(np.max(np.abs(xf)) + 1e-12)
    rms = float(np.sqrt(np.mean(xf * xf)) + 1e-12)
    crest = peak / rms
    zcr = float(np.mean((xf[:-1] * xf[1:]) < 0)) if len(xf) > 1 else 0.0

    thr = peak * (10 ** (-20 / 20))
    idx = np.where(np.abs(xf) >= thr)[0]
    duration_ms = float((idx[-1] - idx[0] + 1) / fs * 1000.0) if idx.size > 0 else 0.0

    # Spectrum features
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


class LiveFFT:
    def __init__(self, fmax: float = None):
        plt.ion()
        self.fig, (self.ax_t, self.ax_f) = plt.subplots(2, 1, figsize=(10, 6))
        # time-domain
        self.t_line, = self.ax_t.plot([], [], lw=0.8)
        self.ax_t.set_xlabel('Time [s]')
        self.ax_t.set_ylabel('Amplitude')
        self.ax_t.set_title('Last Event - Waveform')
        # freq-domain
        self.f_line, = self.ax_f.plot([], [], lw=0.8)
        self.ax_f.set_xlabel('Frequency [Hz]')
        self.ax_f.set_ylabel('Magnitude [dB]')
        self.ax_f.set_title('Last Event - Magnitude Spectrum (Hann)')
        if fmax:
            self.ax_f.set_xlim(0, fmax)
        self.ax_f.set_ylim(-120, 0)
        # streaming state
        self.stream_fs = None
        self.stream_sec = 2.0  # keep last 2 seconds
        self.stream_buf = None
        self.fmax = fmax

    def update(self, x: np.ndarray, fs: int, fmax: float = None):
        # time
        t = np.arange(len(x)) / fs
        xf = x.astype(np.float32) / 32768.0
        self.t_line.set_data(t, xf)
        self.ax_t.set_xlim(0, max(1e-3, t[-1] if len(t) else 1))
        ymin, ymax = float(np.min(xf, initial=-1)), float(np.max(xf, initial=1))
        pad = max(0.1, 0.1 * (ymax - ymin))
        self.ax_t.set_ylim(ymin - pad, ymax + pad)

        # spectrum (dB, Hann window)
        n = len(xf)
        nfft = 1 << (n - 1).bit_length()  # next pow2
        win = np.hanning(n)
        X = np.fft.rfft(xf * win, n=nfft)
        mag = 20 * np.log10(np.abs(X) + 1e-12)
        freqs = np.fft.rfftfreq(nfft, 1 / fs)
        self.f_line.set_data(freqs, mag)
        self.ax_f.set_xlim(0, fmax if fmax else freqs[-1])
        # autoscale Y within [-120, 0] hard limits
        ymax_db = float(np.max(mag))
        self.ax_f.set_ylim(max(-120, ymax_db - 80), min(0, ymax_db + 5))

        plt.pause(0.001)

    def update_stream(self, frame_i16: np.ndarray, fs: int):
        if self.stream_fs != fs or self.stream_buf is None:
            self.stream_fs = fs
            nbuf = int(self.stream_sec * fs)
            nbuf = max(2048, nbuf)
            self.stream_buf = np.zeros(nbuf, dtype=np.float32)
        # append new frame
        xf = frame_i16.astype(np.float32) / 32768.0
        n = len(xf)
        if n >= len(self.stream_buf):
            self.stream_buf[:] = xf[-len(self.stream_buf):]
        else:
            self.stream_buf[:-n] = self.stream_buf[n:]
            self.stream_buf[-n:] = xf
        # time plot
        t = np.arange(len(self.stream_buf)) / self.stream_fs
        self.t_line.set_data(t, self.stream_buf)
        self.ax_t.set_xlim(t[0], t[-1])
        ymin, ymax = float(np.min(self.stream_buf)), float(np.max(self.stream_buf))
        pad = max(0.05, 0.1 * (ymax - ymin))
        self.ax_t.set_ylim(ymin - pad, ymax + pad)
        # spectrum of latest window
        nwin = 1 << (len(self.stream_buf) - 1).bit_length()
        nfft = min(8192, nwin)
        win = np.hanning(nfft)
        X = np.fft.rfft(self.stream_buf[-nfft:] * win)
        mag = 20 * np.log10(np.abs(X) + 1e-12)
        freqs = np.fft.rfftfreq(nfft, 1 / self.stream_fs)
        self.f_line.set_data(freqs, mag)
        self.ax_f.set_xlim(0, self.fmax if self.fmax else freqs[-1])
        ymax_db = float(np.max(mag))
        self.ax_f.set_ylim(max(-120, ymax_db - 80), min(0, ymax_db + 5))
        plt.pause(0.001)


def run(port: str, baud: int, outdir: str, live: bool = False, fmax: float = None):
    # create session subfolder based on start time
    session_name = time.strftime('%Y%m%d_%H%M%S')
    session_dir = os.path.join(outdir, session_name)
    os.makedirs(session_dir, exist_ok=True)
    csv_path = os.path.join(session_dir, 'features.csv')
    ensure_csv(csv_path)

    ser = serial.Serial(port, baud, timeout=1)
    buf = b''
    print(f'Listening on {port} @ {baud} ...')
    print(f'Saving session outputs to: {session_dir}')
    live_plot = LiveFFT(fmax=fmax) if live else None
    try:
        while True:
            chunk = ser.read(4096)
            if not chunk:
                continue
            buf += chunk

            while True:
                # find earliest tag among EVT0 and FRM0
                i_evt = buf.find(b'EVT0')
                i_frm = buf.find(b'FRM0')
                cand = [i for i in [i_evt, i_frm] if i >= 0]
                if not cand:
                    buf = buf[-8:]
                    break
                idx = min(cand)
                tag = buf[idx:idx+4]
                if idx > 0:
                    buf = buf[idx:]
                if len(buf) < 12:
                    break
                sr, count = struct.unpack_from('<II', buf, 4)
                need = 12 + count * 2
                if len(buf) < need:
                    break
                data = buf[12:need]
                buf = buf[need:]

                if tag == b'EVT0':
                    x = np.frombuffer(data, dtype=np.int16)
                    ts = time.strftime('%Y%m%d_%H%M%S')
                    wav_name = f'evt_{ts}_{len(x)}.wav'
                wav_path = os.path.join(session_dir, wav_name)
                save_wav(wav_path, x, sr)
                    feat = extract_features(x, sr)
                with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([ts, wav_name, sr, len(x)] + list(feat))
                    print(
                        f'Event {wav_name} | peak={feat[0]:.3f} rms={feat[1]:.3f} '
                        f'dur={feat[4]:.1f}ms cent={feat[5]:.0f}Hz dom={feat[8]:.0f}Hz'
                    )
                    if live_plot:
                        live_plot.update(x, sr, fmax=fmax)
                elif tag == b'FRM0':
                    # streaming frame for live visualization only
                    if live_plot:
                        frame = np.frombuffer(data, dtype=np.int16)
                        live_plot.update_stream(frame, sr)
    finally:
        ser.close()


def main():
    p = argparse.ArgumentParser(description='Receive ESP32 impact events and analyze features')
    p.add_argument('-p', '--port', required=True, help='Serial port (e.g., COM5)')
    p.add_argument('-b', '--baud', type=int, default=921600, help='Baud rate (default: 921600)')
    p.add_argument('-o', '--outdir', default='events', help='Output directory')
    p.add_argument('--live', action='store_true', help='Show live waveform + FFT updated per event')
    p.add_argument('--fmax', type=float, default=8000.0, help='Max frequency for plot axis')
    args = p.parse_args()
    run(args.port, args.baud, args.outdir, live=args.live, fmax=args.fmax)


if __name__ == '__main__':
    main()
