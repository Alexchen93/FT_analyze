import argparse, wave
import numpy as np
import matplotlib.pyplot as plt


def load_wav(path):
    with wave.open(path, 'rb') as w:
        fs = w.getframerate()
        n = w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    return fs, x


def plot_waveform(fs, x, title):
    t = np.arange(len(x)) / fs
    plt.figure(figsize=(10, 3))
    plt.plot(t, x, lw=0.8)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.tight_layout()


def plot_spectrogram(fs, x, fmax=None):
    nfft = 2048
    hop = nfft // 4
    win = np.hanning(nfft)
    if len(x) < nfft:
        pad = np.zeros(nfft - len(x), dtype=x.dtype)
        x = np.concatenate([x, pad])
    S = []
    for i in range(0, len(x) - nfft, hop):
        X = np.fft.rfft(win * x[i:i + nfft])
        S.append(20 * np.log10(np.abs(X) + 1e-12))
    if not S:
        return
    S = np.array(S).T
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    times = np.arange(S.shape[1]) * hop / fs

    plt.figure(figsize=(10, 4))
    plt.pcolormesh(times, freqs, S, shading='auto', cmap='magma')
    if fmax:
        plt.ylim(0, fmax)
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar(label='dB')
    plt.title('Spectrogram')
    plt.tight_layout()


def main():
    ap = argparse.ArgumentParser(description='Visualize impact event WAV (waveform + spectrogram)')
    ap.add_argument('wav', help='Path to WAV file')
    ap.add_argument('--fmax', type=float, default=8000, help='Max frequency to show')
    args = ap.parse_args()

    fs, x = load_wav(args.wav)
    plot_waveform(fs, x, f'Waveform ({args.wav})')
    plot_spectrogram(fs, x, args.fmax)
    plt.show()


if __name__ == '__main__':
    main()

