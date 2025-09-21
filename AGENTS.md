# Repository Guidelines

## Project Structure & Module Organization
- `impact_analysis/impact_monitor.py`: Tkinter GUI for live capture, plotting (waveform/spectrum/spectrogram), and file analysis.
- `impact_analysis/receive_events.py`: CLI receiver that saves events and (optionally) shows live plots.
- `impact_analysis/visualize_event.py`: Offline visualizer for a single WAV (waveform + spectrogram).
- `impact_analysis/requirements.txt`: Python dependencies (NumPy, Matplotlib, PySerial, etc.).
- `ImpactCapture/ImpactCapture.ino`: ESP32 firmware that emits EVT0/FRM0 frames.
- `events/`: Session outputs (e.g., `YYYYMMDD_HHMMSS/evt_*.wav`, `features.csv`).

## Build, Test, and Development Commands
```bash
# Setup (Python 3.10+ recommended)
python -m venv .venv && . .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r impact_analysis/requirements.txt

# Run GUI monitor
python impact_analysis/impact_monitor.py

# Headless capture (example on Windows COM5)
python impact_analysis/receive_events.py -p COM5 -o events --live --fmax 8000

# Visualize a saved WAV
python impact_analysis/visualize_event.py events/20250101_120000/evt_*.wav --fmax 8000
```

## Coding Style & Naming Conventions
- Python: PEP 8, 4-space indents, type hints where practical; functions/modules use `snake_case`, constants `UPPER_SNAKE_CASE`.
- UI: Prefer menu items for secondary actions (Help/Exit), concise labels, and consistent key bindings (F1/F11/Ctrl+Q).
- Plots: Include axis labels/titles; keep spectrogram color scale in dB and update colorbar with data.

## Testing Guidelines
- No formal test suite. Verify manually:
  - Live: connect ESP32, run the GUI, confirm Start/Stop, plotting, and saved `events/…` outputs.
  - Offline: open File Analysis (or run `visualize_event.py`) and confirm WAV loads and plots render.
- If adding pure functions (e.g., feature extraction), prefer unit tests (e.g., `pytest`) with small fixtures; keep samples out of version control if large.

## Commit & Pull Request Guidelines
- Commits: short, imperative subject (≤72 chars) + concise body explaining why; reference issues when applicable.
- PRs: focused scope, clear description, steps to reproduce/verify, and screenshots/GIFs for UI changes. Note serial port and platform used when relevant.
- Do not commit generated data under `events/` or large binaries.

## Security & Configuration Tips
- Serial access: verify the correct port (e.g., `COM5` on Windows, `/dev/tty.usbserial*` on macOS).
- Data hygiene: sessions write to `events/YYYYMMDD_HHMMSS/`; prune old runs to save space.
- Firmware: open `ImpactCapture/ImpactCapture.ino` in Arduino IDE, select the proper ESP32 board, and upload.

