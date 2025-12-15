# Movie Maker Installation Guide

This document provides installation instructions for the Movie Maker application.

## Dependencies

### Core Dependencies (Required)
- Python 3.8+
- PyQt5 - GUI framework
- numpy - Array operations
- tifffile - TIFF image I/O
- imageio - Video encoding
- imageio-ffmpeg - FFmpeg backend for video encoding

### Optional Dependencies
- superqt - Range sliders for contrast control
- Pillow (PIL) - Image processing for timestamp overlays

---

## Quick Install

```bash
pip install PyQt5 numpy tifffile imageio imageio-ffmpeg superqt pillow
```

---

## Installation with Conda

```bash
# Create a new conda environment
conda create -n movie_maker python=3.10 -y
conda activate movie_maker

# Install dependencies
conda install -c conda-forge pyqt numpy tifffile imageio imageio-ffmpeg ffmpeg pillow -y
pip install superqt
```

---

## Installation with pip

### System Requirements

Before installing with pip, ensure you have:
- Python 3.8 or higher
- pip (latest version recommended)

### Pip Installation

```bash
# Create a virtual environment (recommended)
python -m venv movie_maker_env
source movie_maker_env/bin/activate  # Linux/macOS
# or
movie_maker_env\Scripts\activate  # Windows

# Install dependencies
pip install PyQt5 numpy tifffile imageio imageio-ffmpeg superqt pillow
```

---

## Troubleshooting

### PyQt5 Import Error on Linux
If you get `ImportError: libGL.so.1: cannot open shared object file`:
```bash
# Ubuntu/Debian
sudo apt install libgl1-mesa-glx libegl1-mesa
```

### FFmpeg Not Found
If video encoding fails with "ffmpeg not found":
```python
import imageio_ffmpeg
imageio_ffmpeg.get_ffmpeg_exe()  # Downloads ffmpeg if needed
```

### Qt Platform Plugin Error
If you get "could not find or load the Qt platform plugin":
```bash
# Ubuntu/Debian
sudo apt install libxcb-xinerama0 libxcb-cursor0
```

---

## Running the Application

```bash
python movie_maker_gui.py
```

---

## File Structure

```
movie_test/
├── movie_maker_backend.py    # Backend for movie generation
├── movie_maker_gui.py        # Main GUI application
├── INSTALL.md                # This file
└── README.md                 # User instructions
```
