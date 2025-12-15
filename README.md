# Timelapse Movie Maker

A GUI application for creating timelapse movies from calcium imaging data.

## Quick Start

```bash
python3 movie_maker_gui.py
```

## Usage

1. **Select Output Directory**: Click "Browse..." to choose where movies will be saved.

2. **Add Folders**: Drag and drop acquisition folders into the Baseline and Response areas.
   - **Auto Detection**: When enabled (default), dropping a `before_*` or `baseline_*` folder will automatically find and add the matching `after_*` or `response_*` folder from the same directory.

3. **Adjust Contrast**: Click on a folder pair to select it, then adjust per-channel contrast using the sliders. Click "Auto" for automatic contrast or "Auto Contrast All Channels" to auto-adjust all channels.

4. **Preview**: Use the timepoint sliders to preview different frames from baseline and response.

5. **Generate Movies**: Click "Generate Movies" to create MP4 files. Each pair produces one combined video with baseline frames followed by response frames.

## Folder Naming Convention

For auto-detection to work, folders should follow this pattern:
```
{prefix}_{name}_{date}_{time}
```

- Baseline prefixes: `before_`, `baseline_`
- Response prefixes: `after_`, `response_`

Example:
- `before_A1_2025-01-15_14-30-00` (baseline)
- `after_A1_2025-01-15_14-35-00` (response)

## Dependencies

See `INSTALL.md` for detailed installation instructions.

**Core dependencies:**
- Python 3.8+
- PyQt5
- numpy
- tifffile
- imageio, imageio-ffmpeg

**Optional (for range sliders):**
- superqt

**Quick install:**
```bash
pip install PyQt5 numpy tifffile imageio imageio-ffmpeg superqt
```
