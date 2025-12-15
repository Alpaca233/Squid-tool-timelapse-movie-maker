"""
Backend for creating timelapse movies from calcium imaging data.

Features:
- Reads acquisition folders with timepoint subfolders
- Extracts dt(s) and Nt from acquisition parameters.json
- Handles multiple wavelengths with per-channel contrast and colormaps
- Generates MP4 movies with max intensity projection across channels
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np

# Image I/O
try:
    import tifffile as tf
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False

# Video encoding
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

# Optional: imageio-ffmpeg for better codec support
try:
    import imageio_ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False


# Filename pattern for single TIFF files
FPATTERN = re.compile(r"(?P<r>[^_]+)_(?P<f>\d+)_(?P<z>\d+)_(?P<c>.+)\.tiff?", re.IGNORECASE)


@dataclass
class AcquisitionParams:
    """Acquisition parameters from JSON."""
    dt_seconds: float = 1.0
    nt: int = 1
    dz_um: float = 0.0
    nz: int = 1
    objective_name: str = ""
    magnification: float = 1.0

    @classmethod
    def from_json(cls, json_path: str) -> "AcquisitionParams":
        """Load acquisition parameters from JSON file."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            params = cls()
            params.dt_seconds = data.get("dt(s)", 1.0)
            params.nt = data.get("Nt", 1)
            params.dz_um = data.get("dz(um)", 0.0)
            params.nz = data.get("Nz", 1)

            obj = data.get("objective", {})
            params.objective_name = obj.get("name", "")
            params.magnification = obj.get("magnification", 1.0)

            return params
        except Exception as e:
            print(f"Error reading acquisition parameters: {e}")
            return cls()


@dataclass
class ChannelSettings:
    """Settings for a single wavelength channel."""
    name: str
    wavelength: Optional[int] = None
    contrast_min: float = 0.0
    contrast_max: float = 65535.0
    colormap: str = "gray"
    enabled: bool = True


@dataclass
class FolderPairSettings:
    """Settings for a baseline/response folder pair."""
    baseline_path: str
    response_path: str
    channels: Dict[str, ChannelSettings] = field(default_factory=dict)
    frame_interval_seconds: float = 1.0
    output_fps: int = 10

    def get_channel_names(self) -> List[str]:
        """Get sorted list of channel names."""
        return sorted(self.channels.keys())


def extract_wavelength(channel_str: str) -> Optional[int]:
    """Extract wavelength (nm) from channel string."""
    if not channel_str:
        return None

    lower = channel_str.lower()
    if re.fullmatch(r'ch\d+', lower):
        return None

    # Direct wavelength pattern
    if m := re.search(r'(\d{3,4})[ _]*nm', channel_str, re.IGNORECASE):
        return int(m.group(1))

    # Common fluorophores
    fluor_map = {
        'dapi': 405, 'hoechst': 405,
        'gfp': 488, 'fitc': 488, 'alexa488': 488,
        'tritc': 561, 'cy3': 561, 'mcherry': 561,
        'cy5': 640, 'alexa647': 640, 'cy7': 730
    }
    for fluor, wl in fluor_map.items():
        if fluor in lower:
            return wl

    # Fallback to last 3-4 digit number
    numbers = re.findall(r'\d{3,4}', channel_str)
    if numbers:
        val = int(numbers[-1])
        return val if val > 0 else None
    return None


def wavelength_to_colormap(wavelength: Optional[int]) -> str:
    """Map wavelength to colormap name."""
    if wavelength is None or wavelength == 0:
        return 'gray'
    if wavelength <= 420:
        return 'blue'
    elif 470 <= wavelength <= 510:
        return 'green'
    elif 540 <= wavelength <= 590:
        return 'yellow'
    elif 620 <= wavelength <= 660:
        return 'red'
    elif wavelength >= 700:
        return 'magenta'
    return 'gray'


def wavelength_to_rgb(wavelength: Optional[int]) -> Tuple[int, int, int]:
    """Convert wavelength to RGB color tuple."""
    if wavelength is None or wavelength == 0:
        return (255, 255, 255)  # White for grayscale
    if wavelength <= 420:
        return (0, 0, 255)  # Blue
    elif 470 <= wavelength <= 510:
        return (0, 255, 0)  # Green
    elif 540 <= wavelength <= 590:
        return (255, 255, 0)  # Yellow
    elif 620 <= wavelength <= 660:
        return (255, 0, 0)  # Red
    elif wavelength >= 700:
        return (255, 0, 255)  # Magenta
    return (255, 255, 255)  # White


class AcquisitionFolder:
    """Represents a single acquisition folder with timepoint subfolders."""

    def __init__(self, folder_path: str):
        self.path = Path(folder_path)
        self.params = self._load_params()
        self.timepoints = self._discover_timepoints()
        self.channels = self._discover_channels()

    def _load_params(self) -> AcquisitionParams:
        """Load acquisition parameters from JSON."""
        json_path = self.path / "acquisition parameters.json"
        if json_path.exists():
            return AcquisitionParams.from_json(str(json_path))
        return AcquisitionParams()

    def _discover_timepoints(self) -> List[int]:
        """Find all numeric timepoint directories."""
        timepoints = []
        for item in self.path.iterdir():
            if item.is_dir() and item.name.isdigit():
                timepoints.append(int(item.name))
        return sorted(timepoints)

    def _discover_channels(self) -> List[str]:
        """Discover unique channel names from first timepoint."""
        channels = set()
        if not self.timepoints:
            return []

        tp_dir = self.path / str(self.timepoints[0])
        for f in tp_dir.iterdir():
            if f.suffix.lower() in ['.tif', '.tiff']:
                if m := FPATTERN.search(f.name):
                    channels.add(m.group('c'))
        return sorted(channels)

    def get_image_path(self, timepoint: int, channel: str, z: int = 0, region: str = "current", fov: int = 0) -> Optional[Path]:
        """Get the path to a specific image file."""
        tp_dir = self.path / str(timepoint)
        if not tp_dir.exists():
            return None

        # Pattern: {region}_{fov}_{z}_{channel}.tiff
        filename = f"{region}_{fov}_{z}_{channel}.tiff"
        filepath = tp_dir / filename
        if filepath.exists():
            return filepath

        # Try .tif extension
        filepath = tp_dir / f"{region}_{fov}_{z}_{channel}.tif"
        if filepath.exists():
            return filepath

        return None

    def load_image(self, timepoint: int, channel: str, z: int = 0) -> Optional[np.ndarray]:
        """Load a single image."""
        if not TIFFFILE_AVAILABLE:
            raise ImportError("tifffile is required for image loading")

        path = self.get_image_path(timepoint, channel, z)
        if path is None:
            return None

        try:
            return tf.imread(str(path))
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    def load_timepoint_all_channels(self, timepoint: int) -> Dict[str, np.ndarray]:
        """Load all channels for a single timepoint."""
        images = {}
        for channel in self.channels:
            img = self.load_image(timepoint, channel)
            if img is not None:
                images[channel] = img
        return images


def discover_channels_for_pair(baseline_path: str, response_path: str) -> Dict[str, ChannelSettings]:
    """Discover channels from both baseline and response folders."""
    channels = {}

    for folder_path in [baseline_path, response_path]:
        folder = AcquisitionFolder(folder_path)
        for ch_name in folder.channels:
            if ch_name not in channels:
                wl = extract_wavelength(ch_name)
                channels[ch_name] = ChannelSettings(
                    name=ch_name,
                    wavelength=wl,
                    colormap=wavelength_to_colormap(wl),
                    contrast_min=0.0,
                    contrast_max=65535.0,
                    enabled=True
                )

    return channels


def compute_auto_contrast(folder: AcquisitionFolder, channel: str,
                         percentile_low: float = 1.0,
                         percentile_high: float = 99.5) -> Tuple[float, float]:
    """Compute auto-contrast limits for a channel across all timepoints."""
    all_pixels = []

    # Sample from a few timepoints for speed
    sample_tps = folder.timepoints[::max(1, len(folder.timepoints)//5)][:5]

    for tp in sample_tps:
        img = folder.load_image(tp, channel)
        if img is not None:
            # Subsample for speed
            all_pixels.append(img[::4, ::4].ravel())

    if not all_pixels:
        return 0.0, 65535.0

    combined = np.concatenate(all_pixels)
    lo = np.percentile(combined, percentile_low)
    hi = np.percentile(combined, percentile_high)

    return float(lo), float(hi)


def compute_auto_contrast_for_pair(baseline_path: str, response_path: str,
                                   channel: str) -> Tuple[float, float]:
    """Compute auto-contrast that works for both baseline and response."""
    all_pixels = []

    for folder_path in [baseline_path, response_path]:
        folder = AcquisitionFolder(folder_path)
        sample_tps = folder.timepoints[::max(1, len(folder.timepoints)//3)][:3]

        for tp in sample_tps:
            img = folder.load_image(tp, channel)
            if img is not None:
                all_pixels.append(img[::4, ::4].ravel())

    if not all_pixels:
        return 0.0, 65535.0

    combined = np.concatenate(all_pixels)
    lo = np.percentile(combined, 1.0)
    hi = np.percentile(combined, 99.5)

    return float(lo), float(hi)


def apply_contrast(image: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Apply contrast limits to an image, returning uint8."""
    if vmax <= vmin:
        vmax = vmin + 1

    img_float = (image.astype(np.float32) - vmin) / (vmax - vmin)
    img_float = np.clip(img_float, 0, 1)
    return (img_float * 255).astype(np.uint8)


def colorize_channel(image_uint8: np.ndarray, rgb: Tuple[int, int, int]) -> np.ndarray:
    """Colorize a grayscale image with an RGB color."""
    # image_uint8: HxW uint8
    # Returns: HxWx3 uint8
    h, w = image_uint8.shape
    colored = np.zeros((h, w, 3), dtype=np.float32)

    for i, c in enumerate(rgb):
        colored[:, :, i] = image_uint8.astype(np.float32) * (c / 255.0)

    return colored.astype(np.uint8)


def composite_channels(images: Dict[str, np.ndarray],
                      channel_settings: Dict[str, ChannelSettings]) -> np.ndarray:
    """Composite multiple channels into a single RGB image."""
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    # Get image dimensions from first image
    first_img = next(iter(images.values()))
    h, w = first_img.shape[:2]

    # Accumulate in float for proper blending
    composite = np.zeros((h, w, 3), dtype=np.float32)

    for ch_name, img in images.items():
        if ch_name not in channel_settings:
            continue

        settings = channel_settings[ch_name]
        if not settings.enabled:
            continue

        # Apply contrast
        img_contrast = apply_contrast(img, settings.contrast_min, settings.contrast_max)

        # Get color for this channel
        rgb = wavelength_to_rgb(settings.wavelength)

        # Colorize and add to composite
        colored = colorize_channel(img_contrast, rgb)
        composite += colored.astype(np.float32)

    # Clip and convert to uint8
    composite = np.clip(composite, 0, 255).astype(np.uint8)
    return composite


def create_composite_frame(folder: AcquisitionFolder, timepoint: int,
                          channel_settings: Dict[str, ChannelSettings]) -> np.ndarray:
    """Create a composited frame for a single timepoint."""
    images = folder.load_timepoint_all_channels(timepoint)
    return composite_channels(images, channel_settings)


def create_movie(folder: AcquisitionFolder,
                channel_settings: Dict[str, ChannelSettings],
                output_path: str,
                fps: int = 10,
                add_timestamp: bool = True,
                dt_seconds: float = 1.0) -> bool:
    """Create a movie from an acquisition folder."""
    if not IMAGEIO_AVAILABLE:
        raise ImportError("imageio is required for movie creation")

    if not folder.timepoints:
        print(f"No timepoints found in {folder.path}")
        return False

    try:
        # Determine output format
        output_path_obj = Path(output_path)

        # Use ffmpeg writer for better quality
        if FFMPEG_AVAILABLE:
            writer = imageio.get_writer(
                output_path,
                fps=fps,
                codec='libx264',
                quality=8,  # Higher = better quality
                pixelformat='yuv420p'
            )
        else:
            writer = imageio.get_writer(output_path, fps=fps)

        for tp in folder.timepoints:
            frame = create_composite_frame(folder, tp, channel_settings)

            # Add timestamp overlay if requested
            if add_timestamp:
                frame = add_timestamp_overlay(frame, tp, dt_seconds)

            writer.append_data(frame)

        writer.close()
        return True

    except Exception as e:
        print(f"Error creating movie: {e}")
        import traceback
        traceback.print_exc()
        return False


def add_timestamp_overlay(frame: np.ndarray, timepoint: int, dt_seconds: float, label: str = "") -> np.ndarray:
    """Add timestamp text to frame with optional label (e.g., 'Baseline' or 'Response')."""
    try:
        from PIL import Image, ImageDraw, ImageFont

        # Convert to PIL
        pil_img = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_img)

        # Calculate time
        time_sec = timepoint * dt_seconds
        if time_sec < 60:
            time_str = f"{time_sec:.1f}s"
        else:
            minutes = int(time_sec // 60)
            seconds = time_sec % 60
            time_str = f"{minutes}m {seconds:.1f}s"

        # Add label if provided
        if label:
            time_str = f"{label} - {time_str}"

        # Try to use a reasonable font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
            except:
                font = ImageFont.load_default()

        # Draw text with shadow for visibility
        text_pos = (10, 10)
        draw.text((text_pos[0]+1, text_pos[1]+1), time_str, fill=(0, 0, 0), font=font)
        draw.text(text_pos, time_str, fill=(255, 255, 255), font=font)

        return np.array(pil_img)
    except ImportError:
        # PIL not available, return frame as-is
        return frame


def generate_movie_pair(settings: FolderPairSettings,
                       output_dir: str,
                       output_name: str,
                       add_timestamp: bool = True,
                       progress_callback=None) -> bool:
    """Generate a single combined movie with baseline frames first, then response frames."""
    if not IMAGEIO_AVAILABLE:
        raise ImportError("imageio is required for movie creation")

    baseline_folder = AcquisitionFolder(settings.baseline_path)
    response_folder = AcquisitionFolder(settings.response_path)

    if not baseline_folder.timepoints and not response_folder.timepoints:
        print("No timepoints found in either folder")
        return False

    fps = settings.output_fps
    dt = settings.frame_interval_seconds

    output_path = os.path.join(output_dir, f"{output_name}.mp4")

    try:
        # Use ffmpeg writer for better quality
        if FFMPEG_AVAILABLE:
            writer = imageio.get_writer(
                output_path,
                fps=fps,
                codec='libx264',
                quality=8,
                pixelformat='yuv420p'
            )
        else:
            writer = imageio.get_writer(output_path, fps=fps)

        total_frames = len(baseline_folder.timepoints) + len(response_folder.timepoints)
        current_frame = 0

        # Write baseline frames first
        if progress_callback:
            progress_callback("Creating movie (baseline frames)...")
        for tp in baseline_folder.timepoints:
            frame = create_composite_frame(baseline_folder, tp, settings.channels)
            if add_timestamp:
                frame = add_timestamp_overlay(frame, tp, dt, label="Baseline")
            writer.append_data(frame)
            current_frame += 1
            if progress_callback and total_frames > 0:
                progress_callback(f"Creating movie... {current_frame}/{total_frames}")

        # Write response frames
        if progress_callback:
            progress_callback("Creating movie (response frames)...")
        for tp in response_folder.timepoints:
            frame = create_composite_frame(response_folder, tp, settings.channels)
            if add_timestamp:
                frame = add_timestamp_overlay(frame, tp, dt, label="Response")
            writer.append_data(frame)
            current_frame += 1
            if progress_callback and total_frames > 0:
                progress_callback(f"Creating movie... {current_frame}/{total_frames}")

        writer.close()
        return True

    except Exception as e:
        print(f"Error creating movie: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_preview_frame(folder_path: str, channel_settings: Dict[str, ChannelSettings],
                       timepoint: int = 0) -> Optional[np.ndarray]:
    """Load a single frame for preview."""
    folder = AcquisitionFolder(folder_path)
    if not folder.timepoints:
        return None

    tp = timepoint if timepoint in folder.timepoints else folder.timepoints[0]
    return create_composite_frame(folder, tp, channel_settings)


def get_max_intensity_projection(folder_path: str, channel: str,
                                max_timepoints: int = 10) -> Optional[np.ndarray]:
    """Compute max intensity projection across timepoints for a channel."""
    folder = AcquisitionFolder(folder_path)
    if not folder.timepoints:
        return None

    # Sample timepoints evenly
    step = max(1, len(folder.timepoints) // max_timepoints)
    sample_tps = folder.timepoints[::step]

    mip = None
    for tp in sample_tps:
        img = folder.load_image(tp, channel)
        if img is not None:
            if mip is None:
                mip = img.astype(np.float32)
            else:
                mip = np.maximum(mip, img.astype(np.float32))

    return mip.astype(np.uint16) if mip is not None else None


def get_channel_histogram(folder_path: str, channel: str,
                         nbins: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Compute histogram for a channel (sampling across timepoints)."""
    folder = AcquisitionFolder(folder_path)
    if not folder.timepoints:
        return np.zeros(nbins), np.linspace(0, 65535, nbins+1)

    all_pixels = []
    sample_tps = folder.timepoints[::max(1, len(folder.timepoints)//5)][:5]

    for tp in sample_tps:
        img = folder.load_image(tp, channel)
        if img is not None:
            all_pixels.append(img[::4, ::4].ravel())

    if not all_pixels:
        return np.zeros(nbins), np.linspace(0, 65535, nbins+1)

    combined = np.concatenate(all_pixels)
    hist, bin_edges = np.histogram(combined, bins=nbins, range=(0, 65535))
    return hist, bin_edges


# Compatibility function for old GUI
def auto_contrast(folder_path: str, percentile_low: float = 1.0,
                  percentile_high: float = 99.5) -> Tuple[float, float]:
    """Compute auto-contrast limits (compatibility function)."""
    folder = AcquisitionFolder(folder_path)
    if not folder.channels:
        return 0.0, 65535.0

    all_pixels = []
    sample_tps = folder.timepoints[::max(1, len(folder.timepoints)//5)][:5]

    for tp in sample_tps:
        for channel in folder.channels:
            img = folder.load_image(tp, channel)
            if img is not None:
                all_pixels.append(img[::4, ::4].ravel())

    if not all_pixels:
        return 0.0, 65535.0

    combined = np.concatenate(all_pixels)
    lo = np.percentile(combined, percentile_low)
    hi = np.percentile(combined, percentile_high)

    return float(lo), float(hi)


# Compatibility function for old GUI
def create_timelapse_movie(baseline_path: str, response_path: str,
                          name: str, output_dir: str,
                          contrast_lo: float = 0, contrast_hi: float = 65535) -> bool:
    """Create timelapse movie (compatibility function for old GUI)."""
    # Discover channels
    channels = discover_channels_for_pair(baseline_path, response_path)

    # Apply same contrast to all channels
    for ch_name in channels:
        channels[ch_name].contrast_min = contrast_lo
        channels[ch_name].contrast_max = contrast_hi

    # Create settings
    baseline_folder = AcquisitionFolder(baseline_path)
    settings = FolderPairSettings(
        baseline_path=baseline_path,
        response_path=response_path,
        channels=channels,
        frame_interval_seconds=baseline_folder.params.dt_seconds,
        output_fps=10
    )

    return generate_movie_pair(settings, output_dir, name)


if __name__ == "__main__":
    # Test the backend
    import sys

    if len(sys.argv) < 3:
        print("Usage: python movie_maker_backend.py <baseline_folder> <response_folder>")
        sys.exit(1)

    baseline = sys.argv[1]
    response = sys.argv[2]

    print(f"Baseline: {baseline}")
    print(f"Response: {response}")

    baseline_folder = AcquisitionFolder(baseline)
    print(f"  dt(s): {baseline_folder.params.dt_seconds}")
    print(f"  Nt: {baseline_folder.params.nt}")
    print(f"  Timepoints found: {len(baseline_folder.timepoints)}")
    print(f"  Channels: {baseline_folder.channels}")

    channels = discover_channels_for_pair(baseline, response)
    print(f"\nDiscovered channels:")
    for ch_name, ch_settings in channels.items():
        print(f"  {ch_name}: wavelength={ch_settings.wavelength}, colormap={ch_settings.colormap}")

        # Compute auto contrast
        lo, hi = compute_auto_contrast_for_pair(baseline, response, ch_name)
        print(f"    Auto contrast: {lo:.0f} - {hi:.0f}")
