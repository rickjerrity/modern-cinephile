# Modern Cinephile Movie Player (MCMP)

Never let a good movie get in the way of your doomscrolling...

## Configuration

To get up and running quick, create a `videos` directory in the root of this project and drop some `.mp4` or `.mkv` files in there. Then follow the [Installation](#installation) instructions for your system below.

### Customization

The following constants are located in the `Configuration` section of the `main.py` file and can be modified to customize the video player.

* *`DEBUG_MODE` (default: `True`)* - When debug mode is enabled, the first face detected is drawn on a small pop-up window with lines extending from the head indicating pitch, yaw, and roll.
  * *`DEBUG_MS_INTERVAL` (default: `500`)* - Debug mode can be quite taxing on performance on a Raspberry Pi device, so we display a debug image every 500ms by default. You can lower this value for a more responsive debug mode experience
* *`VIDEO_PATH_DIR` (default: `./videos`)* - Directory containing video files to be played in a looped playlist
* *`VIDEO_FILE_EXTENSIONS` (default: `".mp4", ".mkv"`)* - File extensions to search for in the `VIDEO_PATH_DIR` directory

## Installation

This project is intended to run on a Raspberry Pi (specifically Raspberry Pi 5) using a Pi Camera Module first and foremost, so `picamera2` is listed as a dependency. This project has been tested on Windows 11 as well following the x86 instructions and using a standard USB webcam. Follow the installation instructions for your architecture below.

### Setup on a Raspberry Pi 5

This project was run on a Raspberry Pi 5 with 8GB RAM using the Raspberry Pi Camera Module 3 Wide Angle Lens as the camera.

The operating system is `Raspberry Pi OS (Legacy, 64-bit) (Bookworm)` on `Kernel version 6.12`. This distro and version seems to have the best package support for mediapipe and related picamera2 dependencies from my limited testing.

First, update packages on the Raspberry Pi:

`sudo apt update && sudo apt upgrade -y`

Next, install the necessary libraries to use `picamera2`:

`sudo apt install libc6-dev libcap-dev`

Install the `uv` Python package/project manager ([uv installation](https://docs.astral.sh/uv/getting-started/installation/)):

```
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

Configure your uv `venv` to access system packages as well, so you can access the `libcamera` module:

`uv venv --system-site-packages`

Install the dependencies and run the script:

```
uv pip install -r pyproject.toml
uv run main.py
```

### Setup on x86

In order to make installation and running easier for non-RPi devices, the `picamera2` dependency was added to the standard `dev` group, which makes it easy to specify for exclusion when installing packages and running the project.

Install the `uv` Python package/project manager ([uv installation](https://docs.astral.sh/uv/getting-started/installation/)):

```
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

Install dependencies, excluding the `dev` group:

`uv sync --no-dev`

Now, run the script, again excluding the `dev` group:

`uv run --no-dev ./main.py`