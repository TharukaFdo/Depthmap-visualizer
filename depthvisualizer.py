#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
depthvisualizer.py

Visualize the R, G, B channels (and combined RGB) of an OpenEXR file locally,
using a file‑selector dialog if no filename is given on the CLI.
"""

import os
import argparse
import tkinter as tk
from tkinter import filedialog

import OpenEXR
import Imath
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize R/G/B channels + combined RGB of an EXR file"
    )
    p.add_argument(
        "filename",
        nargs="?",
        default=None,
        help="Path to the EXR file (if omitted, a file dialog will open)",
    )
    return p.parse_args()


def select_file_via_dialog():
    """Open a file‑selector dialog and return the chosen path (or None)."""
    root = tk.Tk()
    root.withdraw()  # hide the main window
    path = filedialog.askopenfilename(
        title="Select an OpenEXR file",
        filetypes=[("OpenEXR files", "*.exr"), ("All files", "*.*")]
    )
    root.destroy()
    return path or None


def load_exr(filename):
    """Load R, G, B channels from an EXR into numpy arrays."""
    exr = OpenEXR.InputFile(filename)
    dw = exr.header()["dataWindow"]
    W = dw.max.x - dw.min.x + 1
    H = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    channels = {}
    for c in ("R", "G", "B"):
        buf = exr.channel(c, pt)
        arr = np.frombuffer(buf, dtype=np.float32)
        channels[c] = arr.reshape(H, W)

    return channels


def normalize_channel(arr):
    """Scale a float32 array to [0,1] for display."""
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-8)


def visualize(channels):
    """Plot each channel with a colorbar, plus the combined RGB image."""
    normed = {c: normalize_channel(channels[c]) for c in channels}
    rgb = np.stack([normed[c] for c in ("R", "G", "B")], axis=-1)

    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    for i, c in enumerate(("R", "G", "B")):
        im = axs[i].imshow(channels[c], cmap="plasma")
        axs[i].set_title(f"{c} Channel")
        axs[i].axis("off")
        plt.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)

    axs[3].imshow(rgb)
    axs[3].set_title("Combined RGB")
    axs[3].axis("off")

    plt.suptitle("EXR Channels with Plasma Colormap + Combined RGB", fontsize=18)
    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()
    filename = args.filename

    if not filename:
        filename = select_file_via_dialog()
        if not filename:
            print("No file selected. Exiting.")
            return

    if not os.path.isfile(filename):
        print(f"Error: file not found -> {filename}")
        return

    channels = load_exr(filename)
    visualize(channels)


if __name__ == "__main__":
    main()
