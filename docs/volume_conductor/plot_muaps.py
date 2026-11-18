# The figure of docs/volume_conductor.md: python docs/volume_conductor/plot_muaps.py -> docs/volume_conductor/muaps.png
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from myogen.params import VolumeConductorParams
from myogen.simulation.volume_conductor import MUAPTable

OUT = Path(__file__).with_name("muaps.png")
SKIN_RADIUS_M, FAT_M, FS = 0.036, 0.005, 2000.0     # the average subject's forearm under a band


def main() -> None:
    table = MUAPTable(VolumeConductorParams(), SKIN_RADIUS_M, FAT_M, FS)
    t = (np.arange(table.basis.shape[1]) - table.basis.shape[1] // 2) / FS * 1000

    def wave(depth, arc_mm, dz):
        theta = arc_mm / 1000 / table.muscle_radius
        return table.weights_at(np.array(depth), np.array(theta), np.array(dz)).ravel() @ table.basis

    fig, axs = plt.subplots(1, 4, figsize=(18, 3.8))
    for depth in (0.001, 0.004, 0.01, 0.02):
        w = wave(depth, 0, 0.02)
        axs[0].plot(t, w / np.abs(w).max(), label=f"{depth * 1000:.0f} mm deep")
    for dz in (0.0, 0.01, 0.02, 0.03, 0.04):
        axs[1].plot(t, wave(0.004, 0, dz) * 1e9, label=f"{dz * 1000:.0f} mm from the end-plate")
        axs[2].plot(t, (wave(0.004, 0, dz) - wave(0.004, 0, dz + 0.01)) * 1e9, label=f"centred {dz * 1000 + 5:.0f} mm")
    arcs = np.arange(0, 41, 2)
    for depth, style in ((0.002, "-"), (0.01, "--"), (0.02, ":")):
        for bipolar, color in ((False, "#5f5e5a"), (True, "#d1495b")):
            signal = [wave(depth, a, 0.02) - (wave(depth, a, 0.03) if bipolar else 0) for a in arcs]
            rms = [np.sqrt(np.mean(x ** 2)) for x in signal]
            axs[3].semilogy(arcs, np.array(rms) / rms[0], style, color=color,
                            label=f"{'bipolar' if bipolar else 'monopolar'}, {depth * 1000:.0f} mm deep")
    titles = ("monopolar, 20 mm from the end-plate (peak 1)", "monopolar, 4 mm deep (nV per unit source)",
              "bipolar 10 mm along the fibres, 4 mm deep", "RMS around the forearm (relative to over the unit)")
    for ax, title in zip(axs, titles):
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlabel("ms after the firing" if ax is not axs[3] else "arc from the unit (mm)")
        if ax is not axs[3]:
            ax.set_xlim(-2, 16)
    fig.tight_layout()
    fig.savefig(OUT, dpi=110)


if __name__ == "__main__":
    main()
