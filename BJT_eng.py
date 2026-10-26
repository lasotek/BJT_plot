"""Four-quadrant characteristics: tabulated data and operating points."""

from itertools import pairwise

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator
from scipy.interpolate import PchipInterpolator  # type: ignore[import-untyped]

# PLOT SETTINGS — edit these values to customize the title and subtitle.
PLOT_TITLE = "Four-quadrant characteristics of an NPN transistor"
PLOT_SUBTITLE = "Example tabulated data: data points + PCHIP interpolation"

# Points excluded from interpolation. They are still plotted as measured data,
# using the same marker as the other points in their series.
# They do NOT take part in PCHIP interpolation or in operating-point calculation.
# Enter pairs (value from column 0, index of the column with the faulty value).
# Column indices are zero-based, as in Python:
#   TABLE_1: (I_SET, column), columns 2–7;
#     e.g. (30, 5) means I_C at U_CE=4 V in the row with I_SET=30.
#   TABLE_2: (U_CE, column), columns 1–6;
#     e.g. (0.5, 3) means U_BE at I_B=30 µA for U_CE=0.5 V.
GROSS_ERRORS_TABLE_1: set[tuple[float, int]] = set()
GROSS_ERRORS_TABLE_2: set[tuple[float, int]] = set()
# GROSS_ERRORS_TABLE_2 = {(9.0, 6)} - example of a faulty point
# Replace set() with the appropriate values
# if you want to mark faulty points in the tables.

# Example data; these are not measurements of a BCP54 transistor.
# Units: I_B [µA], I_C [mA], U_BE and U_CE [V].
# Columns: I_SET, I_B, U_BE(0.5), I_C(0.5), U_BE(4), I_C(4),
# U_BE(9), I_C(9).
TABLE_1 = np.array([
    [2, 2.0, .565, .21, .563, .23, .560, .25],
    [5, 5.0, .590, .54, .588, .59, .585, .63],
    [10, 10.0, .609, 1.09, .607, 1.19, .604, 1.27],
    [15, 15.0, .620, 1.65, .618, 1.80, .615, 1.92],
    [20, 20.0, .628, 2.21, .626, 2.41, .623, 2.57],
    [30, 30.0, .639, 3.35, .637, 3.65, .634, 3.90],
    [40, 40.0, .647, 4.51, .645, 4.91, .642, 5.24],
    [50, 50.0, .653, 5.68, .651, 6.18, .648, 6.60],
    [60, 60.0, .658, 6.86, .656, 7.46, .653, 7.96],
    [70, 70.0, .662, 8.05, .660, 8.75, .657, 9.34],
    [80, 80.0, .666, 9.25, .664, 10.05, .661, 10.73],
    [90, 90.0, .669, 10.46, .667, 11.36, .664, 12.13],
    [100, 100.0, .672, 11.68, .670, 12.68, .667, 13.54],
], dtype=float)

# Columns: U_CE, U_BE(15), I_C(15), U_BE(30), I_C(30),
# U_BE(60), I_C(60).
TABLE_2 = np.array([
    [0.00, .5700, .000, .5890, .000, .6080, .000],
    [0.05, .5880, .310, .6070, .620, .6260, 1.240],
    [0.10, .6020, .800, .6210, 1.620, .6400, 3.250],
    [0.15, .6120, 1.230, .6310, 2.490, .6500, 5.030],
    [0.20, .6170, 1.490, .6360, 3.020, .6550, 6.130],
    [0.30, .6200, 1.610, .6390, 3.270, .6580, 6.650],
    [0.40, .6201, 1.640, .6391, 3.330, .6581, 6.800],
    [0.50, .6200, 1.650, .6390, 3.350, .6580, 6.860],
    [0.60, .6199, 1.655, .6389, 3.360, .6579, 6.880],
    [0.90, .6196, 1.670, .6386, 3.390, .6576, 6.940],
    [1.00, .6195, 1.675, .6385, 3.400, .6575, 6.960],
    [2.00, .6190, 1.720, .6380, 3.490, .6570, 7.140],
    [3.00, .6185, 1.760, .6375, 3.570, .6565, 7.300],
    [4.00, .6180, 1.800, .6370, 3.650, .6560, 7.460],
    [5.00, .6174, 1.825, .6364, 3.700, .6554, 7.560],
    [6.00, .6168, 1.850, .6358, 3.750, .6548, 7.660],
    [7.00, .6162, 1.875, .6352, 3.800, .6542, 7.760],
    [8.00, .6156, 1.900, .6346, 3.850, .6536, 7.860],
    [9.00, .6150, 1.920, .6340, 3.900, .6530, 7.960],
    [10.00, .6144, 1.945, .6334, 3.950, .6524, 8.060],
], dtype=float)

TABLE_1_SERIES = [(0.5, 2, 3), (4.0, 4, 5), (9.0, 6, 7)]
TABLE_2_SERIES = [(15, 1, 2), (30, 3, 4), (60, 5, 6)]
COLORS = ["tab:blue", "tab:orange", "tab:green"]
MARKERS = ["o", "s", "^"]
OPERATING_POINTS = [(4.0, 15.0), (4.0, 30.0), (9.0, 15.0), (9.0, 30.0)]
RECTANGLE_STYLE = {
    "color": "0.40", "linewidth": 0.8,
    "linestyle": (0, (5, 4)), "alpha": 0.65, "zorder": 2,
}


def make_interpolator(x, y):
    """Validate the data and create an interpolator without extrapolation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("x and y must be one-dimensional arrays of equal length.")
    if len(x) < 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("At least two complete data points are required.")
    order = np.argsort(x)
    x, y = x[order], y[order]
    if np.any(np.diff(x) <= 0):
        raise ValueError(
            "Duplicate x coordinates: decide how duplicate values should be handled."
        )
    return x, y, PchipInterpolator(x, y, extrapolate=False)


def gross_error_mask(table, column, gross_errors, first_column):
    """Return a mask of the column rows marked as excluded from interpolation."""
    keys = table[:, 0]
    for key, col in gross_errors:
        if col < first_column or col >= table.shape[1] or not np.any(keys == key):
            raise ValueError(
                f"Invalid gross-error point (key={key}, column={col}): "
                f"allowed columns are {first_column}..{table.shape[1] - 1}, "
                "and the key must exist in column 0."
            )
    return np.array([(key, column) in gross_errors for key in keys])


def table_1_mask(column):
    return gross_error_mask(TABLE_1, column, GROSS_ERRORS_TABLE_1, 2)


def table_2_mask(column):
    return gross_error_mask(TABLE_2, column, GROSS_ERRORS_TABLE_2, 1)


def make_curve(x, y, is_gross_error):
    """Build an interpolator only from points without a gross error."""
    keep = ~is_gross_error
    return make_interpolator(np.asarray(x)[keep], np.asarray(y)[keep])[2]


def draw_characteristic(ax, x, y, color, marker, label, is_gross_error):
    """Plot all points with the same marker; filter only the interpolation."""
    x_all = np.asarray(x, dtype=float)
    y_all = np.asarray(y, dtype=float)
    keep = ~is_gross_error
    x, y, interpolation = make_interpolator(x_all[keep], y_all[keep])
    x_smooth = np.unique(np.concatenate([
        np.linspace(a, b, 30) for a, b in pairwise(x)
    ]))
    ax.plot(x_smooth, interpolation(x_smooth), color=color, linewidth=1.6, zorder=3)
    ax.scatter(
        x_all, y_all, s=30, marker=marker, facecolors="white", edgecolors=color,
        linewidths=1.2, label=label, zorder=5,
    )


def format_tick(value, _position):
    return "" if np.isclose(value, 0) else f"{value:g}"


def add_arrow(ax, start, end):
    ax.annotate(
        "", xy=end, xytext=start, xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.2,
                    "mutation_scale": 14},
        annotation_clip=False, zorder=10,
    )


def configure_axes(axes):
    """Configure grids, backgrounds, spines, ticks, and labels in one place."""
    for (row, col), ax in np.ndenumerate(axes):
        x_major, x_minor = (20, 5) if col == 0 else (2, 0.5)
        y_major, y_minor = (2, 0.5) if row == 0 else (0.1, 0.025)
        ax.xaxis.set_major_locator(MultipleLocator(x_major))
        ax.xaxis.set_minor_locator(MultipleLocator(x_minor))
        ax.yaxis.set_major_locator(MultipleLocator(y_major))
        ax.yaxis.set_minor_locator(MultipleLocator(y_minor))
        ax.xaxis.set_major_formatter(FuncFormatter(format_tick))
        ax.yaxis.set_major_formatter(FuncFormatter(format_tick))
        ax.grid(which="major", color="0.79", linewidth=0.6)
        ax.grid(which="minor", color="0.92", linewidth=0.4)
        ax.patch.set_visible(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(
            axis="both", which="both", bottom=False, top=False,
            left=False, right=False, labelbottom=False, labeltop=False,
            labelleft=False, labelright=False,
        )
        if row == 0:
            ax.spines["bottom"].set_visible(True)
            ax.spines["bottom"].set_linewidth(1.2)
            ax.tick_params(axis="x", which="both", bottom=True)
            ax.tick_params(axis="x", which="major", labelbottom=True)
        if col == 0:
            ax.spines["right"].set_visible(True)
            ax.spines["right"].set_linewidth(1.2)
            ax.yaxis.set_ticks_position("right")
            ax.tick_params(axis="y", which="both", right=True)
            ax.tick_params(axis="y", which="major", labelright=True)
        ax.tick_params(
            axis="both", which="major", direction="in", length=6,
            width=1.1, colors="black", labelsize=10, pad=5,
        )
        ax.tick_params(
            axis="both", which="minor", direction="in", length=3,
            width=0.8, colors="black",
        )


def draw_operating_rectangles(axes):
    """Connect the four representations of each operating point with rectangles.

    U_BE and I_C are interpolated from Table 1 at the selected U_CE and I_B.
    Table 2 is used as a consistency check; the source data are not modified.
    """
    ax_transfer, ax_output = axes[0]
    ax_input, ax_reverse = axes[1]
    for uce_q, ib_q in OPERATING_POINTS:
        _, col_ube, col_ic = next(
            series for series in TABLE_1_SERIES if np.isclose(series[0], uce_q)
        )
        ube_curve = make_curve(
            TABLE_1[:, 1], TABLE_1[:, col_ube], table_1_mask(col_ube),
        )
        ic_curve = make_curve(TABLE_1[:, 1], TABLE_1[:, col_ic], table_1_mask(col_ic))
        ube_q, ic_q = float(ube_curve(ib_q)), float(ic_curve(ib_q))
        if not np.all(np.isfinite([ube_q, ic_q])):
            raise ValueError("Operating point is outside the range of Table 1 data.")

        # Four corners: (I_B, I_C), (U_CE, I_C),
        # (I_B, U_BE), and (U_CE, U_BE). Each plot draws two sides.
        for ax, x_q, y_q in (
            (ax_transfer, ib_q, ic_q), (ax_output, uce_q, ic_q),
            (ax_input, ib_q, ube_q), (ax_reverse, uce_q, ube_q),
        ):
            ax.plot([0, x_q, x_q], [y_q, y_q, 0], **RECTANGLE_STYLE)

        _, col_ube_2, col_ic_2 = next(
            series for series in TABLE_2_SERIES if np.isclose(series[0], ib_q)
        )
        ube_curve_2 = make_curve(
            TABLE_2[:, 0], TABLE_2[:, col_ube_2], table_2_mask(col_ube_2),
        )
        ic_curve_2 = make_curve(
            TABLE_2[:, 0], TABLE_2[:, col_ic_2], table_2_mask(col_ic_2),
        )
        ube_q_2, ic_q_2 = float(ube_curve_2(uce_q)), float(ic_curve_2(uce_q))
        if not np.all(np.isfinite([ube_q_2, ic_q_2])):
            raise ValueError("Operating point is outside the range of Table 2 data.")
        if not (
            np.isclose(ube_q, ube_q_2, rtol=0, atol=1e-9)
            and np.isclose(ic_q, ic_q_2, rtol=0, atol=1e-9)
        ):
            print(
                f"Warning: U_CE={uce_q:g} V, I_B={ib_q:g} µA: "
                f"Table 1: U_BE={ube_q:g} V, I_C={ic_q:g} mA; "
                f"Table 2: U_BE={ube_q_2:g} V, I_C={ic_q_2:g} mA. "
                "The rectangle is based on Table 1; the source data were not modified."
            )


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.axisbelow": True, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 10), sharex="col", sharey="row",
        gridspec_kw={"wspace": 0, "hspace": 0},
    )
    ax_transfer, ax_output = axes[0]
    ax_input, ax_reverse = axes[1]
    fig.subplots_adjust(left=0.09, right=0.90, bottom=0.11, top=0.87)
    ax_transfer.set_xlim(105, 0)
    ax_output.set_xlim(0, 10.5)
    ax_transfer.set_ylim(0, 15)
    ax_input.set_ylim(0.75, 0)

    for (uce_parameter, col_ube, col_ic), color, marker in zip(
        TABLE_1_SERIES, COLORS, MARKERS, strict=True,
    ):
        label = rf"$U_{{CE}}={uce_parameter:g}\ \mathrm{{V}}$"
        draw_characteristic(
            ax_transfer, TABLE_1[:, 1], TABLE_1[:, col_ic], color, marker, label,
            table_1_mask(col_ic),
        )
        draw_characteristic(
            ax_input, TABLE_1[:, 1], TABLE_1[:, col_ube], color, marker, label,
            table_1_mask(col_ube),
        )
    for (ib_parameter, col_ube, col_ic), color, marker in zip(
        TABLE_2_SERIES, COLORS, MARKERS, strict=True,
    ):
        label = rf"$I_B={ib_parameter:g}\ \mu\mathrm{{A}}$"
        draw_characteristic(
            ax_output, TABLE_2[:, 0], TABLE_2[:, col_ic], color, marker, label,
            table_2_mask(col_ic),
        )
        draw_characteristic(
            ax_reverse, TABLE_2[:, 0], TABLE_2[:, col_ube], color, marker, label,
            table_2_mask(col_ube),
        )

    configure_axes(axes)
    draw_operating_rectangles(axes)

    add_arrow(ax_transfer, (0.04, 0), (-0.025, 0))
    add_arrow(ax_output, (0.96, 0), (1.025, 0))
    add_arrow(ax_transfer, (1, 0.96), (1, 1.025))
    add_arrow(ax_input, (1, 0.04), (1, -0.025))
    axis_labels = [
        (ax_transfer, -0.03, -0.035, r"$I_B\ [\mu\mathrm{A}]$", "right", "top"),
        (ax_output, 1.035, -0.035, r"$U_{CE}\ [\mathrm{V}]$", "left", "top"),
        (ax_transfer, 1.025, 1.035, r"$I_C\ [\mathrm{mA}]$", "left", "bottom"),
        (ax_input, 1.025, -0.04, r"$U_{BE}\ [\mathrm{V}]$", "left", "top"),
    ]
    for ax, x, y, text, ha, va in axis_labels:
        ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=12)

    descriptions = [
        (ax_transfer, "Transfer", r"$I_C=f(I_B),\ U_{CE}=\mathrm{const}$",
         0.95, 0.96, "right"),
        (ax_output, "Output", r"$I_C=f(U_{CE}),\ I_B=\mathrm{const}$",
         0.08, 0.96, "left"),
        (ax_input, "Input", r"$U_{BE}=f(I_B),\ U_{CE}=\mathrm{const}$",
         0.04, 0.78, "left"),
        (ax_reverse, "Reverse", r"$U_{BE}=f(U_{CE}),\ I_B=\mathrm{const}$",
         0.42, 0.78, "left"),
    ]
    for ax, title, relation, x, y, ha in descriptions:
        ax.text(
            x, y, title + "\n" + relation, transform=ax.transAxes,
            ha=ha, va="top", fontsize=10,
            bbox={"facecolor": "none", "edgecolor": "none"},
        )
    ax_transfer.legend(
        loc="lower left", bbox_to_anchor=(0.03, 0.08), framealpha=0.95, fontsize=9,
    )
    ax_output.legend(
        loc="upper right", bbox_to_anchor=(0.97, 0.97), framealpha=0.95, fontsize=9,
    )
    fig.suptitle(PLOT_TITLE, fontsize=16, y=0.965)
    fig.text(
        0.5, 0.915, PLOT_SUBTITLE, ha="center", color="0.35",
    )
    fig.savefig("tabulated_characteristics.png", dpi=200, bbox_inches="tight")
    fig.savefig("tabulated_characteristics.svg", bbox_inches="tight")
    fig.savefig("tabulated_characteristics.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
