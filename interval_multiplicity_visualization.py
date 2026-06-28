"""Visualization helpers for interval multiplicities on compressed grids."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import os
import tempfile
from typing import Optional

_CACHE_ROOT = Path(tempfile.gettempdir()) / "interval_multiplicity_visualization_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np

from finite_grid_cone import GridCompression, compute_grid_compression, read_scc2020_chain_complex
from finite_grid_interval_candidates import GridBounds, IntervalCandidate
from intmult_from_filtration import IntervalMultiplicity, IntervalMultiplicityComputation


def load_grid_compression_from_scc2020(path: str | Path) -> GridCompression:
    """Load the original-to-compressed grade coordinate map from a complete scc2020 file."""

    return compute_grid_compression(read_scc2020_chain_complex(path))


def summarize_interval_multiplicity_result(result: IntervalMultiplicityComputation) -> dict[str, object]:
    """Return a compact summary of nonzero interval multiplicities."""

    multiplicities = [record.multiplicity for record in result.nonzero_multiplicities]
    sizes = [record.candidate.size for record in result.nonzero_multiplicities]
    return {
        "source_kind": result.source_kind,
        "grid_upper": result.presentation.bounds.max_grade,
        "candidate_intervals": result.candidate_count,
        "nonzero_intervals": len(result.nonzero_multiplicities),
        "multiplicity_values": sorted(set(multiplicities)),
        "multiplicity_min": min(multiplicities) if multiplicities else None,
        "multiplicity_max": max(multiplicities) if multiplicities else None,
        "interval_size_min": min(sizes) if sizes else None,
        "interval_size_max": max(sizes) if sizes else None,
    }


def plot_interval_multiplicity_panels(
    result: IntervalMultiplicityComputation,
    *,
    interval_indices: Optional[Sequence[int]] = None,
    max_intervals: Optional[int] = None,
    columns: int = 7,
    panel_size: float = 1.45,
    sort_by: str = "index",
    cmap: str = "viridis",
    show_axes: bool = False,
    show_grid_for_small_bounds: bool = True,
    small_grid_threshold: int = 25,
    title: Optional[str] = None,
):
    """Plot every selected nonzero interval as connected filled panels.

    Each subplot is the full compressed grid.  The interval support is drawn as
    filled connected square/rectangular grid cells using the interval's
    multiplicity as color.
    """

    records = _select_interval_records(
        result.nonzero_multiplicities,
        interval_indices=interval_indices,
        max_intervals=max_intervals,
        sort_by=sort_by,
    )
    if columns < 1:
        raise ValueError("columns must be positive.")
    if not records:
        fig, ax = plt.subplots(figsize=(4.5, 2.6))
        ax.axis("off")
        ax.text(0.5, 0.5, "No nonzero interval multiplicities", ha="center", va="center")
        fig.tight_layout()
        return fig, np.asarray([[ax]])

    bounds = result.presentation.bounds
    rows = int(np.ceil(len(records) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(panel_size * columns, panel_size * rows),
        squeeze=False,
    )

    norm, colorbar_ticks = _multiplicity_norm(result.nonzero_multiplicities)
    colormap = plt.get_cmap(cmap)

    for ax, record in zip(axes.ravel(), records):
        color = colormap(norm(record.multiplicity))
        _draw_interval_panel(
            ax,
            record.candidate,
            bounds,
            facecolor=color,
            show_axes=show_axes,
            show_grid=show_grid_for_small_bounds and _is_small_grid(bounds, small_grid_threshold),
        )
        ax.set_title(
            f"#{record.interval_index}  mult={record.multiplicity}\nsize={record.candidate.size}",
            fontsize=8,
        )

    for ax in axes.ravel()[len(records) :]:
        ax.axis("off")

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(
        scalar_mappable,
        ax=list(axes.ravel()[: len(records)]),
        fraction=0.025,
        pad=0.012,
    )
    if colorbar_ticks is not None:
        colorbar.set_ticks(colorbar_ticks)
    colorbar.set_label("multiplicity")

    fig.suptitle(title or "Nonzero interval multiplicities on compression grid", y=0.995)
    fig.subplots_adjust(left=0.02, right=0.94, bottom=0.02, top=0.94, wspace=0.08, hspace=0.32)
    return fig, axes


def plot_interval_multiplicities_on_grid(
    result: IntervalMultiplicityComputation,
    *,
    interval_indices: Optional[Sequence[int]] = None,
    max_intervals: Optional[int] = None,
    sort_by: str = "index",
    cmap: str = "viridis",
    alpha: float = 0.28,
    linewidth: float = 0.35,
    draw_internal_cell_edges: bool = False,
    draw_outlines: bool = True,
    outline_linewidth: float = 1.6,
    outline_alpha: float = 1.0,
    outline_color: Optional[str] = None,
    outline_cmap: str = "Set1",
    outline_halo_color: Optional[str] = "black",
    outline_halo_linewidth: float = 3.0,
    outline_halo_alpha: float = 0.9,
    draw_shadows: bool = False,
    shadow_color: str = "black",
    shadow_alpha: float = 0.12,
    shadow_linewidth: Optional[float] = None,
    use_true_radius_scale: bool = False,
    crop_x_to_intervals: bool = False,
    x_crop_padding: Optional[float] = None,
    x_crop_left_padding: Optional[float] = None,
    x_crop_right_padding: Optional[float] = None,
    clamp_x_crop_to_data: bool = False,
    show_src_snk_x_ticks: bool = False,
    only_src_snk_x_ticks: bool = False,
    show_axes: bool = True,
    show_grid_for_small_bounds: bool = True,
    small_grid_threshold: int = 45,
    preserve_cell_aspect: Optional[bool] = None,
    compression: Optional[GridCompression] = None,
    x_axis_label: str = "radius",
    y_axis_label: str = "function value",
    title: Optional[str] = None,
):
    """Draw all selected nonzero intervals on one compressed grid.

    Each interval is drawn as its connected grid support.  Color encodes the
    interval multiplicity, and transparent fills make overlaps visible without
    collapsing the intervals into a single aggregate count.  Internal cell
    edges are hidden by default so long intervals do not show one vertical line
    per compressed grade.  Optional outlines are colored by interval index
    unless a fixed ``outline_color`` is supplied;
    a thicker halo is drawn underneath to keep outlines visible over fills.
    Optional shadows draw a wider low-opacity boundary line underneath each
    interval; the interval fill covers the inner side, leaving the shade mainly
    outside the contour.
    When ``use_true_radius_scale`` is true, the horizontal geometry is drawn
    with cell widths proportional to the original radius values in
    ``compression.x_values``.
    When ``crop_x_to_intervals`` is true, the x-axis is cropped to the selected
    interval supports plus a small padding.  By default the right edge can
    extend past the data boundary to leave visual breathing room.  Left and
    right padding can be set independently.  ``show_src_snk_x_ticks`` adds
    x-axis ticks at the source and sink x-coordinates of selected intervals.
    If ``only_src_snk_x_ticks`` is true, those source/sink ticks replace the
    default x-axis ticks.
    """

    records = _select_interval_records(
        result.nonzero_multiplicities,
        interval_indices=interval_indices,
        max_intervals=max_intervals,
        sort_by=sort_by,
    )
    bounds = result.presentation.bounds
    max_x, max_y = bounds.max_grade
    if compression is not None:
        _validate_compression_matches_bounds(compression, bounds)
    if use_true_radius_scale and compression is None:
        raise ValueError("use_true_radius_scale=True requires a GridCompression.")

    if not records:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No nonzero interval multiplicities", ha="center", va="center")
        fig.tight_layout()
        return fig, ax

    figure_width = min(16.0, max(6.0, 0.045 * (max_x + 1)))
    figure_height = min(12.0, max(4.0, figure_width * (max_y + 1) / max(max_x + 1, 1)))
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    if preserve_cell_aspect is None:
        preserve_cell_aspect = not _is_flat_wide_grid(bounds)

    norm, colorbar_ticks = _multiplicity_norm(result.nonzero_multiplicities)
    colormap = plt.get_cmap(cmap)
    outline_colormap = plt.get_cmap(outline_cmap)
    show_grid = show_grid_for_small_bounds and _is_small_grid(bounds, small_grid_threshold)
    x_edges = _axis_value_cell_edges(compression.x_values, clamp_lower_to_zero=True) if use_true_radius_scale else None
    if shadow_linewidth is None:
        shadow_linewidth = max(outline_halo_linewidth, outline_linewidth) + 3.0
    if use_true_radius_scale and x_edges is not None:
        global_xlim = (x_edges[0], x_edges[-1])
    else:
        global_xlim = (-0.5, max_x + 0.5)
    plot_xlim = global_xlim
    if crop_x_to_intervals:
        selected_xlim = _records_x_support_bounds(records, x_edges=x_edges)
        if selected_xlim is not None:
            plot_xlim = _padded_xlim(
                selected_xlim,
                global_xlim,
                padding=x_crop_padding,
                left_padding=x_crop_left_padding,
                right_padding=x_crop_right_padding,
                use_true_scale=use_true_radius_scale,
                clamp_to_data=clamp_x_crop_to_data,
            )
    src_snk_x_tick_indices = (
        _records_src_snk_x_indices(records, max_x)
        if show_src_snk_x_ticks or only_src_snk_x_ticks
        else ()
    )

    for z_order, record in enumerate(records, start=2):
        polygons = _candidate_column_polygons(record.candidate, x_edges=x_edges)
        if not polygons:
            continue
        boundary_polylines = None
        if draw_shadows or draw_outlines:
            boundary_polylines = _candidate_boundary_polylines(record.candidate, x_edges=x_edges)
        if draw_shadows and shadow_alpha > 0 and shadow_linewidth > 0 and boundary_polylines:
            ax.add_collection(
                LineCollection(
                    boundary_polylines,
                    colors=[shadow_color],
                    linewidths=shadow_linewidth,
                    alpha=shadow_alpha,
                    capstyle="round",
                    joinstyle="round",
                    zorder=z_order - 0.6,
                )
        )
        color = colormap(norm(record.multiplicity))
        if draw_internal_cell_edges:
            fill_edgecolors = [(*color[:3], min(0.95, alpha + 0.28))]
            fill_linewidths = linewidth
        else:
            fill_edgecolors = "none"
            fill_linewidths = 0.0
        ax.add_collection(
            PolyCollection(
                polygons,
                facecolors=[color],
                edgecolors=fill_edgecolors,
                linewidths=fill_linewidths,
                alpha=alpha,
                zorder=z_order,
            )
        )
        if draw_outlines:
            if not boundary_polylines:
                continue
            if outline_color is None:
                outline_position = ((record.interval_index - 1) % outline_colormap.N) / max(
                    outline_colormap.N - 1,
                    1,
                )
                outline_rgba = outline_colormap(outline_position)
            else:
                outline_rgba = outline_color
            if outline_halo_color is not None and outline_halo_linewidth > 0:
                ax.add_collection(
                    LineCollection(
                        boundary_polylines,
                        colors=[outline_halo_color],
                        linewidths=outline_halo_linewidth,
                        alpha=outline_halo_alpha,
                        capstyle="round",
                        joinstyle="round",
                        zorder=len(records) + z_order + 2,
                    )
                )
            ax.add_collection(
                LineCollection(
                    boundary_polylines,
                    colors=[outline_rgba],
                    linewidths=outline_linewidth,
                    alpha=outline_alpha,
                    capstyle="round",
                    joinstyle="round",
                    zorder=len(records) + z_order + 3,
                )
            )

    if use_true_radius_scale and x_edges is not None:
        border_left = x_edges[0]
        border_width = x_edges[-1] - x_edges[0]
    else:
        border_left = -0.5
        border_width = max_x + 1
    ax.add_patch(
        plt.Rectangle(
            (border_left, -0.5),
            border_width,
            max_y + 1,
            fill=False,
            edgecolor="0.15",
            linewidth=0.8,
            zorder=len(records) + 3,
        )
    )
    ax.set_xlim(*plot_xlim)
    ax.set_ylim(-0.5, max_y + 0.5)
    ax.set_aspect("equal" if preserve_cell_aspect else "auto", adjustable="box")

    if show_grid:
        if use_true_radius_scale and compression is not None:
            ax.set_xticks([float(value) for value in compression.x_values])
        else:
            ax.set_xticks(range(max_x + 1))
        ax.set_yticks(range(max_y + 1))
        ax.grid(True, color="0.9", linewidth=0.45, zorder=0)
    elif show_axes:
        if use_true_radius_scale:
            ax.set_yticks(_sparse_ticks(max_y))
        else:
            _set_sparse_grid_ticks(ax, max_x, max_y)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    if show_axes:
        if use_true_radius_scale and compression is not None:
            _apply_true_radius_axis_labels(
                ax,
                max_x,
                max_y,
                compression=compression,
                x_axis_label=x_axis_label,
                y_axis_label=y_axis_label,
                extra_x_tick_indices=src_snk_x_tick_indices,
                only_extra_x_ticks=only_src_snk_x_ticks,
            )
        else:
            _apply_parameter_axis_labels(
                ax,
                max_x,
                max_y,
                compression=compression,
                x_axis_label=x_axis_label,
                y_axis_label=y_axis_label,
                extra_x_tick_indices=src_snk_x_tick_indices,
                only_extra_x_ticks=only_src_snk_x_ticks,
            )
        ax.tick_params(labelsize=8)
        ax.set_xlim(*plot_xlim)
    ax.set_title(title or "All nonzero interval multiplicities on one compression grid")

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=ax, fraction=0.025, pad=0.02)
    if colorbar_ticks is not None:
        colorbar.set_ticks(colorbar_ticks)
    colorbar.set_label("interval multiplicity")

    fig.tight_layout()
    return fig, ax


def plot_interval_multiplicity_overlay(
    result: IntervalMultiplicityComputation,
    *,
    cmap: str = "magma",
    compression: Optional[GridCompression] = None,
    x_axis_label: str = "radius",
    y_axis_label: str = "function value",
    title: Optional[str] = None,
):
    """Plot the total nonzero interval multiplicity covering each grid point."""

    if compression is not None:
        _validate_compression_matches_bounds(compression, result.presentation.bounds)
    matrix = interval_multiplicity_coverage_matrix(result)
    max_x, max_y = result.presentation.bounds.max_grade
    fig, ax = plt.subplots(figsize=(7, 4.5))
    image = ax.imshow(matrix, origin="lower", cmap=cmap, aspect="auto")
    _set_sparse_grid_ticks(ax, max_x, max_y)
    _apply_parameter_axis_labels(
        ax,
        max_x,
        max_y,
        compression=compression,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
    )
    ax.set_title(title or "Total interval multiplicity coverage")
    fig.colorbar(image, ax=ax, label="sum of interval multiplicities")
    fig.tight_layout()
    return fig, ax


def plot_interval_multiplicity_binned_histogram(
    result: IntervalMultiplicityComputation,
    *,
    x_bins: int = 120,
    aggregation: str = "sum",
    stacked: bool = True,
    cmap: str = "tab10",
    compression: Optional[GridCompression] = None,
    x_axis_label: str = "radius",
    y_axis_label: str = "function value",
    title: Optional[str] = None,
):
    """Plot binned coverage of nonzero interval multiplicities along the x-axis.

    The exact coverage matrix is first computed on the compression grid.  Its
    x-coordinate is then grouped into bins, and each row contributes one
    histogram series.  For a two-row filtration, this gives a compact view of
    row-0 and row-1 interval density without overplotting individual intervals.
    """

    if x_bins < 1:
        raise ValueError("x_bins must be positive.")
    if aggregation not in {"sum", "mean", "max"}:
        raise ValueError("aggregation must be one of 'sum', 'mean', or 'max'.")
    if compression is not None:
        _validate_compression_matches_bounds(compression, result.presentation.bounds)

    matrix = interval_multiplicity_coverage_matrix(result)
    binned, bin_ranges = binned_interval_multiplicity_coverage(
        result,
        x_bins=x_bins,
        aggregation=aggregation,
        coverage_matrix=matrix,
    )
    row_count, bin_count = binned.shape
    positions = np.arange(bin_count)
    fig_width = min(16.0, max(7.0, 0.09 * bin_count))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))

    colormap = plt.get_cmap(cmap)
    row_labels = _histogram_row_labels(row_count, compression=compression, y_axis_label=y_axis_label)
    if stacked:
        bottoms = np.zeros(bin_count)
        for row_index in range(row_count):
            ax.bar(
                positions,
                binned[row_index],
                bottom=bottoms,
                width=0.9,
                color=colormap(row_index % colormap.N),
                label=row_labels[row_index],
                linewidth=0,
            )
            bottoms += binned[row_index]
    else:
        group_width = 0.86
        bar_width = group_width / max(row_count, 1)
        start_offset = -group_width / 2 + bar_width / 2
        for row_index in range(row_count):
            ax.bar(
                positions + start_offset + row_index * bar_width,
                binned[row_index],
                width=bar_width,
                color=colormap(row_index % colormap.N),
                label=row_labels[row_index],
                linewidth=0,
            )

    tick_positions = _sparse_ticks(bin_count - 1)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [_format_histogram_bin_tick(index, bin_ranges, compression) for index in tick_positions],
        rotation=0,
        ha="center",
    )
    ax.set_xlim(-0.6, bin_count - 0.4)
    ax.set_xlabel(f"{x_axis_label} bins")
    if aggregation == "sum":
        ax.set_ylabel("sum of interval multiplicity coverage")
    else:
        ax.set_ylabel(f"{aggregation} interval multiplicity coverage")
    ax.set_title(title or "Binned interval multiplicity coverage")
    if row_count > 1:
        ax.legend(loc="upper right", frameon=False)
    ax.grid(True, axis="y", color="0.88", linewidth=0.6)
    fig.tight_layout()
    return fig, ax


def plot_interval_multiplicity_binned_2d_histogram(
    result: IntervalMultiplicityComputation,
    *,
    x_bins: int = 120,
    y_bins: Optional[int] = None,
    aggregation: str = "sum",
    cmap: str = "magma",
    compression: Optional[GridCompression] = None,
    x_axis_label: str = "radius",
    y_axis_label: str = "function value",
    title: Optional[str] = None,
):
    """Plot binned coverage as a discrete 2D histogram on parameter space."""

    if x_bins < 1:
        raise ValueError("x_bins must be positive.")
    if y_bins is not None and y_bins < 1:
        raise ValueError("y_bins must be positive when provided.")
    if aggregation not in {"sum", "mean", "max"}:
        raise ValueError("aggregation must be one of 'sum', 'mean', or 'max'.")
    if compression is not None:
        _validate_compression_matches_bounds(compression, result.presentation.bounds)

    binned, x_ranges, y_ranges = binned_interval_multiplicity_2d_coverage(
        result,
        x_bins=x_bins,
        y_bins=y_bins,
        aggregation=aggregation,
    )
    y_bin_count, x_bin_count = binned.shape
    fig_width = min(16.0, max(7.0, 0.09 * x_bin_count))
    fig_height = min(10.0, max(3.2, 0.55 * y_bin_count + 2.4))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(binned, origin="lower", cmap=cmap, aspect="auto")

    x_tick_positions = _sparse_ticks(x_bin_count - 1)
    y_tick_positions = list(range(y_bin_count)) if y_bin_count <= 12 else _sparse_ticks(y_bin_count - 1)
    ax.set_xticks(x_tick_positions)
    ax.set_yticks(y_tick_positions)
    ax.set_xticklabels(
        [_format_histogram_bin_tick(index, x_ranges, compression) for index in x_tick_positions],
        rotation=0,
        ha="center",
    )
    ax.set_yticklabels(
        [_format_histogram_y_bin_tick(index, y_ranges, compression) for index in y_tick_positions]
    )
    ax.set_xlabel(f"{x_axis_label} bins")
    ax.set_ylabel(f"{y_axis_label} bins")
    label = "sum of interval multiplicity coverage" if aggregation == "sum" else f"{aggregation} interval multiplicity coverage"
    ax.set_title(title or "Binned 2D interval multiplicity coverage")
    fig.colorbar(image, ax=ax, label=label)
    ax.set_xticks(np.arange(-0.5, x_bin_count, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, y_bin_count, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.35, alpha=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    return fig, ax


def binned_interval_multiplicity_coverage(
    result: IntervalMultiplicityComputation,
    *,
    x_bins: int,
    aggregation: str = "sum",
    coverage_matrix: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return binned coverage values and inclusive x-grade bin ranges."""

    if x_bins < 1:
        raise ValueError("x_bins must be positive.")
    if aggregation not in {"sum", "mean", "max"}:
        raise ValueError("aggregation must be one of 'sum', 'mean', or 'max'.")
    matrix = interval_multiplicity_coverage_matrix(result) if coverage_matrix is None else coverage_matrix
    width = matrix.shape[1]
    bin_count = min(int(x_bins), width)
    raw_edges = np.linspace(0, width, bin_count + 1)
    binned = np.zeros((matrix.shape[0], bin_count), dtype=float if aggregation == "mean" else int)
    bin_ranges: list[tuple[int, int]] = []
    for bin_index in range(bin_count):
        start = int(np.floor(raw_edges[bin_index]))
        end = int(np.floor(raw_edges[bin_index + 1]))
        if bin_index == bin_count - 1:
            end = width
        if end <= start:
            end = min(width, start + 1)
        chunk = matrix[:, start:end]
        if aggregation == "sum":
            binned[:, bin_index] = chunk.sum(axis=1)
        elif aggregation == "mean":
            binned[:, bin_index] = chunk.mean(axis=1)
        else:
            binned[:, bin_index] = chunk.max(axis=1)
        bin_ranges.append((start, end - 1))
    return binned, bin_ranges


def binned_interval_multiplicity_2d_coverage(
    result: IntervalMultiplicityComputation,
    *,
    x_bins: int,
    y_bins: Optional[int] = None,
    aggregation: str = "sum",
    coverage_matrix: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    """Return 2D-binned coverage and inclusive x/y grade bin ranges."""

    if x_bins < 1:
        raise ValueError("x_bins must be positive.")
    if y_bins is not None and y_bins < 1:
        raise ValueError("y_bins must be positive when provided.")
    if aggregation not in {"sum", "mean", "max"}:
        raise ValueError("aggregation must be one of 'sum', 'mean', or 'max'.")
    matrix = interval_multiplicity_coverage_matrix(result) if coverage_matrix is None else coverage_matrix
    height, width = matrix.shape
    x_bin_count = min(int(x_bins), width)
    y_bin_count = min(int(y_bins), height) if y_bins is not None else height
    x_ranges = _axis_bin_ranges(width, x_bin_count)
    y_ranges = _axis_bin_ranges(height, y_bin_count)
    binned = np.zeros((y_bin_count, x_bin_count), dtype=float if aggregation == "mean" else int)
    for y_index, (y_start, y_end) in enumerate(y_ranges):
        for x_index, (x_start, x_end) in enumerate(x_ranges):
            chunk = matrix[y_start : y_end + 1, x_start : x_end + 1]
            if aggregation == "sum":
                binned[y_index, x_index] = chunk.sum()
            elif aggregation == "mean":
                binned[y_index, x_index] = chunk.mean()
            else:
                binned[y_index, x_index] = chunk.max()
    return binned, x_ranges, y_ranges


def interval_row_lifetime(record: IntervalMultiplicity, *, row: int = 0) -> int:
    """Return how many compressed x-grades the interval occupies on one row."""

    if row < 0:
        raise ValueError("row must be nonnegative.")
    return sum(end_x - start_x + 1 for start_x, end_x in _candidate_row_segments(record.candidate, row=row))


def select_interval_indices_by_row_lifetime(
    result: IntervalMultiplicityComputation,
    *,
    row: int = 0,
    top_fraction: float = 0.03,
    include_ties: bool = True,
    positive_only: bool = True,
) -> tuple[int, ...]:
    """Select interval indices with largest lifetime on a specified grid row.

    ``top_fraction=0.03`` keeps the top 3 percent after optional removal of
    zero-lifetime intervals.  If ``include_ties`` is true, all intervals whose
    row lifetime equals the cutoff lifetime are retained.
    """

    if row < 0:
        raise ValueError("row must be nonnegative.")
    if not (0 < top_fraction <= 1):
        raise ValueError("top_fraction must be in the interval (0, 1].")

    scored = [
        (interval_row_lifetime(record, row=row), record.interval_index)
        for record in result.nonzero_multiplicities
    ]
    if positive_only:
        scored = [(lifetime, index) for lifetime, index in scored if lifetime > 0]
    if not scored:
        return ()

    scored.sort(key=lambda item: (-item[0], item[1]))
    keep_count = max(1, int(np.ceil(len(scored) * top_fraction)))
    cutoff_lifetime = scored[keep_count - 1][0]
    if include_ties:
        return tuple(index for lifetime, index in scored if lifetime >= cutoff_lifetime)
    return tuple(index for _lifetime, index in scored[:keep_count])


def connected_persistence_diagram_dot_histogram(
    result: IntervalMultiplicityComputation,
    *,
    bins: int,
    interval_indices: Optional[Sequence[int]] = None,
    max_intervals: Optional[int] = None,
    sort_by: str = "index",
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return a binned connected-PD histogram of dots only.

    Segment endpoints are intentionally excluded because they are drawn as
    segment endpoints, not as standalone PD dots.
    """

    if bins < 1:
        raise ValueError("bins must be positive.")
    bounds = result.presentation.bounds
    max_x, max_y = bounds.max_grade
    if max_y > 1:
        raise ValueError(
            "Connected persistence diagram is only defined here for one-row or two-row grids; "
            f"got grid upper {bounds.max_grade}."
        )
    records = _select_interval_records(
        result.nonzero_multiplicities,
        interval_indices=interval_indices,
        max_intervals=max_intervals,
        sort_by=sort_by,
    )
    coordinate_count = max_x + 2
    bin_count = min(int(bins), coordinate_count)
    pd_bin_ranges = _axis_bin_ranges(coordinate_count, bin_count)
    dot_weights: list[tuple[tuple[int, int], int]] = []
    for record in records:
        geometry = _connected_pd_geometry(record.candidate, bounds)
        if geometry is not None and geometry["type"] == "dot":
            dot_weights.append((geometry["data"], record.multiplicity))
    return _connected_pd_dot_histogram(dot_weights, pd_bin_ranges), pd_bin_ranges


def interval_multiplicity_visualization_mode(result: IntervalMultiplicityComputation) -> str:
    """Return the default visualization mode for the interval multiplicity result."""

    max_x, max_y = result.presentation.bounds.max_grade
    if max_y == 0:
        return "persistence_diagram"
    if max_y == 1 and max_x >= 0:
        return "connected_persistence_diagram"
    return "compression_grid"


def plot_interval_multiplicities_auto(
    result: IntervalMultiplicityComputation,
    *,
    interval_indices: Optional[Sequence[int]] = None,
    max_intervals: Optional[int] = None,
    sort_by: str = "index",
    compression: Optional[GridCompression] = None,
    cmap: str = "Blues",
    alpha: float = 0.28,
    linewidth: float = 0.35,
    x_axis_label: str = "radius",
    y_axis_label: str = "function value",
    title: Optional[str] = None,
):
    """Automatically choose a readable interval multiplicity visualization.

    One-row grids are shown as a standard persistence diagram.  Two-row grids
    are shown as a connected persistence diagram using the half-open endpoint
    convention from the 2D-decomposable tutorial.  Larger grids fall back to the
    compressed-grid support visualization.
    """

    mode = interval_multiplicity_visualization_mode(result)
    if mode in {"persistence_diagram", "connected_persistence_diagram"}:
        return plot_connected_persistence_diagram_for_two_row_filtration(
            result,
            interval_indices=interval_indices,
            max_intervals=max_intervals,
            sort_by=sort_by,
            compression=compression,
            cmap=cmap,
            x_axis_label=x_axis_label,
            title=title,
        )
    return plot_interval_multiplicities_on_grid(
        result,
        interval_indices=interval_indices,
        max_intervals=max_intervals,
        sort_by=sort_by,
        cmap=cmap,
        alpha=alpha,
        linewidth=linewidth,
        compression=compression,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        title=title,
    )


def plot_connected_persistence_diagram_for_two_row_filtration(
    result: IntervalMultiplicityComputation,
    *,
    interval_indices: Optional[Sequence[int]] = None,
    max_intervals: Optional[int] = None,
    sort_by: str = "index",
    compression: Optional[GridCompression] = None,
    cmap: str = "Blues",
    dot_size: float = 24.0,
    segment_linewidth: float = 1.6,
    show_segment_endpoints: bool = True,
    x_axis_label: str = "radius",
    title: Optional[str] = None,
):
    """Plot interval multiplicities as a connected persistence diagram.

    This is intended for horizontal one-row and two-row filtrations.  For a
    two-row grid, an upper-row interval ``[a,b]`` is represented by the endpoint
    ``(a,b+1)`` and a lower-row interval ``[c,d]`` by ``(d+1,c)``.  If one
    interval occupies both rows, the two endpoints are connected by a segment.
    """

    bounds = result.presentation.bounds
    max_x, max_y = bounds.max_grade
    if max_y > 1:
        raise ValueError(
            "Connected persistence diagram is only defined here for one-row or two-row grids; "
            f"got grid upper {bounds.max_grade}."
        )
    if compression is not None:
        _validate_compression_matches_bounds(compression, bounds)

    records = _select_interval_records(
        result.nonzero_multiplicities,
        interval_indices=interval_indices,
        max_intervals=max_intervals,
        sort_by=sort_by,
    )
    if not records:
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No nonzero interval multiplicities", ha="center", va="center")
        fig.tight_layout()
        return fig, ax

    plot_items = []
    all_coords: list[int] = []
    for record in records:
        geometry = _connected_pd_geometry(record.candidate, bounds)
        if geometry is None:
            continue
        plot_items.append((record, geometry))
        if geometry["type"] == "dot":
            all_coords.extend(geometry["data"])
        else:
            endpoint_a, endpoint_b = geometry["data"]
            all_coords.extend(endpoint_a)
            all_coords.extend(endpoint_b)

    if not plot_items:
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No drawable interval multiplicities", ha="center", va="center")
        fig.tight_layout()
        return fig, ax

    max_mult = max(record.multiplicity for record, _ in plot_items)
    norm = Normalize(vmin=0, vmax=max_mult)
    colormap = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    limit_min = min(min(all_coords), 0) - 0.5
    limit_max = max(max(all_coords), max_x + 1) + 0.5
    ax.set_xlim(limit_min, limit_max)
    ax.set_ylim(limit_min, limit_max)
    ax.plot(
        [limit_min, limit_max],
        [limit_min, limit_max],
        color="black",
        linestyle="-",
        linewidth=1.0,
        zorder=1,
    )

    for record, geometry in plot_items:
        color = colormap(norm(record.multiplicity))
        if geometry["type"] == "dot":
            x_coord, y_coord = geometry["data"]
            ax.scatter(x_coord, y_coord, color=color, s=dot_size, marker="s", zorder=5)
        else:
            (x0, y0), (x1, y1) = geometry["data"]
            ax.plot(
                [x0, x1],
                [y0, y1],
                color=color,
                linewidth=segment_linewidth,
                zorder=4,
            )
            if show_segment_endpoints:
                ax.scatter(
                    [x0, x1],
                    [y0, y1],
                    color=color,
                    s=max(dot_size * 0.65, 8.0),
                    marker="s",
                    zorder=5,
                )

    _apply_connected_pd_axis_labels(ax, max_x, compression=compression, axis_label=x_axis_label)
    ax.set_title(
        title
        or (
            "Persistence diagram"
            if interval_multiplicity_visualization_mode(result) == "persistence_diagram"
            else "Connected persistence diagram"
        )
    )
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_aspect("equal", adjustable="box")

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("interval multiplicity")
    colorbar.locator = MaxNLocator(integer=True)
    colorbar.update_ticks()

    fig.tight_layout()
    return fig, ax


def plot_connected_persistence_diagram_binned_histogram_for_two_row_filtration(
    result: IntervalMultiplicityComputation,
    *,
    bins: int = 120,
    interval_indices: Optional[Sequence[int]] = None,
    max_intervals: Optional[int] = None,
    sort_by: str = "index",
    compression: Optional[GridCompression] = None,
    cmap: str = "magma",
    segment_color: str = "#00d5ff",
    segment_linewidth: float = 1.25,
    segment_alpha: float = 0.88,
    show_segment_endpoints: bool = True,
    segment_endpoint_size: float = 18.0,
    segment_endpoint_marker: str = "s",
    x_axis_label: str = "radius",
    tick_count: Optional[int] = None,
    show_grid: bool = True,
    grid_color: str = "white",
    grid_linewidth: float = 0.25,
    grid_alpha: float = 0.65,
    title: Optional[str] = None,
):
    """Plot a binned connected persistence diagram with segment overlays.

    Dots and segment endpoints are aggregated into a 2D histogram in connected
    persistence-diagram coordinates.  Segment lines are drawn on top with a
    fixed color so they remain visually distinct from the histogram colormap.
    """

    if bins < 1:
        raise ValueError("bins must be positive.")
    bounds = result.presentation.bounds
    max_x, max_y = bounds.max_grade
    if max_y > 1:
        raise ValueError(
            "Connected persistence diagram is only defined here for one-row or two-row grids; "
            f"got grid upper {bounds.max_grade}."
        )
    if compression is not None:
        _validate_compression_matches_bounds(compression, bounds)

    records = _select_interval_records(
        result.nonzero_multiplicities,
        interval_indices=interval_indices,
        max_intervals=max_intervals,
        sort_by=sort_by,
    )
    if not records:
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No nonzero interval multiplicities", ha="center", va="center")
        fig.tight_layout()
        return fig, ax

    dot_weights: list[tuple[tuple[int, int], int]] = []
    segments: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for record in records:
        geometry = _connected_pd_geometry(record.candidate, bounds)
        if geometry is None:
            continue
        if geometry["type"] == "dot":
            dot_weights.append((geometry["data"], record.multiplicity))
        else:
            endpoint_a, endpoint_b = geometry["data"]
            segments.append((endpoint_a, endpoint_b))

    if not dot_weights and not segments:
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No drawable interval multiplicities", ha="center", va="center")
        fig.tight_layout()
        return fig, ax

    coordinate_count = max_x + 2
    bin_count = min(int(bins), coordinate_count)
    pd_bin_ranges = _axis_bin_ranges(coordinate_count, bin_count)
    histogram = _connected_pd_dot_histogram(dot_weights, pd_bin_ranges)

    fig, ax = plt.subplots(figsize=(6.2, 5.1))
    masked_histogram = np.ma.masked_where(histogram == 0, histogram)
    image = ax.imshow(masked_histogram, origin="lower", cmap=cmap, aspect="equal")
    ax.plot(
        [-0.5, bin_count - 0.5],
        [-0.5, bin_count - 0.5],
        color="black",
        linestyle="-",
        linewidth=1.0,
        zorder=3,
    )

    for endpoint_a, endpoint_b in segments:
        x0 = _value_to_bin_index(endpoint_a[0], pd_bin_ranges)
        y0 = _value_to_bin_index(endpoint_a[1], pd_bin_ranges)
        x1 = _value_to_bin_index(endpoint_b[0], pd_bin_ranges)
        y1 = _value_to_bin_index(endpoint_b[1], pd_bin_ranges)
        ax.plot(
            [x0, x1],
            [y0, y1],
            color="black",
            linewidth=segment_linewidth + 1.1,
            alpha=min(1.0, segment_alpha + 0.1),
            zorder=4,
        )
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=segment_color,
            linewidth=segment_linewidth,
            alpha=segment_alpha,
            zorder=5,
        )
        if show_segment_endpoints:
            ax.scatter(
                [x0, x1],
                [y0, y1],
                color=segment_color,
                edgecolors="black",
                linewidths=0.35,
                s=segment_endpoint_size,
                marker=segment_endpoint_marker,
                zorder=6,
            )

    tick_coordinates = _coordinate_ticks_for_count(coordinate_count - 1, tick_count)
    tick_positions = [_value_to_bin_index(coordinate, pd_bin_ranges) for coordinate in tick_coordinates]
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    labels = [
        _format_connected_pd_coordinate_tick(coordinate, max_x, compression)
        for coordinate in tick_coordinates
    ]
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"{x_axis_label} endpoint bins")
    ax.set_ylabel(f"{x_axis_label} endpoint bins")
    ax.set_title(title or "Binned connected persistence diagram")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("sum of dot multiplicities")
    if show_grid:
        grid_positions = np.arange(-0.5, bin_count + 0.5, 1)
        ax.vlines(
            grid_positions,
            -0.5,
            bin_count - 0.5,
            colors=grid_color,
            linewidth=grid_linewidth,
            alpha=grid_alpha,
            zorder=2.5,
        )
        ax.hlines(
            grid_positions,
            -0.5,
            bin_count - 0.5,
            colors=grid_color,
            linewidth=grid_linewidth,
            alpha=grid_alpha,
            zorder=2.5,
        )
    fig.tight_layout()
    return fig, ax


def interval_multiplicity_coverage_matrix(result: IntervalMultiplicityComputation) -> np.ndarray:
    """Return a matrix whose entry counts multiplicity of intervals covering it."""

    max_x, max_y = result.presentation.bounds.max_grade
    matrix = np.zeros((max_y + 1, max_x + 1), dtype=int)
    for record in result.nonzero_multiplicities:
        for x_coord, column_range in enumerate(record.candidate.column_ranges):
            if column_range is None:
                continue
            lower_y, upper_y = column_range
            matrix[lower_y : upper_y + 1, x_coord] += record.multiplicity
    return matrix


def _select_interval_records(
    records: Sequence[IntervalMultiplicity],
    *,
    interval_indices: Optional[Sequence[int]],
    max_intervals: Optional[int],
    sort_by: str,
) -> list[IntervalMultiplicity]:
    if interval_indices is not None:
        wanted = set(int(index) for index in interval_indices)
        selected = [record for record in records if record.interval_index in wanted]
    else:
        selected = list(records)

    if sort_by == "index":
        selected.sort(key=lambda record: record.interval_index)
    elif sort_by == "multiplicity":
        selected.sort(key=lambda record: (-record.multiplicity, record.interval_index))
    elif sort_by == "size":
        selected.sort(key=lambda record: (-record.candidate.size, record.interval_index))
    else:
        raise ValueError("sort_by must be one of 'index', 'multiplicity', or 'size'.")

    if max_intervals is not None:
        selected = selected[:max_intervals]
    return selected


def _draw_interval_panel(
    ax,
    candidate: IntervalCandidate,
    bounds: GridBounds,
    *,
    facecolor,
    show_axes: bool,
    show_grid: bool,
) -> None:
    max_x, max_y = bounds.max_grade
    polygons = _candidate_column_polygons(candidate)
    if polygons:
        ax.add_collection(
            PolyCollection(
                polygons,
                facecolors=[facecolor],
                edgecolors="white" if show_grid else "none",
                linewidths=0.18 if show_grid else 0.0,
                alpha=0.88,
            )
        )

    ax.set_xlim(-0.5, max_x + 0.5)
    ax.set_ylim(-0.5, max_y + 0.5)
    ax.set_box_aspect(1)
    ax.add_patch(
        plt.Rectangle(
            (-0.5, -0.5),
            max_x + 1,
            max_y + 1,
            fill=False,
            edgecolor="0.35",
            linewidth=0.6,
        )
    )
    if show_grid:
        ax.set_xticks(range(max_x + 1))
        ax.set_yticks(range(max_y + 1))
        ax.grid(True, color="0.9", linewidth=0.45)
    if show_axes:
        ax.tick_params(labelsize=6, length=2)
    else:
        ax.set_xticks([])
        ax.set_yticks([])


def _candidate_column_polygons(
    candidate: IntervalCandidate,
    *,
    x_edges: Optional[Sequence[float]] = None,
) -> list[list[tuple[float, float]]]:
    polygons = []
    for x_coord, column_range in enumerate(candidate.column_ranges):
        if column_range is None:
            continue
        lower_y, upper_y = column_range
        left, right = _column_x_bounds(x_coord, x_edges=x_edges)
        polygons.append(
            [
                (left, lower_y - 0.5),
                (right, lower_y - 0.5),
                (right, upper_y + 0.5),
                (left, upper_y + 0.5),
            ]
        )
    return polygons


def _candidate_x_support_bounds(
    candidate: IntervalCandidate,
    *,
    x_edges: Optional[Sequence[float]] = None,
) -> Optional[tuple[float, float]]:
    active_columns = [index for index, column_range in enumerate(candidate.column_ranges) if column_range is not None]
    if not active_columns:
        return None
    left_column = min(active_columns)
    right_column = max(active_columns)
    left, _ = _column_x_bounds(left_column, x_edges=x_edges)
    _, right = _column_x_bounds(right_column, x_edges=x_edges)
    return left, right


def _records_x_support_bounds(
    records: Sequence[IntervalMultiplicity],
    *,
    x_edges: Optional[Sequence[float]] = None,
) -> Optional[tuple[float, float]]:
    bounds = [
        support_bounds
        for record in records
        if (support_bounds := _candidate_x_support_bounds(record.candidate, x_edges=x_edges)) is not None
    ]
    if not bounds:
        return None
    return min(left for left, _right in bounds), max(right for _left, right in bounds)


def _records_src_snk_x_indices(records: Sequence[IntervalMultiplicity], max_x: int) -> tuple[int, ...]:
    indices = {
        int(point[0])
        for record in records
        for point in (*record.candidate.src, *record.candidate.snk)
        if 0 <= int(point[0]) <= max_x
    }
    return tuple(sorted(indices))


def _padded_xlim(
    selected_xlim: tuple[float, float],
    global_xlim: tuple[float, float],
    *,
    padding: Optional[float],
    left_padding: Optional[float],
    right_padding: Optional[float],
    use_true_scale: bool,
    clamp_to_data: bool,
) -> tuple[float, float]:
    selected_left, selected_right = selected_xlim
    global_left, global_right = global_xlim
    selected_width = max(selected_right - selected_left, 0.0)
    global_width = max(global_right - global_left, 1.0)
    if padding is None:
        if use_true_scale:
            auto_padding = max(0.02 * global_width, 0.08 * selected_width)
        else:
            auto_padding = max(2.0, 0.05 * selected_width)
    else:
        auto_padding = max(float(padding), 0.0)
    left_padding_value = auto_padding if left_padding is None else max(float(left_padding), 0.0)
    right_padding_value = auto_padding if right_padding is None else max(float(right_padding), 0.0)
    left = max(global_left, selected_left - left_padding_value)
    right = selected_right + right_padding_value
    if clamp_to_data:
        right = min(global_right, right)
    return left, right


def _candidate_boundary_segments(
    candidate: IntervalCandidate,
    *,
    x_edges: Optional[Sequence[float]] = None,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    edge_counts: dict[tuple[tuple[float, float], tuple[float, float]], int] = {}
    for x_coord, column_range in enumerate(candidate.column_ranges):
        if column_range is None:
            continue
        lower_y, upper_y = column_range
        for y_coord in range(lower_y, upper_y + 1):
            left, right = _column_x_bounds(x_coord, x_edges=x_edges)
            bottom = y_coord - 0.5
            top = y_coord + 0.5
            edges = (
                ((left, bottom), (right, bottom)),
                ((right, bottom), (right, top)),
                ((left, top), (right, top)),
                ((left, bottom), (left, top)),
            )
            for endpoint_a, endpoint_b in edges:
                key = tuple(sorted((endpoint_a, endpoint_b)))
                edge_counts[key] = edge_counts.get(key, 0) + 1
    return [edge for edge, count in edge_counts.items() if count == 1]


def _candidate_boundary_polylines(
    candidate: IntervalCandidate,
    *,
    x_edges: Optional[Sequence[float]] = None,
) -> list[list[tuple[float, float]]]:
    edges = _candidate_boundary_segments(candidate, x_edges=x_edges)
    adjacency: dict[tuple[float, float], set[tuple[float, float]]] = {}
    unused: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    for endpoint_a, endpoint_b in edges:
        edge = _canonical_edge(endpoint_a, endpoint_b)
        unused.add(edge)
        adjacency.setdefault(endpoint_a, set()).add(endpoint_b)
        adjacency.setdefault(endpoint_b, set()).add(endpoint_a)

    polylines: list[list[tuple[float, float]]] = []
    while unused:
        start, current = min(unused)
        unused.remove((start, current))
        polyline = [start, current]
        previous = start
        while current != start:
            next_candidates = [
                neighbor
                for neighbor in sorted(adjacency[current])
                if _canonical_edge(current, neighbor) in unused
            ]
            if not next_candidates:
                break
            turn_candidates = [neighbor for neighbor in next_candidates if neighbor != previous]
            next_point = turn_candidates[0] if turn_candidates else next_candidates[0]
            unused.remove(_canonical_edge(current, next_point))
            polyline.append(next_point)
            previous, current = current, next_point
        polylines.append(polyline)
    return polylines


def _column_x_bounds(x_coord: int, *, x_edges: Optional[Sequence[float]]) -> tuple[float, float]:
    if x_edges is None:
        return x_coord - 0.5, x_coord + 0.5
    return float(x_edges[x_coord]), float(x_edges[x_coord + 1])


def _axis_value_cell_edges(
    values: Sequence[int | float],
    *,
    clamp_lower_to_zero: bool = False,
) -> tuple[float, ...]:
    if not values:
        raise ValueError("Cannot compute true-scale cell edges for an empty axis.")
    numeric_values = tuple(float(value) for value in values)
    if len(numeric_values) == 1:
        radius = 0.5
        left = numeric_values[0] - radius
        right = numeric_values[0] + radius
        if clamp_lower_to_zero and numeric_values[0] >= 0 and left < 0:
            left = 0.0
        return (left, right)

    edges: list[float] = []
    first_gap = numeric_values[1] - numeric_values[0]
    left_edge = numeric_values[0] - first_gap / 2
    if clamp_lower_to_zero and numeric_values[0] >= 0 and left_edge < 0:
        left_edge = 0.0
    edges.append(left_edge)
    for left_value, right_value in zip(numeric_values, numeric_values[1:]):
        edges.append((left_value + right_value) / 2)
    last_gap = numeric_values[-1] - numeric_values[-2]
    edges.append(numeric_values[-1] + last_gap / 2)
    return tuple(edges)


def _canonical_edge(
    endpoint_a: tuple[float, float],
    endpoint_b: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    return tuple(sorted((endpoint_a, endpoint_b)))


def _connected_pd_dot_histogram(
    dot_weights: Sequence[tuple[tuple[int, int], int]],
    pd_bin_ranges: Sequence[tuple[int, int]],
) -> np.ndarray:
    bin_count = len(pd_bin_ranges)
    histogram = np.zeros((bin_count, bin_count), dtype=int)
    for (x_coord, y_coord), weight in dot_weights:
        x_bin = _value_to_bin_index(x_coord, pd_bin_ranges)
        y_bin = _value_to_bin_index(y_coord, pd_bin_ranges)
        histogram[y_bin, x_bin] += weight
    return histogram


def _connected_pd_geometry(candidate: IntervalCandidate, bounds: GridBounds) -> Optional[dict[str, object]]:
    max_x, max_y = bounds.max_grade
    if max_y == 0:
        segments = _candidate_row_segments(candidate, row=0)
        if not segments:
            return None
        if len(segments) != 1:
            raise ValueError("A one-row interval has non-contiguous row support.")
        start_x, end_x = segments[0]
        return {"type": "dot", "data": (start_x, end_x + 1)}

    upper_segments = _candidate_row_segments(candidate, row=1)
    lower_segments = _candidate_row_segments(candidate, row=0)
    if len(upper_segments) > 1 or len(lower_segments) > 1:
        raise ValueError("A two-row interval has non-contiguous row support.")

    upper_endpoint = None
    lower_endpoint = None
    if upper_segments:
        start_x, end_x = upper_segments[0]
        upper_endpoint = (start_x, end_x + 1)
    if lower_segments:
        start_x, end_x = lower_segments[0]
        lower_endpoint = (end_x + 1, start_x)

    if upper_endpoint is not None and lower_endpoint is not None:
        return {"type": "segment", "data": (upper_endpoint, lower_endpoint)}
    if upper_endpoint is not None:
        return {"type": "dot", "data": upper_endpoint}
    if lower_endpoint is not None:
        return {"type": "dot", "data": lower_endpoint}
    return None


def _candidate_row_segments(candidate: IntervalCandidate, *, row: int) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    active_start: Optional[int] = None
    previous_x: Optional[int] = None

    for x_coord, column_range in enumerate(candidate.column_ranges):
        row_present = False
        if column_range is not None:
            lower_y, upper_y = column_range
            row_present = lower_y <= row <= upper_y

        if row_present:
            if active_start is None:
                active_start = x_coord
            previous_x = x_coord
        elif active_start is not None and previous_x is not None:
            segments.append((active_start, previous_x))
            active_start = None
            previous_x = None

    if active_start is not None and previous_x is not None:
        segments.append((active_start, previous_x))
    return segments


def _is_small_grid(bounds: GridBounds, threshold: int) -> bool:
    max_x, max_y = bounds.max_grade
    return max(max_x + 1, max_y + 1) <= threshold


def _is_flat_wide_grid(bounds: GridBounds) -> bool:
    max_x, max_y = bounds.max_grade
    width = max_x + 1
    height = max_y + 1
    return height <= 2 and width / max(height, 1) > 20


def _validate_compression_matches_bounds(compression: GridCompression, bounds: GridBounds) -> None:
    expected = bounds.max_grade
    if (compression.n, compression.m) != expected:
        raise ValueError(
            "Compression mapping does not match interval grid: "
            f"mapping upper {(compression.n, compression.m)}, interval grid upper {expected}."
        )


def _apply_parameter_axis_labels(
    ax,
    max_x: int,
    max_y: int,
    *,
    compression: Optional[GridCompression],
    x_axis_label: str,
    y_axis_label: str,
    extra_x_tick_indices: Sequence[int] = (),
    only_extra_x_ticks: bool = False,
) -> None:
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    extra_x_ticks = sorted(
        {int(index) for index in extra_x_tick_indices if 0 <= int(index) <= max_x}
    )
    current_x_ticks = [
        int(tick)
        for tick in ax.get_xticks()
        if float(tick).is_integer() and 0 <= int(tick) <= max_x
    ]
    if only_extra_x_ticks:
        x_ticks = extra_x_ticks
    else:
        x_ticks = sorted(set(current_x_ticks).union(extra_x_ticks))
    if compression is None:
        if extra_x_ticks or only_extra_x_ticks:
            ax.set_xticks(x_ticks)
        return

    y_ticks = [int(tick) for tick in ax.get_yticks() if float(tick).is_integer() and 0 <= int(tick) <= max_y]
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([_format_parameter_tick(index, compression.x_values[index]) for index in x_ticks])
    ax.set_yticklabels([_format_parameter_tick(index, compression.y_values[index]) for index in y_ticks])


def _apply_true_radius_axis_labels(
    ax,
    max_x: int,
    max_y: int,
    *,
    compression: GridCompression,
    x_axis_label: str,
    y_axis_label: str,
    extra_x_tick_indices: Sequence[int] = (),
    only_extra_x_ticks: bool = False,
) -> None:
    ax.set_xlabel(f"{x_axis_label} (true scale)")
    ax.set_ylabel(y_axis_label)

    extra_x_ticks = sorted(
        {int(index) for index in extra_x_tick_indices if 0 <= int(index) <= max_x}
    )
    if only_extra_x_ticks:
        x_indices = extra_x_ticks
    else:
        x_indices = sorted(set(_sparse_ticks(max_x)).union(extra_x_ticks))
    x_positions = [float(compression.x_values[index]) for index in x_indices]
    ax.set_xticks(x_positions)
    ax.set_xticklabels([_format_parameter_tick(index, compression.x_values[index]) for index in x_indices])

    y_ticks = [int(tick) for tick in ax.get_yticks() if float(tick).is_integer() and 0 <= int(tick) <= max_y]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([_format_parameter_tick(index, compression.y_values[index]) for index in y_ticks])


def _format_parameter_tick(index: int, value: int | float) -> str:
    return f"{index}\n{_format_grid_value(value)}"


def _apply_connected_pd_axis_labels(
    ax,
    max_x: int,
    *,
    compression: Optional[GridCompression],
    axis_label: str,
) -> None:
    ax.set_xlabel(f"{axis_label} endpoint coordinate")
    ax.set_ylabel(f"{axis_label} endpoint coordinate")
    ticks = _sparse_ticks(max_x + 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if compression is not None:
        labels = [_format_connected_pd_tick(index, max_x, compression) for index in ticks]
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    ax.tick_params(labelsize=8)


def _format_connected_pd_tick(index: int, max_x: int, compression: GridCompression) -> str:
    if 0 <= index <= max_x:
        return _format_parameter_tick(index, compression.x_values[index])
    return f"{index}\n>last"


def _histogram_row_labels(
    row_count: int,
    *,
    compression: Optional[GridCompression],
    y_axis_label: str,
) -> list[str]:
    labels = []
    for row_index in range(row_count):
        if compression is not None and row_index < len(compression.y_values):
            labels.append(f"{y_axis_label} {row_index} ({_format_grid_value(compression.y_values[row_index])})")
        else:
            labels.append(f"{y_axis_label} {row_index}")
    return labels


def _axis_bin_ranges(length: int, bin_count: int) -> list[tuple[int, int]]:
    if length < 1:
        raise ValueError("Cannot bin an empty axis.")
    bin_count = min(bin_count, length)
    raw_edges = np.linspace(0, length, bin_count + 1)
    ranges: list[tuple[int, int]] = []
    for bin_index in range(bin_count):
        start = int(np.floor(raw_edges[bin_index]))
        end = int(np.floor(raw_edges[bin_index + 1]))
        if bin_index == bin_count - 1:
            end = length
        if end <= start:
            end = min(length, start + 1)
        ranges.append((start, end - 1))
    return ranges


def _value_to_bin_index(value: int, bin_ranges: Sequence[tuple[int, int]]) -> int:
    for index, (start, end) in enumerate(bin_ranges):
        if start <= value <= end:
            return index
    if value < bin_ranges[0][0]:
        return 0
    return len(bin_ranges) - 1


def _format_histogram_bin_tick(
    bin_index: int,
    bin_ranges: Sequence[tuple[int, int]],
    compression: Optional[GridCompression],
) -> str:
    start, end = bin_ranges[bin_index]
    if start == end:
        compressed = str(start)
    else:
        compressed = f"{start}-{end}"
    if compression is None:
        return compressed
    start_value = compression.x_values[start]
    end_value = compression.x_values[end]
    if start == end:
        original = _format_grid_value(start_value)
    else:
        original = f"{_format_grid_value(start_value)}-{_format_grid_value(end_value)}"
    return f"{compressed}\n{original}"


def _format_connected_pd_coordinate_tick(
    coordinate: int,
    max_x: int,
    compression: Optional[GridCompression],
) -> str:
    compressed = str(coordinate)
    if compression is None:
        return compressed

    original = _connected_pd_coordinate_label(coordinate, max_x, compression)
    return f"{compressed}\n{original}"


def _connected_pd_coordinate_label(index: int, max_x: int, compression: GridCompression) -> str:
    if 0 <= index <= max_x:
        return _format_grid_value(compression.x_values[index])
    return ">last"


def _format_histogram_y_bin_tick(
    bin_index: int,
    bin_ranges: Sequence[tuple[int, int]],
    compression: Optional[GridCompression],
) -> str:
    start, end = bin_ranges[bin_index]
    if start == end:
        compressed = str(start)
    else:
        compressed = f"{start}-{end}"
    if compression is None:
        return compressed
    start_value = compression.y_values[start]
    end_value = compression.y_values[end]
    if start == end:
        original = _format_grid_value(start_value)
    else:
        original = f"{_format_grid_value(start_value)}-{_format_grid_value(end_value)}"
    return f"{compressed}\n{original}"


def _format_grid_value(value: int | float) -> str:
    if isinstance(value, int):
        if value and abs(value) >= 100000:
            return _format_scientific(value)
        return str(value)
    abs_value = abs(value)
    compact = f"{value:.4g}"
    if abs_value and (abs_value < 1e-3 or abs_value >= 1e4 or len(compact) > 6):
        return _format_scientific(value)
    return compact


def _format_scientific(value: int | float) -> str:
    mantissa, exponent = f"{value:.1e}".split("e")
    exponent_value = int(exponent)
    return f"{mantissa}e{exponent_value}"


def _multiplicity_norm(records: Sequence[IntervalMultiplicity]) -> tuple[Normalize, Optional[list[int]]]:
    multiplicities = [record.multiplicity for record in records]
    min_value = min(multiplicities)
    max_value = max(multiplicities)
    if min_value == max_value:
        return Normalize(vmin=min_value - 0.5, vmax=max_value + 0.5), [min_value]
    return Normalize(vmin=min_value, vmax=max_value), None


def _set_sparse_grid_ticks(ax, max_x: int, max_y: int) -> None:
    ax.set_xticks(_sparse_ticks(max_x))
    ax.set_yticks(_sparse_ticks(max_y))


def _tick_positions_for_count(max_value: int, tick_count: Optional[int]) -> list[int]:
    if tick_count is None:
        return _sparse_ticks(max_value)
    if tick_count < 1:
        raise ValueError("tick_count must be positive when provided.")
    if max_value <= 0:
        return [0]
    if tick_count >= max_value + 1:
        return list(range(max_value + 1))
    positions = np.linspace(0, max_value, tick_count)
    return sorted(set(int(round(value)) for value in positions if 0 <= int(round(value)) <= max_value))


def _coordinate_ticks_for_count(max_coordinate: int, tick_count: Optional[int]) -> list[int]:
    if max_coordinate <= 0:
        return [0]
    if tick_count is None:
        return _sparse_ticks(max_coordinate)
    if tick_count < 1:
        raise ValueError("tick_count must be positive when provided.")
    if tick_count == 1:
        return [0, max_coordinate]
    if tick_count >= max_coordinate + 1:
        return list(range(max_coordinate + 1))
    ticks = {0, max_coordinate}
    ticks.update(int(round(value)) for value in np.linspace(0, max_coordinate, tick_count))
    return sorted(tick for tick in ticks if 0 <= tick <= max_coordinate)


def _sparse_ticks(max_value: int) -> list[int]:
    if max_value <= 10:
        return list(range(max_value + 1))
    ticks = {0, max_value}
    quarter_marks = np.linspace(0, max_value, 5)
    ticks.update(int(round(value)) for value in quarter_marks)
    return sorted(tick for tick in ticks if 0 <= tick <= max_value)
