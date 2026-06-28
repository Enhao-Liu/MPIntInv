"""Visualization helpers for point clouds and compressed scc2020 bifiltrations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import random
import tempfile
from typing import Optional, Union

_CACHE_ROOT = Path(tempfile.gettempdir()) / "bifiltration_visualization_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
import numpy as np

from resolution_compt import Scc2020ValidationResult, validate_scc2020_file
from finite_grid_cone import (
    GridCompression,
    MaterializedChainComplex,
    compress_chain_complex,
    compute_grid_compression,
    read_scc2020_chain_complex,
)


PathLike = Union[str, Path]
Grade = tuple[int | float, int | float]


@dataclass(frozen=True)
class PointCloudData:
    path: Path
    points: np.ndarray
    function_values: Optional[np.ndarray]
    mode: str
    comments: tuple[str, ...]

    @property
    def dimension(self) -> int:
        return int(self.points.shape[1])

    @property
    def point_count(self) -> int:
        return int(self.points.shape[0])


@dataclass(frozen=True)
class CompressedChainComplexData:
    path: Path
    validation: Scc2020ValidationResult
    complex: MaterializedChainComplex
    compression: GridCompression
    compressed_complex: MaterializedChainComplex


@dataclass(frozen=True)
class GeneratorRecord:
    dimension: int
    index: int
    original_grade: Grade
    compressed_grade: tuple[int, int]


@dataclass(frozen=True)
class GeometricSimplexRecord:
    dimension: int
    index: int
    vertices: tuple[int, ...]
    original_grade: Grade
    compressed_grade: tuple[int, int]


def discover_pointcloud_files(root: PathLike) -> list[Path]:
    """Return point-cloud text files under ``root``."""

    directory = Path(root)
    if not directory.exists():
        return []
    return sorted(path for path in directory.glob("*.txt") if path.is_file())


def discover_scc2020_files(root: PathLike) -> list[Path]:
    """Return likely scc2020 text files under ``root``."""

    directory = Path(root)
    if not directory.exists():
        return []
    candidates = set(directory.glob("*.txt")) | set(directory.glob("*.scc2020"))
    return sorted(path for path in candidates if path.is_file())


def load_pointcloud_for_visualization(path: PathLike, mode: str = "auto") -> PointCloudData:
    """Load point cloud rows in ``xy``, ``xyz``, ``xyf``, or ``xyzf`` format."""

    source_path = Path(path)
    rows: list[list[float]] = []
    comments: list[str] = []
    for line_number, raw_line in enumerate(source_path.read_text(encoding="utf-8").splitlines(), start=1):
        comment = raw_line.partition("#")[2].strip() if "#" in raw_line else ""
        if comment:
            comments.append(comment)
        content = raw_line.partition("#")[0].strip()
        if not content:
            continue
        try:
            rows.append([float(token) for token in content.split()])
        except ValueError as exc:
            raise ValueError(f"{source_path}:{line_number}: expected numeric columns.") from exc

    if not rows:
        raise ValueError(f"{source_path} contains no point rows.")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"{source_path} has inconsistent row widths.")

    resolved_mode = _resolve_pointcloud_mode(width, comments, mode)
    values = np.asarray(rows, dtype=float)
    if resolved_mode == "xy":
        points = values[:, :2]
        function_values = None
    elif resolved_mode == "xyz":
        points = values[:, :3]
        function_values = None
    elif resolved_mode == "xyf":
        points = values[:, :2]
        function_values = values[:, 2]
    elif resolved_mode == "xyzf":
        points = values[:, :3]
        function_values = values[:, 3]
    else:
        raise ValueError(f"Unsupported point-cloud mode {resolved_mode!r}.")

    return PointCloudData(
        path=source_path,
        points=points,
        function_values=function_values,
        mode=resolved_mode,
        comments=tuple(comments),
    )


def plot_pointcloud(
    points: PointCloudData | Sequence[Sequence[float]] | np.ndarray,
    function_values: Optional[Sequence[float] | np.ndarray] = None,
    title: Optional[str] = None,
):
    """Plot a 2D or 3D point cloud with optional function-value coloring."""

    if isinstance(points, PointCloudData):
        data = points
        point_array = data.points
        color_values = data.function_values
        title = title or data.path.name
    else:
        point_array = np.asarray(points, dtype=float)
        color_values = None if function_values is None else np.asarray(function_values, dtype=float)

    if point_array.ndim != 2 or point_array.shape[1] not in {2, 3}:
        raise ValueError("points must be an n x 2 or n x 3 array.")

    if point_array.shape[1] == 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        scatter = ax.scatter(
            point_array[:, 0],
            point_array[:, 1],
            c=color_values,
            cmap="viridis" if color_values is not None else None,
            s=22,
            edgecolors="none",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
    else:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            point_array[:, 0],
            point_array[:, 1],
            point_array[:, 2],
            c=color_values,
            cmap="viridis" if color_values is not None else None,
            s=18,
            depthshade=True,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    ax.set_title(title or "Point cloud")
    if color_values is not None:
        fig.colorbar(scatter, ax=ax, label="function value")
    fig.tight_layout()
    return fig, ax


def describe_pointcloud(data: PointCloudData) -> dict[str, object]:
    """Return a compact point-cloud summary for display."""

    summary: dict[str, object] = {
        "path": str(data.path),
        "mode": data.mode,
        "points": data.point_count,
        "dimension": data.dimension,
        "coordinate_min": np.min(data.points, axis=0).tolist(),
        "coordinate_max": np.max(data.points, axis=0).tolist(),
    }
    if data.function_values is not None:
        summary["function_min"] = float(np.min(data.function_values))
        summary["function_max"] = float(np.max(data.function_values))
    return summary


def load_compressed_chain_complex(path: PathLike) -> CompressedChainComplexData:
    """Validate, read, and compress a complete scc2020 chain complex."""

    source_path = Path(path)
    validation = validate_scc2020_file(source_path)
    if not validation.is_valid:
        raise ValueError(f"{source_path} is not a valid scc2020 file: {validation.errors}")
    if not validation.has_final_grade_block:
        raise ValueError(
            f"{source_path} has no final 0-simplex grade block; this visualization needs a complete complex."
        )

    complex_ = read_scc2020_chain_complex(source_path)
    compression = compute_grid_compression(complex_)
    compressed_complex = compress_chain_complex(complex_, compression)
    return CompressedChainComplexData(
        path=source_path,
        validation=validation,
        complex=complex_,
        compression=compression,
        compressed_complex=compressed_complex,
    )


def generator_records(
    complex_: MaterializedChainComplex,
    compressed_complex: MaterializedChainComplex,
) -> list[GeneratorRecord]:
    """Return generator records with original and compressed grades."""

    if len(complex_.term_grades) != len(compressed_complex.term_grades):
        raise ValueError("Complexes have different numbers of terms.")
    records: list[GeneratorRecord] = []
    for dim, (original_grades, compressed_grades) in enumerate(
        zip(complex_.term_grades, compressed_complex.term_grades)
    ):
        if len(original_grades) != len(compressed_grades):
            raise ValueError(f"Dimension {dim} has mismatched grade counts.")
        for local_index, (original_grade, compressed_grade) in enumerate(
            zip(original_grades, compressed_grades),
            start=1,
        ):
            records.append(
                GeneratorRecord(
                    dimension=dim,
                    index=local_index,
                    original_grade=original_grade,
                    compressed_grade=(int(compressed_grade[0]), int(compressed_grade[1])),
                )
            )
    return records


def reconstruct_geometric_simplices(
    complex_: MaterializedChainComplex,
    compressed_complex: MaterializedChainComplex,
) -> list[GeometricSimplexRecord]:
    """Reconstruct vertex sets of generators from boundary matrices.

    This assumes the scc2020 file comes from a simplicial complex whose C0
    generators are ordered like the paired point cloud rows.  It is intended
    for visualizing vertices, edges, and triangles from Alpha/Rips complexes.
    """

    if len(complex_.term_grades) != len(compressed_complex.term_grades):
        raise ValueError("Complexes have different numbers of terms.")

    simplex_vertices_by_dim: list[list[tuple[int, ...]]] = []
    records: list[GeometricSimplexRecord] = []

    for dim, (original_grades, compressed_grades) in enumerate(
        zip(complex_.term_grades, compressed_complex.term_grades)
    ):
        dim_vertices: list[tuple[int, ...]] = []
        if dim == 0:
            dim_vertices = [(index,) for index in range(len(original_grades))]
        else:
            boundary = complex_.boundaries[dim]
            if boundary is None:
                raise ValueError(f"Missing boundary matrix for C{dim}.")
            lower_vertices = simplex_vertices_by_dim[dim - 1]
            for column_index, rows in enumerate(boundary.columns):
                vertices = sorted({vertex for row in rows for vertex in lower_vertices[row]})
                if len(vertices) != dim + 1:
                    raise ValueError(
                        f"Cannot reconstruct a {dim}-simplex from C{dim} generator "
                        f"{column_index + 1}; got vertices {vertices}."
                    )
                dim_vertices.append(tuple(vertices))

        simplex_vertices_by_dim.append(dim_vertices)
        for local_index, (vertices, original_grade, compressed_grade) in enumerate(
            zip(dim_vertices, original_grades, compressed_grades),
            start=1,
        ):
            records.append(
                GeometricSimplexRecord(
                    dimension=dim,
                    index=local_index,
                    vertices=vertices,
                    original_grade=original_grade,
                    compressed_grade=(int(compressed_grade[0]), int(compressed_grade[1])),
                )
            )

    return records


def summarize_compressed_complex(data: CompressedChainComplexData) -> dict[str, object]:
    """Return generator counts and grid dimensions for display."""

    return {
        "path": str(data.path),
        "max_dim": data.complex.max_dim,
        "term_sizes": {f"C{dim}": len(grades) for dim, grades in enumerate(data.complex.term_grades)},
        "grid_upper": (data.compression.n, data.compression.m),
        "x_values": list(data.compression.x_values),
        "y_values": list(data.compression.y_values),
    }


def plot_birth_grades_on_grid(
    records: Sequence[GeneratorRecord],
    compression: GridCompression,
    *,
    title: str = "Generator birth grades on compression grid",
):
    """Plot each generator at its compressed birth grade."""

    fig, ax = plt.subplots(figsize=(7, 6))
    _draw_grid(ax, compression)
    markers = ("o", "s", "^", "D", "P", "X", "v")
    colors = plt.cm.tab10.colors
    for dim in sorted({record.dimension for record in records}):
        dim_records = [record for record in records if record.dimension == dim]
        xs = [record.compressed_grade[0] for record in dim_records]
        ys = [record.compressed_grade[1] for record in dim_records]
        ax.scatter(
            xs,
            ys,
            s=55,
            alpha=0.72,
            marker=markers[dim % len(markers)],
            color=colors[dim % len(colors)],
            label=f"C{dim} ({len(dim_records)})",
        )

    ax.set_title(title)
    ax.set_xlabel("compressed x")
    ax.set_ylabel("compressed y")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


def plot_existing_simplex_counts(
    records: Sequence[GeneratorRecord],
    compression: GridCompression,
    *,
    by_dimension: bool = False,
    annotate_limit: int = 100,
):
    """Plot cumulative existing-generator counts on the compression grid."""

    if by_dimension:
        dims = sorted({record.dimension for record in records})
    else:
        dims = [None]

    columns = min(3, len(dims))
    rows = int(np.ceil(len(dims) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.4 * rows), squeeze=False)
    for axis, dim in zip(axes.ravel(), dims):
        selected = records if dim is None else [record for record in records if record.dimension == dim]
        matrix = existing_count_matrix(selected, compression)
        image = axis.imshow(matrix, origin="lower", cmap="magma", aspect="auto")
        axis.set_title("all dimensions" if dim is None else f"C{dim}")
        axis.set_xlabel("compressed x")
        axis.set_ylabel("compressed y")
        axis.set_xticks(range(compression.n + 1))
        axis.set_yticks(range(compression.m + 1))
        if matrix.size <= annotate_limit:
            for y_index in range(matrix.shape[0]):
                for x_index in range(matrix.shape[1]):
                    axis.text(x_index, y_index, str(int(matrix[y_index, x_index])), ha="center", va="center")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    for axis in axes.ravel()[len(dims) :]:
        axis.axis("off")
    fig.tight_layout()
    return fig, axes


def selected_grid_coordinates(
    compression: GridCompression,
    *,
    edge_count: int = 2,
    middle_count: int = 4,
    random_seed: Optional[int] = None,
    max_full_cells: int = 100,
) -> tuple[list[int], list[int]]:
    """Return x/y grid coordinates to display in panel plots."""

    total_cells = (compression.n + 1) * (compression.m + 1)
    if total_cells <= max_full_cells:
        return list(range(compression.n + 1)), list(range(compression.m + 1))

    rng = random.Random(random_seed)
    return (
        _axis_sample_coordinates(compression.n, edge_count, middle_count, rng),
        _axis_sample_coordinates(compression.m, edge_count, middle_count, rng),
    )


def plot_alpha_complex_at_grid(
    ax,
    points: PointCloudData | Sequence[Sequence[float]] | np.ndarray,
    simplex_records: Sequence[GeometricSimplexRecord],
    grid_grade: tuple[int, int],
    *,
    show_all_points: bool = False,
    show_labels: bool = False,
) -> None:
    """Draw the geometric simplicial complex present at one compressed grade."""

    point_array = points.points if isinstance(points, PointCloudData) else np.asarray(points, dtype=float)
    if point_array.ndim != 2 or point_array.shape[1] != 2:
        raise ValueError("Compression-grid alpha complex panels require 2D point coordinates.")

    present = [
        record
        for record in simplex_records
        if record.compressed_grade[0] <= grid_grade[0] and record.compressed_grade[1] <= grid_grade[1]
    ]
    triangles = [record.vertices for record in present if record.dimension == 2]
    edges = [record.vertices for record in present if record.dimension == 1]
    vertices = sorted({vertex for record in present for vertex in record.vertices})

    if show_all_points:
        ax.scatter(
            point_array[:, 0],
            point_array[:, 1],
            s=3,
            color="0.82",
            edgecolors="none",
            zorder=1,
        )
    if triangles:
        polys = [point_array[list(triangle), :2] for triangle in triangles]
        ax.add_collection(
            PolyCollection(polys, facecolors="#7aa6d8", edgecolors="none", alpha=0.24, zorder=2)
        )
    if edges:
        segments = [point_array[list(edge), :2] for edge in edges]
        ax.add_collection(LineCollection(segments, colors="#1f4e79", linewidths=0.8, alpha=0.78, zorder=3))
    if vertices:
        vertex_points = point_array[vertices, :2]
        ax.scatter(vertex_points[:, 0], vertex_points[:, 1], s=5, color="#111111", zorder=4)

    ax.set_title(f"{grid_grade}", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    x_min, x_max = float(np.min(point_array[:, 0])), float(np.max(point_array[:, 0]))
    y_min, y_max = float(np.min(point_array[:, 1])), float(np.max(point_array[:, 1]))
    x_pad = max((x_max - x_min) * 0.04, 1e-9)
    y_pad = max((y_max - y_min) * 0.04, 1e-9)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    if show_labels:
        ax.set_xlabel(f"x={grid_grade[0]}", fontsize=8)
        ax.set_ylabel(f"y={grid_grade[1]}", fontsize=8)


def plot_alpha_complex_grid(
    points: PointCloudData | Sequence[Sequence[float]] | np.ndarray,
    simplex_records: Sequence[GeometricSimplexRecord],
    compression: GridCompression,
    *,
    x_coordinates: Optional[Sequence[int]] = None,
    y_coordinates: Optional[Sequence[int]] = None,
    edge_count: int = 2,
    middle_count: int = 4,
    random_seed: Optional[int] = None,
    max_full_cells: int = 100,
    vertical_spacing: Optional[float] = None,
    show_all_points: bool = False,
    title: str = "Alpha complex at selected compression-grid grades",
):
    """Draw alpha complexes in a grid of compressed parameter grades."""

    if x_coordinates is None or y_coordinates is None:
        default_x, default_y = selected_grid_coordinates(
            compression,
            edge_count=edge_count,
            middle_count=middle_count,
            random_seed=random_seed,
            max_full_cells=max_full_cells,
        )
        x_coordinates = default_x if x_coordinates is None else list(x_coordinates)
        y_coordinates = default_y if y_coordinates is None else list(y_coordinates)

    x_values = list(x_coordinates)
    y_values = list(y_coordinates)
    if not x_values or not y_values:
        raise ValueError("At least one x and y grid coordinate is required.")

    columns = len(x_values)
    rows = len(y_values)
    fig, axes = plt.subplots(rows, columns, figsize=(2.1 * columns, 2.1 * rows), squeeze=False)
    for row_index, y_value in enumerate(reversed(y_values)):
        for column_index, x_value in enumerate(x_values):
            ax = axes[row_index][column_index]
            plot_alpha_complex_at_grid(
                ax,
                points,
                simplex_records,
                (x_value, y_value),
                show_all_points=show_all_points,
                show_labels=False,
            )
            if row_index == rows - 1:
                ax.set_xlabel(f"x={x_value}\nr={_format_grid_value(compression.x_values[x_value])}", fontsize=8)
            if column_index == 0:
                ax.set_ylabel(f"y={y_value}\nf={_format_grid_value(compression.y_values[y_value])}", fontsize=8)

    hspace = _auto_vertical_spacing(rows) if vertical_spacing is None else vertical_spacing
    fig.suptitle(title, y=0.995)
    fig.subplots_adjust(
        left=0.045,
        right=0.99,
        bottom=0.045,
        top=0.965,
        wspace=0.08,
        hspace=hspace,
    )
    return fig, axes


def existing_count_matrix(
    records: Sequence[GeneratorRecord],
    compression: GridCompression,
) -> np.ndarray:
    """Return cumulative counts where entry (y, x) counts births <= (x, y)."""

    birth_counts = np.zeros((compression.m + 1, compression.n + 1), dtype=int)
    for record in records:
        x_value, y_value = record.compressed_grade
        birth_counts[y_value, x_value] += 1
    return np.cumsum(np.cumsum(birth_counts, axis=0), axis=1)


def list_existing_simplices_at(
    records: Sequence[GeneratorRecord],
    grid_grade: tuple[int, int],
) -> list[GeneratorRecord]:
    """List generators with compressed birth grade <= ``grid_grade``."""

    x_value, y_value = grid_grade
    existing = [
        record
        for record in records
        if record.compressed_grade[0] <= x_value and record.compressed_grade[1] <= y_value
    ]
    return sorted(existing, key=lambda record: (record.dimension, record.index))


def records_to_table(records: Sequence[GeneratorRecord], *, limit: Optional[int] = None) -> list[dict[str, object]]:
    """Convert generator records into a notebook-friendly table."""

    selected = records if limit is None else records[:limit]
    return [
        {
            "dimension": record.dimension,
            "index": record.index,
            "compressed_grade": record.compressed_grade,
            "original_grade": record.original_grade,
        }
        for record in selected
    ]


def birth_multiplicity_table(records: Sequence[GeneratorRecord]) -> list[dict[str, object]]:
    """Return birth-grade multiplicities grouped by dimension and compressed grade."""

    counter = Counter((record.dimension, record.compressed_grade) for record in records)
    return [
        {"dimension": dim, "compressed_grade": grade, "multiplicity": multiplicity}
        for (dim, grade), multiplicity in sorted(counter.items())
    ]


def _resolve_pointcloud_mode(width: int, comments: Sequence[str], mode: str) -> str:
    if mode != "auto":
        expected_width = {"xy": 2, "xyz": 3, "xyf": 3, "xyzf": 4}.get(mode)
        if expected_width is None:
            raise ValueError("mode must be one of auto, xy, xyz, xyf, xyzf.")
        if width != expected_width:
            raise ValueError(f"mode {mode!r} expects {expected_width} columns, got {width}.")
        return mode

    column_comment = next((comment for comment in comments if comment.lower().startswith("columns:")), "")
    if column_comment:
        tokens = column_comment.partition(":")[2].strip().lower().replace(",", " ").split()
        if tokens == ["x", "y"]:
            return "xy"
        if tokens == ["x", "y", "z"]:
            return "xyz"
        if tokens == ["x", "y", "f"] or tokens == ["x", "y", "function"]:
            return "xyf"
        if tokens == ["x", "y", "z", "f"] or tokens == ["x", "y", "z", "function"]:
            return "xyzf"

    if width == 2:
        return "xy"
    if width == 3:
        return "xyf"
    if width == 4:
        return "xyzf"
    raise ValueError("auto mode supports 2, 3, or 4 numeric columns.")


def _draw_grid(ax, compression: GridCompression) -> None:
    ax.set_xlim(-0.5, compression.n + 0.5)
    ax.set_ylim(-0.5, compression.m + 0.5)
    ax.set_xticks(range(compression.n + 1))
    ax.set_yticks(range(compression.m + 1))
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")


def _axis_sample_coordinates(
    upper: int,
    edge_count: int,
    middle_count: int,
    rng: random.Random,
) -> list[int]:
    if edge_count <= 0:
        raise ValueError("edge_count must be positive.")
    if middle_count < 0:
        raise ValueError("middle_count must be nonnegative.")

    first = range(0, min(edge_count, upper + 1))
    last = range(max(0, upper - edge_count + 1), upper + 1)
    selected = set(first) | set(last)

    middle_start = min(edge_count, upper + 1)
    middle_stop = max(middle_start, upper - edge_count + 1)
    middle_candidates = list(range(middle_start, middle_stop))
    if middle_candidates and middle_count:
        sample_size = min(middle_count, len(middle_candidates))
        selected.update(rng.sample(middle_candidates, sample_size))

    return sorted(selected)


def _format_grid_value(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.4g}"


def _auto_vertical_spacing(rows: int) -> float:
    if rows <= 4:
        return 0.12
    if rows <= 8:
        return 0.08
    return 0.05
