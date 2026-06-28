"""Convert point-cloud bifiltrations to complete scc2020 chain complexes.

The generated files use two grading parameters and GF(2) coefficients.  The
first parameter is the geometric filtration value, and the second parameter is
the function value filtration.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from itertools import combinations
import math
from pathlib import Path
from typing import Optional, Union


PathLike = Union[str, Path]
Point = tuple[float, ...]
Simplex = tuple[int, ...]
Grade = tuple[float, float]


def read_pointcloud_txt(
    path: PathLike,
    *,
    function_mode: str = "last",
    function_column: Optional[int] = None,
    encoding: str = "utf-8",
) -> tuple[list[Point], list[float]]:
    """Read a 2D/3D point cloud and scalar function values from a text file.

    Blank lines and text after ``#`` are ignored.  With the default
    ``function_mode="last"``, rows must be ``x y f`` or ``x y z f``.  With
    ``function_column`` set, the selected zero-based column is used as the
    function value and all other columns are coordinates.
    """

    rows: list[list[float]] = []
    source_path = Path(path)
    for line_number, raw_line in enumerate(source_path.read_text(encoding=encoding).splitlines(), start=1):
        content = raw_line.partition("#")[0].strip()
        if not content:
            continue
        try:
            row = [float(token) for token in content.split()]
        except ValueError as exc:
            raise ValueError(f"{source_path}:{line_number}: expected numeric columns.") from exc
        if not row:
            continue
        rows.append(row)

    if not rows:
        raise ValueError(f"{source_path} contains no point rows.")

    width = len(rows[0])
    for row_number, row in enumerate(rows, start=1):
        if len(row) != width:
            raise ValueError(
                f"{source_path}: row {row_number} has {len(row)} columns, expected {width}."
            )

    if function_column is not None:
        function_index = _resolve_column_index(function_column, width)
        points = [
            tuple(value for index, value in enumerate(row) if index != function_index)
            for row in rows
        ]
        function_values = [row[function_index] for row in rows]
    else:
        points, function_values = _split_points_by_function_mode(rows, function_mode, source_path)

    _validate_points_and_function_values(points, function_values)
    return points, function_values


def write_simplicial_complex_scc2020(
    simplices_by_dim: Mapping[int, Sequence[Sequence[int]]] | Sequence[Sequence[Sequence[int]]],
    grades_by_dim: Mapping[int, Sequence[Sequence[float]]] | Sequence[Sequence[Sequence[float]]],
    output_path: PathLike,
    *,
    comments: Optional[Sequence[str]] = None,
) -> Path:
    """Write a finite two-parameter simplicial chain complex in scc2020 format."""

    simplices, grades = _normalize_complex_input(simplices_by_dim, grades_by_dim)
    max_dim = len(simplices) - 1
    index_by_dim = [
        {simplex: index for index, simplex in enumerate(dim_simplices)}
        for dim_simplices in simplices
    ]

    lines = [
        "scc2020",
        "2",
        " ".join(str(len(simplices[dim])) for dim in range(max_dim, -1, -1)),
        "--field GF(2)",
    ]
    for comment in comments or ():
        lines.append(f"# {comment}")

    for dim in range(max_dim, 0, -1):
        lines.append(f"# Boundary C_{dim} -> C_{dim - 1}")
        face_index = index_by_dim[dim - 1]
        for simplex, grade in zip(simplices[dim], grades[dim]):
            boundary_rows: list[int] = []
            for face in _codimension_one_faces(simplex):
                if face not in face_index:
                    raise ValueError(f"Missing face {face} of simplex {simplex}.")
                row_index = face_index[face]
                if not _grade_leq(grades[dim - 1][row_index], grade):
                    raise ValueError(
                        "Boundary grade is not compatible: "
                        f"face {face} has grade {_format_grade(grades[dim - 1][row_index])}, "
                        f"simplex {simplex} has grade {_format_grade(grade)}."
                    )
                boundary_rows.append(row_index + 1)
            if len(set(boundary_rows)) != len(boundary_rows):
                raise ValueError(f"Repeated boundary row for simplex {simplex}.")
            line = f"{_format_grade(grade)} ;"
            if boundary_rows:
                line += " " + " ".join(str(row) for row in boundary_rows)
            lines.append(line)

    lines.append("# Row grades for C_0")
    lines.extend(_format_grade(grade) for grade in grades[0])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def build_function_rips_complex(
    points: Sequence[Sequence[float]],
    function_values: Sequence[float],
    *,
    max_dim: int = 2,
    rips_cutoff: Optional[float] = None,
) -> tuple[list[list[Simplex]], list[list[Grade]]]:
    """Build simplices and grades for a function-Rips bifiltration."""

    normalized_points, normalized_values = _validate_points_and_function_values(points, function_values)
    max_dim = _validate_max_dim(max_dim)
    if rips_cutoff is not None and rips_cutoff < 0:
        raise ValueError("rips_cutoff must be non-negative.")

    n_points = len(normalized_points)
    distances: dict[tuple[int, int], float] = {}
    for i, j in combinations(range(n_points), 2):
        distances[(i, j)] = math.dist(normalized_points[i], normalized_points[j])

    simplices: list[list[Simplex]] = []
    grades: list[list[Grade]] = []
    for dim in range(max_dim + 1):
        dim_simplices: list[Simplex] = []
        dim_grades: list[Grade] = []
        for simplex in combinations(range(n_points), dim + 1):
            if dim == 0:
                diameter = 0.0
            else:
                diameter = max(distances[_ordered_pair(i, j)] for i, j in combinations(simplex, 2))
            if rips_cutoff is not None and diameter > rips_cutoff:
                continue
            function_birth = max(normalized_values[index] for index in simplex)
            dim_simplices.append(tuple(simplex))
            dim_grades.append((_normalize_zero(diameter), function_birth))
        simplices.append(dim_simplices)
        grades.append(dim_grades)

    return simplices, grades


def write_function_rips_scc2020(
    points: Sequence[Sequence[float]],
    function_values: Sequence[float],
    output_path: PathLike,
    *,
    max_dim: int = 2,
    rips_cutoff: Optional[float] = None,
    comments: Optional[Sequence[str]] = None,
) -> Path:
    """Write the function-Rips bifiltration as a complete scc2020 file."""

    simplices, grades = build_function_rips_complex(
        points,
        function_values,
        max_dim=max_dim,
        rips_cutoff=rips_cutoff,
    )
    return write_simplicial_complex_scc2020(
        simplices,
        grades,
        output_path,
        comments=comments,
    )


def build_function_alpha_complex(
    points: Sequence[Sequence[float]],
    function_values: Sequence[float],
    *,
    max_dim: int = 2,
    alpha_cutoff: Optional[float] = None,
    squared_radius: bool = False,
) -> tuple[list[list[Simplex]], list[list[Grade]]]:
    """Build simplices and grades for a function-Alpha bifiltration.

    GUDHI returns squared alpha values.  By default, this function converts them
    to radii before writing grades.
    """

    try:
        import gudhi  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "function-Alpha conversion requires the optional 'gudhi' package. "
            "Use the 'rips' subcommand without GUDHI, or install GUDHI in this environment."
        ) from exc

    normalized_points, normalized_values = _validate_points_and_function_values(points, function_values)
    max_dim = _validate_max_dim(max_dim)
    if alpha_cutoff is not None and alpha_cutoff < 0:
        raise ValueError("alpha_cutoff must be non-negative.")

    alpha_complex = gudhi.AlphaComplex(points=[list(point) for point in normalized_points])
    simplex_tree = alpha_complex.create_simplex_tree()

    raw_simplices_by_dim: list[list[Simplex]] = [[] for _ in range(max_dim + 1)]
    raw_grades_by_dim: list[list[Grade]] = [[] for _ in range(max_dim + 1)]
    for simplex_vertices, squared_alpha_value in simplex_tree.get_filtration():
        simplex = tuple(sorted(int(vertex) for vertex in simplex_vertices))
        dim = len(simplex) - 1
        if dim < 0 or dim > max_dim:
            continue
        alpha_value = _convert_alpha_value(float(squared_alpha_value), squared_radius)
        if alpha_cutoff is not None and alpha_value > alpha_cutoff:
            continue
        raw_simplices_by_dim[dim].append(simplex)
        raw_grades_by_dim[dim].append((alpha_value, max(normalized_values[index] for index in simplex)))

    for dim in range(max_dim + 1):
        order = sorted(range(len(raw_simplices_by_dim[dim])), key=lambda index: raw_simplices_by_dim[dim][index])
        raw_simplices_by_dim[dim] = [raw_simplices_by_dim[dim][index] for index in order]
        raw_grades_by_dim[dim] = [raw_grades_by_dim[dim][index] for index in order]

    closed_grades = _monotone_closure(raw_simplices_by_dim, raw_grades_by_dim)
    return raw_simplices_by_dim, closed_grades


def write_function_alpha_scc2020(
    points: Sequence[Sequence[float]],
    function_values: Sequence[float],
    output_path: PathLike,
    *,
    max_dim: int = 2,
    alpha_cutoff: Optional[float] = None,
    squared_radius: bool = False,
    comments: Optional[Sequence[str]] = None,
) -> Path:
    """Write the function-Alpha bifiltration as a complete scc2020 file."""

    simplices, grades = build_function_alpha_complex(
        points,
        function_values,
        max_dim=max_dim,
        alpha_cutoff=alpha_cutoff,
        squared_radius=squared_radius,
    )
    return write_simplicial_complex_scc2020(
        simplices,
        grades,
        output_path,
        comments=comments,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    try:
        points, function_values = read_pointcloud_txt(
            args.input_path,
            function_mode=args.function_mode,
            function_column=args.function_column,
        )
        comments = list(args.comments or ())

        if args.command == "rips":
            simplices, grades = build_function_rips_complex(
                points,
                function_values,
                max_dim=args.max_dim,
                rips_cutoff=args.cutoff,
            )
        elif args.command == "alpha":
            simplices, grades = build_function_alpha_complex(
                points,
                function_values,
                max_dim=args.max_dim,
                alpha_cutoff=args.alpha_cutoff,
                squared_radius=args.squared_radius,
            )
        else:
            parser.error(f"Unknown command {args.command!r}.")

        output_path = write_simplicial_complex_scc2020(
            simplices,
            grades,
            args.output_path,
            comments=comments,
        )
    except (ImportError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")

    counts = ", ".join(f"C_{dim}={len(dim_simplices)}" for dim, dim_simplices in enumerate(simplices))
    print(f"Wrote {output_path}")
    print(f"Points: {len(points)}")
    print(f"Simplices: {counts}")
    if args.command == "alpha":
        print("Alpha radius convention: squared" if args.squared_radius else "Alpha radius convention: radius")
    return 0


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert function-Rips or function-Alpha point-cloud bifiltrations to scc2020.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_arguments(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("input_path", help="Input point-cloud txt file.")
        command_parser.add_argument("output_path", help="Output scc2020 txt file.")
        command_parser.add_argument("--max-dim", type=int, default=2, help="Maximum simplex dimension.")
        command_parser.add_argument(
            "--function-mode",
            choices=("last", "zero", "x", "y", "z"),
            default="last",
            help="How to obtain function values when --function-column is not set.",
        )
        command_parser.add_argument(
            "--function-column",
            type=int,
            default=None,
            help="Zero-based function-value column; negative indices are allowed.",
        )
        command_parser.add_argument(
            "--comments",
            action="append",
            default=[],
            help="Comment line to include in the output; may be repeated.",
        )

    rips_parser = subparsers.add_parser("rips", help="Build a function-Rips bifiltration.")
    add_common_arguments(rips_parser)
    rips_parser.add_argument("--cutoff", type=float, default=None, help="Maximum Rips diameter.")

    alpha_parser = subparsers.add_parser("alpha", help="Build a function-Alpha bifiltration using GUDHI.")
    add_common_arguments(alpha_parser)
    alpha_parser.add_argument("--alpha-cutoff", type=float, default=None, help="Maximum Alpha radius.")
    alpha_parser.add_argument(
        "--squared-radius",
        action="store_true",
        help="Write GUDHI's squared alpha values instead of true radii.",
    )
    return parser


def _split_points_by_function_mode(
    rows: Sequence[Sequence[float]],
    function_mode: str,
    source_path: Path,
) -> tuple[list[Point], list[float]]:
    width = len(rows[0])
    if function_mode == "last":
        if width not in {3, 4}:
            raise ValueError(
                f"{source_path}: --function-mode last expects rows 'x y f' or 'x y z f'."
            )
        return [tuple(row[:-1]) for row in rows], [row[-1] for row in rows]

    if width not in {2, 3}:
        raise ValueError(
            f"{source_path}: --function-mode {function_mode} expects 2D or 3D coordinate rows."
        )
    points = [tuple(row) for row in rows]
    if function_mode == "zero":
        return points, [0.0 for _ in rows]
    if function_mode == "x":
        return points, [row[0] for row in rows]
    if function_mode == "y":
        return points, [row[1] for row in rows]
    if function_mode == "z":
        if width < 3:
            raise ValueError(f"{source_path}: --function-mode z requires 3D coordinates.")
        return points, [row[2] for row in rows]
    raise ValueError(f"Unknown function mode {function_mode!r}.")


def _resolve_column_index(column: int, width: int) -> int:
    index = column if column >= 0 else width + column
    if index < 0 or index >= width:
        raise ValueError(f"function_column {column} is outside row width {width}.")
    return index


def _validate_points_and_function_values(
    points: Sequence[Sequence[float]],
    function_values: Sequence[float],
) -> tuple[list[Point], list[float]]:
    if len(points) != len(function_values):
        raise ValueError("points and function_values must have the same length.")
    if not points:
        raise ValueError("At least one point is required.")

    normalized_points = [tuple(float(coordinate) for coordinate in point) for point in points]
    dimension = len(normalized_points[0])
    if dimension not in {2, 3}:
        raise ValueError("Point coordinates must be 2D or 3D.")
    for index, point in enumerate(normalized_points):
        if len(point) != dimension:
            raise ValueError(f"Point {index} has dimension {len(point)}, expected {dimension}.")
        for coordinate in point:
            _require_finite(coordinate, "point coordinate")

    normalized_values = [float(value) for value in function_values]
    for value in normalized_values:
        _require_finite(value, "function value")
    return normalized_points, normalized_values


def _validate_max_dim(max_dim: int) -> int:
    if max_dim < 0:
        raise ValueError("max_dim must be non-negative.")
    return int(max_dim)


def _normalize_complex_input(
    simplices_by_dim: Mapping[int, Sequence[Sequence[int]]] | Sequence[Sequence[Sequence[int]]],
    grades_by_dim: Mapping[int, Sequence[Sequence[float]]] | Sequence[Sequence[Sequence[float]]],
) -> tuple[list[list[Simplex]], list[list[Grade]]]:
    simplex_dims = _available_dimensions(simplices_by_dim)
    grade_dims = _available_dimensions(grades_by_dim)
    all_dims = simplex_dims | grade_dims
    if not all_dims:
        raise ValueError("The complex must contain at least one dimension.")
    max_dim = max(all_dims)

    simplices: list[list[Simplex]] = []
    grades: list[list[Grade]] = []
    for dim in range(max_dim + 1):
        raw_simplices = _get_dimension_items(simplices_by_dim, dim)
        raw_grades = _get_dimension_items(grades_by_dim, dim)
        if len(raw_simplices) != len(raw_grades):
            raise ValueError(
                f"Dimension {dim} has {len(raw_simplices)} simplices but {len(raw_grades)} grades."
            )
        dim_simplices = [_canonical_simplex(simplex, dim) for simplex in raw_simplices]
        if len(set(dim_simplices)) != len(dim_simplices):
            raise ValueError(f"Dimension {dim} contains duplicate simplices.")
        simplices.append(dim_simplices)
        grades.append([_normalize_grade(grade) for grade in raw_grades])
    return simplices, grades


def _available_dimensions(
    by_dim: Mapping[int, Sequence[object]] | Sequence[Sequence[object]],
) -> set[int]:
    if isinstance(by_dim, Mapping):
        return {int(dim) for dim in by_dim}
    return set(range(len(by_dim)))


def _get_dimension_items(
    by_dim: Mapping[int, Sequence[object]] | Sequence[Sequence[object]],
    dim: int,
) -> Sequence[object]:
    if isinstance(by_dim, Mapping):
        return by_dim.get(dim, ())
    if dim < len(by_dim):
        return by_dim[dim]
    return ()


def _canonical_simplex(simplex: Sequence[int], dim: int) -> Simplex:
    canonical = tuple(sorted(int(vertex) for vertex in simplex))
    if len(canonical) != dim + 1:
        raise ValueError(f"Simplex {tuple(simplex)} has dimension {len(canonical) - 1}, expected {dim}.")
    if len(set(canonical)) != len(canonical):
        raise ValueError(f"Simplex {tuple(simplex)} contains repeated vertices.")
    if any(vertex < 0 for vertex in canonical):
        raise ValueError(f"Simplex {tuple(simplex)} contains a negative vertex index.")
    return canonical


def _normalize_grade(grade: Sequence[float]) -> Grade:
    if len(grade) != 2:
        raise ValueError(f"Expected a two-coordinate grade, got {tuple(grade)}.")
    normalized = (float(grade[0]), float(grade[1]))
    _require_finite(normalized[0], "grade coordinate")
    _require_finite(normalized[1], "grade coordinate")
    return (_normalize_zero(normalized[0]), _normalize_zero(normalized[1]))


def _monotone_closure(
    simplices_by_dim: Sequence[Sequence[Simplex]],
    grades_by_dim: Sequence[Sequence[Grade]],
) -> list[list[Grade]]:
    closed_grades = [list(dim_grades) for dim_grades in grades_by_dim]
    index_by_dim = [
        {simplex: index for index, simplex in enumerate(dim_simplices)}
        for dim_simplices in simplices_by_dim
    ]
    for dim in range(1, len(simplices_by_dim)):
        for simplex_index, simplex in enumerate(simplices_by_dim[dim]):
            x_value, y_value = closed_grades[dim][simplex_index]
            for face in _codimension_one_faces(simplex):
                if face not in index_by_dim[dim - 1]:
                    raise ValueError(f"Missing face {face} of Alpha simplex {simplex}.")
                face_grade = closed_grades[dim - 1][index_by_dim[dim - 1][face]]
                x_value = max(x_value, face_grade[0])
                y_value = max(y_value, face_grade[1])
            closed_grades[dim][simplex_index] = (_normalize_zero(x_value), _normalize_zero(y_value))
    return closed_grades


def _codimension_one_faces(simplex: Simplex) -> list[Simplex]:
    return [simplex[:index] + simplex[index + 1 :] for index in range(len(simplex))]


def _ordered_pair(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def _convert_alpha_value(squared_alpha_value: float, squared_radius: bool) -> float:
    if squared_alpha_value < 0:
        if squared_alpha_value > -1e-12:
            squared_alpha_value = 0.0
        else:
            raise ValueError(f"GUDHI returned a negative squared alpha value: {squared_alpha_value}.")
    if squared_radius:
        return _normalize_zero(squared_alpha_value)
    return _normalize_zero(math.sqrt(squared_alpha_value))


def _grade_leq(left: Grade, right: Grade) -> bool:
    return left[0] <= right[0] and left[1] <= right[1]


def _require_finite(value: float, label: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite, got {value!r}.")


def _normalize_zero(value: float) -> float:
    return 0.0 if value == 0 else value


def _format_grade(grade: Grade) -> str:
    return f"{_format_number(grade[0])} {_format_number(grade[1])}"


def _format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return format(value, ".17g")


if __name__ == "__main__":
    raise SystemExit(main())
