"""
Interfaces for computing presentations and resolutions with 2pac.

This module does not vendor or modify 2pac. It only calls a user-installed
2pac executable and returns file-based results that can be consumed by
downstream Python code in this repository.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from shutil import which
from typing import Callable, Iterable, Optional, Sequence, Union

from finite_grid_cone import FiniteGridConeResult, run_finite_grid_cone_workflow


PathLike = Union[str, Path]
GradeCoordinate = Union[int, float, str]
Grade = tuple[GradeCoordinate, ...]


class Backend(str, Enum):
    """Supported external computation backends."""

    TWOPAC = "2pac"
    AUTO = "auto"


class ResolutionKind(str, Enum):
    """Type of resolution requested by downstream computations."""

    PROJECTIVE = "projective"
    INJECTIVE = "injective"
    PRESENTATION = "presentation"


class InputKind(str, Enum):
    """Input formats supported by the wrappers."""

    SCC2020 = "scc2020"
    FIREP = "firep"
    TWOPAC_MATRIX = "2pac-matrix"
    TWOPAC_CLIQUE = "2pac-clique"
    TWOPAC_FUNCTION_RIPS = "2pac-function-rips"


class TwoPacAlgorithm(str, Enum):
    """Which 2pac algorithm should be used."""

    HOMOLOGY = "homology"
    COHOMOLOGY = "cohomology"
    BOTH = "both"


class ExternalToolError(RuntimeError):
    """Base error for external computation failures."""


class ExternalToolNotFoundError(ExternalToolError):
    """Raised when an expected external executable cannot be found."""


class UnsupportedComputationError(ExternalToolError):
    """Raised when the requested computation is not provided natively."""


class Scc2020ValidationError(ValueError):
    """Raised when an scc2020 file violates the expected header format."""


@dataclass(frozen=True)
class ExternalComputationResult:
    """File-based result returned by 2pac."""

    backend: Backend
    resolution_kind: ResolutionKind
    input_kind: InputKind
    input_path: Path
    output_path: Path
    output_format: str
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float

    def read_output_text(self, encoding: str = "utf-8") -> str:
        """Read a text output file, typically scc2020."""
        return self.output_path.read_text(encoding=encoding)


@dataclass(frozen=True)
class Scc2020Summary:
    """Lightweight summary of an scc2020 output file."""

    path: Path
    num_parameters: Optional[int]
    generator_sizes: tuple[int, ...]
    reverse_coordinates: tuple[int, ...]
    field: Optional[str]
    meaningful_lines: tuple[str, ...]
    data_start_index: int
    data_start_line_number: Optional[int]
    format_warnings: tuple[str, ...]
    raw_lines: tuple[str, ...]


@dataclass(frozen=True)
class Scc2020ValidationResult:
    """Validation result for the scc2020 file structure."""

    summary: Scc2020Summary
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    expected_data_lines_without_final_block: int
    expected_data_lines_with_final_block: int
    actual_data_lines: int
    has_final_grade_block: bool

    @property
    def is_valid(self) -> bool:
        """Return whether no validation errors were found."""
        return not self.errors


@dataclass(frozen=True)
class Scc2020MatrixColumn:
    """One column of an scc2020 graded matrix block."""

    grade: Grade
    row_indices: tuple[int, ...]
    coefficients: tuple[str, ...]


@dataclass(frozen=True)
class Scc2020FreeResolution:
    """Structured view of an scc2020 free resolution output."""

    path: Path
    num_parameters: int
    generator_sizes: tuple[int, ...]
    field: Optional[str]
    generator_grades: tuple[tuple[Grade, ...], ...]
    matrices: tuple[tuple[Scc2020MatrixColumn, ...], ...]

    @property
    def homological_dimension(self) -> int:
        """Return n for a resolution F_n -> ... -> F_0."""
        return len(self.generator_sizes) - 1


@dataclass(frozen=True)
class NakayamaInjectiveCoresolution:
    """
    Injective coresolution data read from a free resolution by Corollary 14.

    For a two-parameter module, the default shift is epsilon=(1, 1). The qth
    injective term is nu(F_{n-q})<epsilon>; with the paper's shift convention,
    finite cogenerator coordinates are therefore shifted down by epsilon.
    """

    source_free_resolution: Scc2020FreeResolution
    epsilon: Grade
    generator_grades: tuple[tuple[Grade, ...], ...]
    matrices: tuple[tuple[Scc2020MatrixColumn, ...], ...]

    @property
    def homological_dimension(self) -> int:
        """Return n for a coresolution Q^0 -> ... -> Q^n."""
        return self.source_free_resolution.homological_dimension


@dataclass(frozen=True)
class GridBounds:
    """Finite rectangular grid bounds ell <= z <= u."""

    ell: Grade
    u: Grade

    def __post_init__(self) -> None:
        ell = tuple(self.ell)
        u = tuple(self.u)
        if len(ell) != len(u):
            raise ValueError("Grid lower and upper bounds must have the same dimension.")
        if not ell:
            raise ValueError("Grid bounds must have at least one coordinate.")
        if any(not _is_finite_numeric_coordinate(coordinate) for coordinate in ell + u):
            raise ValueError("Grid bounds must be finite numeric coordinates.")
        if not _grade_leq(ell, u):
            raise ValueError(f"Grid lower bound must be <= upper bound: {ell} <= {u}.")
        object.__setattr__(self, "ell", ell)
        object.__setattr__(self, "u", u)


@dataclass(frozen=True)
class FiniteGridProjectiveResolution:
    """A non-minimal projective/free resolution restricted to a finite grid."""

    source_free_resolution: Scc2020FreeResolution
    grid: GridBounds
    generator_grades: tuple[tuple[Grade, ...], ...]
    matrices: tuple[tuple[Scc2020MatrixColumn, ...], ...]
    minimal: bool = False

    @property
    def homological_dimension(self) -> int:
        """Return n for a resolution P_n -> ... -> P_0."""
        return len(self.generator_grades) - 1


@dataclass(frozen=True)
class FiniteGridInjectiveCoresolution:
    """A non-minimal injective coresolution restricted to a finite grid."""

    source_injective_coresolution: NakayamaInjectiveCoresolution
    grid: GridBounds
    generator_grades: tuple[tuple[Grade, ...], ...]
    matrices: tuple[tuple[Scc2020MatrixColumn, ...], ...]
    minimal: bool = False

    @property
    def homological_dimension(self) -> int:
        """Return n for a coresolution Q^0 -> ... -> Q^n."""
        return len(self.generator_grades) - 1


def is_executable_available(binary: PathLike) -> bool:
    """Return whether an executable path or executable name is available."""
    try:
        _resolve_executable(binary)
    except ExternalToolNotFoundError:
        return False
    return True


def _is_executable_file(path: Path) -> bool:
    return path.exists() and path.is_file() and os.access(path, os.X_OK)


def _resolve_executable(binary: PathLike) -> str:
    binary_path = Path(binary)
    if binary_path.parent != Path("."):
        if _is_executable_file(binary_path):
            return str(binary_path)
        raise ExternalToolNotFoundError(f"Executable not found: {binary_path}")

    resolved = which(str(binary))
    if resolved is not None:
        return resolved

    venv_bin_dirs = (
        Path(sys.executable).parent,
        Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin"),
    )
    for bin_dir in dict.fromkeys(venv_bin_dirs):
        venv_candidate = bin_dir / str(binary)
        if _is_executable_file(venv_candidate):
            return str(venv_candidate)

    raise ExternalToolNotFoundError(
        f"Executable '{binary}' was not found on PATH or next to the active Python interpreter. "
        "Install it separately from the upstream project or pass its path."
    )


def _scc2020_content(line: str) -> str:
    """Return a line without inline comments for lightweight scc2020 parsing."""
    return line.partition("#")[0].strip()


def is_twopac_python_available(module_name: str = "twopac") -> bool:
    """Return whether a user-installed 2pac Python binding can be imported."""
    return importlib.util.find_spec(module_name) is not None


def read_scc2020_summary(path: PathLike, encoding: str = "utf-8") -> Scc2020Summary:
    """
    Read the top-level metadata of an scc2020 file.

    Per the scc2020 proposal, empty lines and lines starting with ``#`` are
    ignored. The standard order is:

    1. ``scc2020``
    2. number of persistence parameters ``d``
    3. generator-set sizes ``m_1 ... m_{n+1}``
    4. optional ``--reverse`` and ``--field`` lines

    Some tool outputs have historically put ``--field`` before the sizes.
    This reader tolerates that order and records a warning so downstream code
    can still inspect such outputs.
    """
    output_path = Path(path)
    raw_lines = tuple(output_path.read_text(encoding=encoding).splitlines())
    records = _meaningful_scc2020_records(raw_lines)
    if not records:
        raise Scc2020ValidationError(f"Empty scc2020 file: {output_path}")
    if records[0][1] != "scc2020":
        raise Scc2020ValidationError(
            "The first non-empty, non-comment line must be exactly 'scc2020'."
        )
    if len(records) < 3:
        raise Scc2020ValidationError(
            "An scc2020 file must contain at least a header, parameter count, and generator sizes."
        )

    try:
        num_parameters = int(records[1][1])
    except ValueError as exc:
        raise Scc2020ValidationError(
            f"Invalid number of persistence parameters on line {records[1][0]}: {records[1][1]}"
        ) from exc

    if num_parameters < 0:
        raise Scc2020ValidationError("The number of persistence parameters must be non-negative.")

    index = 2
    generator_sizes: Optional[tuple[int, ...]] = None
    reverse_coordinates: tuple[int, ...] = ()
    field = None
    warnings = []

    if records[index][1].startswith("--"):
        warnings.append(
            "Optional header lines appeared before generator sizes. "
            "The scc2020 proposal puts generator sizes third."
        )
        while index < len(records) and records[index][1].startswith("--"):
            reverse_coordinates, field = _parse_scc2020_header_option(
                records[index][0],
                records[index][1],
                num_parameters,
                reverse_coordinates,
                field,
                warnings,
            )
            index += 1
        if index >= len(records):
            raise Scc2020ValidationError("Missing generator sizes line.")
        generator_sizes = _parse_scc2020_generator_sizes(records[index][0], records[index][1])
        index += 1
    else:
        generator_sizes = _parse_scc2020_generator_sizes(records[index][0], records[index][1])
        index += 1
        while index < len(records) and records[index][1].startswith("--"):
            reverse_coordinates, field = _parse_scc2020_header_option(
                records[index][0],
                records[index][1],
                num_parameters,
                reverse_coordinates,
                field,
                warnings,
            )
            index += 1

    data_start_line_number = records[index][0] if index < len(records) else None

    return Scc2020Summary(
        path=output_path,
        num_parameters=num_parameters,
        generator_sizes=generator_sizes,
        reverse_coordinates=reverse_coordinates,
        field=field,
        meaningful_lines=tuple(line for _line_no, line in records),
        data_start_index=index,
        data_start_line_number=data_start_line_number,
        format_warnings=tuple(warnings),
        raw_lines=raw_lines,
    )


def validate_scc2020_file(
    path: PathLike,
    encoding: str = "utf-8",
    *,
    strict_header_order: bool = False,
) -> Scc2020ValidationResult:
    """
    Validate the basic scc2020 structure from the Lesnick-Kerber proposal.

    This checks the header, optional lines, expected block lengths, one-based
    matrix row indices, and repeated row indices within each column. It does
    not validate arithmetic in the specified field. Some tools, including
    2pac, emit optional header lines before generator sizes; by default this is
    accepted with a warning.
    """
    summary = read_scc2020_summary(path, encoding=encoding)
    errors = []
    warnings = list(summary.format_warnings)

    d = summary.num_parameters
    sizes = summary.generator_sizes
    if strict_header_order and any(
        "Optional header lines appeared before generator sizes" in warning for warning in warnings
    ):
        errors.append(
            "Header options such as --reverse or --field must appear after the generator sizes line."
        )
    if len(sizes) < 2:
        errors.append("Generator sizes must contain at least two module sizes.")
    if any(size < 0 for size in sizes):
        errors.append("Generator sizes must be non-negative.")

    for coord in summary.reverse_coordinates:
        if coord < 1 or coord > d:
            errors.append(f"--reverse coordinate {coord} is outside the range 1..{d}.")

    data_lines = summary.meaningful_lines[summary.data_start_index:]
    expected_without_final = sum(sizes[:-1]) if sizes else 0
    expected_with_final = sum(sizes) if sizes else 0
    actual = len(data_lines)
    has_final_block = actual == expected_with_final
    if actual not in {expected_without_final, expected_with_final}:
        errors.append(
            "Unexpected number of data lines: "
            f"got {actual}, expected {expected_without_final} without the final grade block "
            f"or {expected_with_final} with it."
        )

    line_offset = 0
    for block_index in range(max(len(sizes) - 1, 0)):
        source_size = sizes[block_index]
        target_size = sizes[block_index + 1]
        for local_index in range(source_size):
            if line_offset >= len(data_lines):
                break
            errors.extend(
                _validate_scc2020_chain_line(
                    data_lines[line_offset],
                    d,
                    target_size,
                    block_index + 1,
                    local_index + 1,
                )
            )
            line_offset += 1

    if actual >= expected_with_final and sizes:
        for local_index in range(sizes[-1]):
            if line_offset >= len(data_lines):
                break
            errors.extend(
                _validate_scc2020_final_grade_line(
                    data_lines[line_offset],
                    d,
                    local_index + 1,
                )
            )
            line_offset += 1

    return Scc2020ValidationResult(
        summary=summary,
        errors=tuple(errors),
        warnings=tuple(warnings),
        expected_data_lines_without_final_block=expected_without_final,
        expected_data_lines_with_final_block=expected_with_final,
        actual_data_lines=actual,
        has_final_grade_block=has_final_block,
    )


def _meaningful_scc2020_records(raw_lines: Sequence[str]) -> tuple[tuple[int, str], ...]:
    records = []
    for line_number, line in enumerate(raw_lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        records.append((line_number, stripped))
    return tuple(records)


def _parse_scc2020_generator_sizes(line_number: int, line: str) -> tuple[int, ...]:
    tokens = line.split()
    try:
        sizes = tuple(int(token) for token in tokens)
    except ValueError as exc:
        raise Scc2020ValidationError(
            f"Invalid generator sizes on line {line_number}: {line}"
        ) from exc
    if not sizes:
        raise Scc2020ValidationError(f"Missing generator sizes on line {line_number}.")
    return sizes


def _parse_scc2020_header_option(
    line_number: int,
    line: str,
    num_parameters: int,
    reverse_coordinates: tuple[int, ...],
    field: Optional[str],
    warnings: list[str],
) -> tuple[tuple[int, ...], Optional[str]]:
    tokens = line.split()
    option = tokens[0]

    if option == "--reverse":
        if len(tokens) == 1:
            raise Scc2020ValidationError(
                f"Line {line_number}: '--reverse' must list at least one coordinate."
            )
        try:
            coords = tuple(int(token) for token in tokens[1:])
        except ValueError as exc:
            raise Scc2020ValidationError(
                f"Invalid --reverse coordinate on line {line_number}: {line}"
            ) from exc
        if len(set(coords)) != len(coords):
            warnings.append(f"Line {line_number}: repeated coordinates in --reverse.")
        out_of_range = [coord for coord in coords if coord < 1 or coord > num_parameters]
        if out_of_range:
            warnings.append(
                f"Line {line_number}: --reverse coordinates out of range 1..{num_parameters}: "
                f"{out_of_range}"
            )
        return coords, field

    if option == "--field":
        if len(tokens) == 1:
            warnings.append(f"Line {line_number}: --field has no field name.")
            return reverse_coordinates, field
        return reverse_coordinates, " ".join(tokens[1:])

    warnings.append(f"Line {line_number}: unrecognized scc2020 header option '{option}'.")
    return reverse_coordinates, field


def _tokenize_scc2020_data_line(line: str) -> list[str]:
    return line.replace(";", " ; ").split()


def _validate_scc2020_chain_line(
    line: str,
    num_parameters: int,
    target_size: int,
    block_number: int,
    column_number: int,
) -> list[str]:
    errors = []
    tokens = _tokenize_scc2020_data_line(line)
    location = f"block {block_number}, column {column_number}"

    if len(tokens) < num_parameters:
        return [
            f"{location}: expected {num_parameters} grade coordinates, got {len(tokens)} in line '{line}'."
        ]

    entries = tokens[num_parameters:]
    if entries and entries[0] == ";":
        entries = entries[1:]
    if ";" in entries:
        errors.append(f"{location}: semicolon must appear immediately after the grade coordinates.")
        entries = [entry for entry in entries if entry != ";"]

    row_indices = []
    for entry in entries:
        row_token, _sep, coefficient = entry.partition(":")
        if _sep and coefficient == "":
            errors.append(f"{location}: missing coefficient in entry '{entry}'.")
        try:
            row_index = int(row_token)
        except ValueError:
            errors.append(f"{location}: invalid row index in entry '{entry}'.")
            continue
        if row_index <= 0:
            errors.append(f"{location}: row index {row_index} is not one-based positive.")
        if row_index > target_size:
            errors.append(
                f"{location}: row index {row_index} exceeds target generator size {target_size}."
            )
        row_indices.append(row_index)

    repeated = sorted({row_index for row_index in row_indices if row_indices.count(row_index) > 1})
    if repeated:
        errors.append(f"{location}: repeated row indices are not allowed: {repeated}.")

    return errors


def _validate_scc2020_final_grade_line(
    line: str,
    num_parameters: int,
    generator_number: int,
) -> list[str]:
    tokens = _tokenize_scc2020_data_line(line)
    if len(tokens) != num_parameters:
        return [
            "final grade block, generator "
            f"{generator_number}: expected exactly {num_parameters} grade coordinates, "
            f"got {len(tokens)} in line '{line}'."
        ]
    return []


def read_scc2020_free_resolution(path: PathLike, encoding: str = "utf-8") -> Scc2020FreeResolution:
    """
    Parse an scc2020 free-resolution file into matrix blocks and generator grades.

    For a file with sizes ``m_n ... m_0``, the returned ``generator_grades``
    are ordered as ``F_n, ..., F_0``. The ``matrices`` are ordered as
    ``U_n, ..., U_1`` and use the one-based row indices from the scc2020 file.
    A final row-grade block is required, because otherwise the grades of
    ``F_0`` are not present in the file.
    """
    summary = read_scc2020_summary(path, encoding=encoding)
    if summary.num_parameters is None:
        raise Scc2020ValidationError("The scc2020 file has no parameter count.")

    data_lines = summary.meaningful_lines[summary.data_start_index:]
    expected_line_count = sum(summary.generator_sizes)
    if len(data_lines) != expected_line_count:
        raise Scc2020ValidationError(
            "A free-resolution parse requires matrix blocks plus the final row-grade block: "
            f"got {len(data_lines)} data lines, expected {expected_line_count}."
        )

    offset = 0
    generator_grades: list[tuple[Grade, ...]] = []
    matrices: list[tuple[Scc2020MatrixColumn, ...]] = []

    for block_index, source_size in enumerate(summary.generator_sizes[:-1]):
        target_size = summary.generator_sizes[block_index + 1]
        block_columns = []
        block_grades = []
        for _ in range(source_size):
            column = _parse_scc2020_matrix_column(
                data_lines[offset],
                summary.num_parameters,
                target_size,
            )
            block_columns.append(column)
            block_grades.append(column.grade)
            offset += 1
        matrices.append(tuple(block_columns))
        generator_grades.append(tuple(block_grades))

    final_grades = []
    for _ in range(summary.generator_sizes[-1]):
        final_grades.append(
            _parse_scc2020_grade_only_line(data_lines[offset], summary.num_parameters)
        )
        offset += 1
    generator_grades.append(tuple(final_grades))

    return Scc2020FreeResolution(
        path=summary.path,
        num_parameters=summary.num_parameters,
        generator_sizes=summary.generator_sizes,
        field=summary.field,
        generator_grades=tuple(generator_grades),
        matrices=tuple(matrices),
    )


def corollary14_injective_coresolution_from_free_resolution(
    free_resolution: Union[Scc2020FreeResolution, PathLike],
    *,
    epsilon: Optional[Sequence[GradeCoordinate]] = None,
) -> NakayamaInjectiveCoresolution:
    """
    Read the injective coresolution encoded by a full free resolution.

    This implements Bauer-Lenzen-Lesnick Corollary 14. If
    ``F_n -> ... -> F_0 -> M`` is a free resolution of a finite-total-dimension
    ``n``-parameter persistence module, then
    ``Q^q = nu(F_{n-q})<epsilon>`` gives an injective resolution of ``M``.

    In the generator labels returned here, every finite coordinate is shifted
    by subtracting ``epsilon``. Boundary labels such as ``"inf"`` are kept as
    boundary labels.
    """
    resolution = (
        free_resolution
        if isinstance(free_resolution, Scc2020FreeResolution)
        else read_scc2020_free_resolution(free_resolution)
    )

    shift = tuple(epsilon) if epsilon is not None else (1,) * resolution.num_parameters
    if len(shift) != resolution.num_parameters:
        raise ValueError(
            f"epsilon must have {resolution.num_parameters} coordinates, got {len(shift)}."
        )

    injective_grades = tuple(
        tuple(_shift_grade_for_nakayama(grade, shift) for grade in term_grades)
        for term_grades in resolution.generator_grades
    )

    return NakayamaInjectiveCoresolution(
        source_free_resolution=resolution,
        epsilon=shift,
        generator_grades=injective_grades,
        matrices=resolution.matrices,
    )


def restrict_free_resolution_to_grid(
    free_resolution: Union[Scc2020FreeResolution, PathLike],
    grid: Union[GridBounds, tuple[Sequence[GradeCoordinate], Sequence[GradeCoordinate]]],
    *,
    check_composition: bool = True,
) -> FiniteGridProjectiveResolution:
    """
    Restrict a free resolution to finite-grid projectives.

    A summand ``F(z)`` restricts to ``P_G(z join ell)`` when ``z <= u`` and
    to zero otherwise. This operation is functorial but does not minimize the
    resulting finite-grid projective resolution.
    """
    resolution = (
        free_resolution
        if isinstance(free_resolution, Scc2020FreeResolution)
        else read_scc2020_free_resolution(free_resolution)
    )
    bounds = _coerce_grid_bounds(grid)
    _require_matching_grid_dimension(resolution.num_parameters, bounds)

    keep_indices = tuple(
        _kept_indices(term_grades, lambda grade: _grade_leq(grade, bounds.u))
        for term_grades in resolution.generator_grades
    )
    restricted_grades = tuple(
        tuple(_grade_join(term_grades[index], bounds.ell) for index in term_keep)
        for term_grades, term_keep in zip(resolution.generator_grades, keep_indices)
    )
    restricted_matrices = tuple(
        _restrict_matrix_block(
            block,
            row_indices=keep_indices[index + 1],
            col_indices=keep_indices[index],
            new_row_grades=restricted_grades[index + 1],
            new_col_grades=restricted_grades[index],
        )
        for index, block in enumerate(resolution.matrices)
    )

    _validate_matrix_blocks(
        restricted_matrices,
        restricted_grades,
        check_composition=check_composition,
    )

    return FiniteGridProjectiveResolution(
        source_free_resolution=resolution,
        grid=bounds,
        generator_grades=restricted_grades,
        matrices=restricted_matrices,
    )


def restrict_injective_coresolution_to_grid(
    injective_coresolution: Union[NakayamaInjectiveCoresolution, Scc2020FreeResolution, PathLike],
    grid: Union[GridBounds, tuple[Sequence[GradeCoordinate], Sequence[GradeCoordinate]]],
    *,
    epsilon: Optional[Sequence[GradeCoordinate]] = None,
    check_composition: bool = True,
) -> FiniteGridInjectiveCoresolution:
    """
    Restrict a Nakayama injective coresolution to finite-grid injectives.

    An unrestricted summand ``Q(z)`` restricts to the finite-grid summand
    ``Q_G(z meet u)`` when ``z >= ell`` and to zero otherwise. Passing a free
    resolution or path first constructs the Nakayama coresolution using
    ``epsilon``.
    """
    if isinstance(injective_coresolution, NakayamaInjectiveCoresolution):
        coresolution = injective_coresolution
    else:
        coresolution = corollary14_injective_coresolution_from_free_resolution(
            injective_coresolution,
            epsilon=epsilon,
        )

    bounds = _coerce_grid_bounds(grid)
    _require_matching_grid_dimension(
        coresolution.source_free_resolution.num_parameters,
        bounds,
    )

    keep_indices = tuple(
        _kept_indices(term_grades, lambda grade: _grade_geq(grade, bounds.ell))
        for term_grades in coresolution.generator_grades
    )
    restricted_grades = tuple(
        tuple(_grade_meet(term_grades[index], bounds.u) for index in term_keep)
        for term_grades, term_keep in zip(coresolution.generator_grades, keep_indices)
    )
    restricted_matrices = tuple(
        _restrict_matrix_block(
            block,
            row_indices=keep_indices[index + 1],
            col_indices=keep_indices[index],
            new_row_grades=restricted_grades[index + 1],
            new_col_grades=restricted_grades[index],
        )
        for index, block in enumerate(coresolution.matrices)
    )

    _validate_matrix_blocks(
        restricted_matrices,
        restricted_grades,
        check_composition=check_composition,
    )

    return FiniteGridInjectiveCoresolution(
        source_injective_coresolution=coresolution,
        grid=bounds,
        generator_grades=restricted_grades,
        matrices=restricted_matrices,
    )


def finite_grid_projective_resolution_to_dict(
    resolution: FiniteGridProjectiveResolution,
) -> dict:
    """Convert a finite-grid projective resolution to JSON-serializable data."""
    n = resolution.homological_dimension
    return {
        "type": "finite_grid_projective_resolution",
        "minimal": resolution.minimal,
        "field": resolution.source_free_resolution.field,
        "grid": _grid_bounds_to_json(resolution.grid),
        "terms": [
            {
                "degree": degree,
                "summand_kind": "P_G",
                "grades": _grades_to_json(resolution.generator_grades[n - degree]),
            }
            for degree in range(n + 1)
        ],
        "differentials": [
            _matrix_block_to_json(
                name=f"f{degree - 1}",
                from_degree=degree,
                to_degree=degree - 1,
                block=resolution.matrices[n - degree],
            )
            for degree in range(1, n + 1)
        ],
    }


def finite_grid_injective_coresolution_to_dict(
    coresolution: FiniteGridInjectiveCoresolution,
) -> dict:
    """Convert a finite-grid injective coresolution to JSON-serializable data."""
    n = coresolution.homological_dimension
    return {
        "type": "finite_grid_injective_coresolution",
        "minimal": coresolution.minimal,
        "field": coresolution.source_injective_coresolution.source_free_resolution.field,
        "epsilon": _grade_to_json(coresolution.source_injective_coresolution.epsilon),
        "grid": _grid_bounds_to_json(coresolution.grid),
        "terms": [
            {
                "degree": degree,
                "summand_kind": "Q_G",
                "grades": _grades_to_json(coresolution.generator_grades[degree]),
            }
            for degree in range(n + 1)
        ],
        "differentials": [
            _matrix_block_to_json(
                name=f"b{degree}",
                from_degree=degree,
                to_degree=degree + 1,
                block=coresolution.matrices[degree],
            )
            for degree in range(n)
        ],
    }


def write_finite_grid_projective_resolution_json(
    resolution: FiniteGridProjectiveResolution,
    output_path: PathLike,
    *,
    indent: int = 2,
) -> Path:
    """Write a finite-grid projective resolution JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(finite_grid_projective_resolution_to_dict(resolution), indent=indent) + "\n",
        encoding="utf-8",
    )
    return output_file


def write_finite_grid_injective_coresolution_json(
    coresolution: FiniteGridInjectiveCoresolution,
    output_path: PathLike,
    *,
    indent: int = 2,
) -> Path:
    """Write a finite-grid injective coresolution JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(finite_grid_injective_coresolution_to_dict(coresolution), indent=indent) + "\n",
        encoding="utf-8",
    )
    return output_file


def finite_grid_projective_resolution_to_text(
    resolution: FiniteGridProjectiveResolution,
) -> str:
    """Render a finite-grid projective resolution as scc2020-style text."""
    comments = (
        "finite_grid_projective_resolution",
        "summand_kind: P_G",
        f"minimal: {str(resolution.minimal).lower()}",
        f"grid min: {' '.join(str(coordinate) for coordinate in resolution.grid.ell)}",
        f"grid max: {' '.join(str(coordinate) for coordinate in resolution.grid.u)}",
        "semantic note: grades label finite-grid projectives P_G, not unrestricted free modules.",
    )
    return _resolution_blocks_to_scc2020_text(
        num_parameters=len(resolution.grid.ell),
        field=resolution.source_free_resolution.field,
        generator_grades=resolution.generator_grades,
        matrices=resolution.matrices,
        comments=comments,
    )


def finite_grid_injective_coresolution_to_text(
    coresolution: FiniteGridInjectiveCoresolution,
) -> str:
    """Render a finite-grid injective coresolution as scc2020-style text."""
    comments = (
        "finite_grid_injective_coresolution",
        "summand_kind: Q_G",
        f"minimal: {str(coresolution.minimal).lower()}",
        f"epsilon: {' '.join(str(coordinate) for coordinate in coresolution.source_injective_coresolution.epsilon)}",
        f"grid min: {' '.join(str(coordinate) for coordinate in coresolution.grid.ell)}",
        f"grid max: {' '.join(str(coordinate) for coordinate in coresolution.grid.u)}",
        "semantic note: grades label finite-grid injectives Q_G, not unrestricted free modules.",
        "semantic note: matrix blocks are ordered b0, b1, ... for Q^0 -> Q^1 -> ... .",
    )
    return _resolution_blocks_to_scc2020_text(
        num_parameters=len(coresolution.grid.ell),
        field=coresolution.source_injective_coresolution.source_free_resolution.field,
        generator_grades=coresolution.generator_grades,
        matrices=coresolution.matrices,
        comments=comments,
    )


def write_finite_grid_projective_resolution_text(
    resolution: FiniteGridProjectiveResolution,
    output_path: PathLike,
) -> Path:
    """Write a finite-grid projective resolution text file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(finite_grid_projective_resolution_to_text(resolution), encoding="utf-8")
    return output_file


def write_finite_grid_injective_coresolution_text(
    coresolution: FiniteGridInjectiveCoresolution,
    output_path: PathLike,
) -> Path:
    """Write a finite-grid injective coresolution text file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(finite_grid_injective_coresolution_to_text(coresolution), encoding="utf-8")
    return output_file


def _parse_scc2020_matrix_column(
    line: str,
    num_parameters: int,
    target_size: int,
) -> Scc2020MatrixColumn:
    tokens = _tokenize_scc2020_data_line(line)
    if len(tokens) < num_parameters:
        raise Scc2020ValidationError(
            f"Expected {num_parameters} grade coordinates in line '{line}'."
        )

    grade = _parse_scc2020_grade(tokens[:num_parameters])
    entries = tokens[num_parameters:]
    if entries and entries[0] == ";":
        entries = entries[1:]

    row_indices = []
    coefficients = []
    for entry in entries:
        if entry == ";":
            raise Scc2020ValidationError(
                f"Unexpected semicolon after boundary entries in line '{line}'."
            )
        row_token, separator, coefficient = entry.partition(":")
        try:
            row_index = int(row_token)
        except ValueError as exc:
            raise Scc2020ValidationError(
                f"Invalid row index in matrix entry '{entry}'."
            ) from exc
        if row_index <= 0 or row_index > target_size:
            raise Scc2020ValidationError(
                f"Row index {row_index} is outside the target range 1..{target_size}."
            )
        row_indices.append(row_index)
        coefficients.append(coefficient if separator else "1")

    return Scc2020MatrixColumn(
        grade=grade,
        row_indices=tuple(row_indices),
        coefficients=tuple(coefficients),
    )


def _parse_scc2020_grade_only_line(line: str, num_parameters: int) -> Grade:
    tokens = _tokenize_scc2020_data_line(line)
    if len(tokens) != num_parameters:
        raise Scc2020ValidationError(
            f"Expected exactly {num_parameters} grade coordinates in line '{line}'."
        )
    return _parse_scc2020_grade(tokens)


def _parse_scc2020_grade(tokens: Sequence[str]) -> Grade:
    return tuple(_parse_scc2020_grade_coordinate(token) for token in tokens)


def _parse_scc2020_grade_coordinate(token: str) -> GradeCoordinate:
    lowered = token.lower()
    if lowered in {"inf", "+inf", "infinity", "+infinity"}:
        return "inf"
    if lowered in {"-inf", "-infinity"}:
        return "-inf"
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token


def _shift_grade_for_nakayama(grade: Grade, epsilon: Sequence[GradeCoordinate]) -> Grade:
    return tuple(
        _shift_coordinate_for_nakayama(coordinate, shift)
        for coordinate, shift in zip(grade, epsilon)
    )


def _shift_coordinate_for_nakayama(
    coordinate: GradeCoordinate,
    shift: GradeCoordinate,
) -> GradeCoordinate:
    if isinstance(coordinate, (int, float)) and isinstance(shift, (int, float)):
        shifted = coordinate - shift
        if isinstance(shifted, float) and shifted.is_integer():
            return int(shifted)
        return shifted
    return coordinate


def _coerce_grid_bounds(
    grid: Union[GridBounds, tuple[Sequence[GradeCoordinate], Sequence[GradeCoordinate]]],
) -> GridBounds:
    if isinstance(grid, GridBounds):
        return grid
    ell, u = grid
    return GridBounds(tuple(ell), tuple(u))


def _require_matching_grid_dimension(num_parameters: int, grid: GridBounds) -> None:
    if len(grid.ell) != num_parameters:
        raise ValueError(
            f"Grid dimension {len(grid.ell)} does not match "
            f"the resolution dimension {num_parameters}."
        )


def _kept_indices(
    grades: Sequence[Grade],
    keep: Callable[[Grade], bool],
) -> tuple[int, ...]:
    return tuple(index for index, grade in enumerate(grades) if keep(grade))


def _restrict_matrix_block(
    block: Sequence[Scc2020MatrixColumn],
    *,
    row_indices: Sequence[int],
    col_indices: Sequence[int],
    new_row_grades: Sequence[Grade],
    new_col_grades: Sequence[Grade],
) -> tuple[Scc2020MatrixColumn, ...]:
    row_lut = {old_index + 1: new_index + 1 for new_index, old_index in enumerate(row_indices)}
    restricted_columns = []

    for old_column_index, new_column_grade in zip(col_indices, new_col_grades):
        column = block[old_column_index]
        rows = []
        coefficients = []
        for old_row_index, coefficient in zip(column.row_indices, column.coefficients):
            new_row_index = row_lut.get(old_row_index)
            if new_row_index is None:
                continue
            rows.append(new_row_index)
            coefficients.append(coefficient)
        restricted_columns.append(
            Scc2020MatrixColumn(
                grade=tuple(new_column_grade),
                row_indices=tuple(rows),
                coefficients=tuple(coefficients),
            )
        )

    return tuple(restricted_columns)


def _validate_matrix_blocks(
    matrices: Sequence[Sequence[Scc2020MatrixColumn]],
    generator_grades: Sequence[Sequence[Grade]],
    *,
    check_composition: bool,
) -> None:
    if len(matrices) != max(len(generator_grades) - 1, 0):
        raise ValueError("Matrix block count must be one less than the number of terms.")

    for index, block in enumerate(matrices):
        source_grades = generator_grades[index]
        target_grades = generator_grades[index + 1]
        if len(block) != len(source_grades):
            raise ValueError(
                f"Matrix block {index} has {len(block)} columns, "
                f"but its source term has {len(source_grades)} generators."
            )
        _validate_matrix_block_well_defined(block, target_grades)

    if check_composition:
        for index in range(len(matrices) - 1):
            if not _matrix_block_composition_is_zero(matrices[index], matrices[index + 1]):
                raise ValueError(f"Composition of matrix blocks {index} and {index + 1} is nonzero.")


def _validate_matrix_block_well_defined(
    block: Sequence[Scc2020MatrixColumn],
    row_grades: Sequence[Grade],
) -> None:
    for column_index, column in enumerate(block, start=1):
        for row_index in column.row_indices:
            if row_index < 1 or row_index > len(row_grades):
                raise ValueError(
                    f"Matrix column {column_index} has row index {row_index}, "
                    f"outside 1..{len(row_grades)}."
                )
            if not _grade_leq(row_grades[row_index - 1], column.grade):
                raise ValueError(
                    "Matrix entry is not grade-compatible: "
                    f"row grade {row_grades[row_index - 1]} is not <= column grade {column.grade}."
                )


def _matrix_block_composition_is_zero(
    first: Sequence[Scc2020MatrixColumn],
    second: Sequence[Scc2020MatrixColumn],
) -> bool:
    for first_column in first:
        accumulated_rows: set[int] = set()
        for middle_row_index, coefficient in zip(first_column.row_indices, first_column.coefficients):
            if not _coefficient_is_one_gf2(coefficient):
                continue
            if middle_row_index < 1 or middle_row_index > len(second):
                raise ValueError(
                    f"Middle row index {middle_row_index} is outside 1..{len(second)}."
                )
            second_column = second[middle_row_index - 1]
            for target_row_index, second_coefficient in zip(
                second_column.row_indices,
                second_column.coefficients,
            ):
                if not _coefficient_is_one_gf2(second_coefficient):
                    continue
                if target_row_index in accumulated_rows:
                    accumulated_rows.remove(target_row_index)
                else:
                    accumulated_rows.add(target_row_index)
        if accumulated_rows:
            return False
    return True


def _coefficient_is_one_gf2(coefficient: str) -> bool:
    if coefficient in {"", "1"}:
        return True
    if coefficient == "0":
        return False
    try:
        return int(coefficient) % 2 == 1
    except ValueError as exc:
        raise UnsupportedComputationError(
            "Finite-grid composition checks currently support GF(2) coefficients only; "
            f"got coefficient {coefficient!r}."
        ) from exc


def _grade_leq(left: Grade, right: Grade) -> bool:
    return all(
        _compare_grade_coordinate(left_coordinate, right_coordinate) <= 0
        for left_coordinate, right_coordinate in zip(left, right)
    )


def _grade_geq(left: Grade, right: Grade) -> bool:
    return all(
        _compare_grade_coordinate(left_coordinate, right_coordinate) >= 0
        for left_coordinate, right_coordinate in zip(left, right)
    )


def _grade_join(left: Grade, right: Grade) -> Grade:
    return tuple(
        left_coordinate
        if _compare_grade_coordinate(left_coordinate, right_coordinate) >= 0
        else right_coordinate
        for left_coordinate, right_coordinate in zip(left, right)
    )


def _grade_meet(left: Grade, right: Grade) -> Grade:
    return tuple(
        left_coordinate
        if _compare_grade_coordinate(left_coordinate, right_coordinate) <= 0
        else right_coordinate
        for left_coordinate, right_coordinate in zip(left, right)
    )


def _compare_grade_coordinate(left: GradeCoordinate, right: GradeCoordinate) -> int:
    left_rank = _infinite_coordinate_rank(left)
    right_rank = _infinite_coordinate_rank(right)
    if left_rank is not None or right_rank is not None:
        left_rank = 0 if left_rank is None else left_rank
        right_rank = 0 if right_rank is None else right_rank
        return (left_rank > right_rank) - (left_rank < right_rank)

    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        raise ValueError(f"Cannot compare non-numeric grade coordinates: {left!r}, {right!r}.")
    return (left > right) - (left < right)


def _infinite_coordinate_rank(coordinate: GradeCoordinate) -> Optional[int]:
    if not isinstance(coordinate, str):
        return None
    lowered = coordinate.lower()
    if lowered in {"-inf", "-infinity"}:
        return -1
    if lowered in {"inf", "+inf", "infinity", "+infinity"}:
        return 1
    raise ValueError(f"Unknown symbolic grade coordinate: {coordinate!r}.")


def _is_finite_numeric_coordinate(coordinate: GradeCoordinate) -> bool:
    return isinstance(coordinate, (int, float)) and not isinstance(coordinate, bool)


def _grid_bounds_to_json(grid: GridBounds) -> dict:
    return {
        "ell": _grade_to_json(grid.ell),
        "u": _grade_to_json(grid.u),
    }


def _grades_to_json(grades: Sequence[Grade]) -> list[list[GradeCoordinate]]:
    return [_grade_to_json(grade) for grade in grades]


def _grade_to_json(grade: Sequence[GradeCoordinate]) -> list[GradeCoordinate]:
    return [
        int(coordinate)
        if isinstance(coordinate, int) and not isinstance(coordinate, bool)
        else coordinate
        for coordinate in grade
    ]


def _matrix_block_to_json(
    *,
    name: str,
    from_degree: int,
    to_degree: int,
    block: Sequence[Scc2020MatrixColumn],
) -> dict:
    return {
        "name": name,
        "from_degree": from_degree,
        "to_degree": to_degree,
        "matrix_format": "sparse_columns",
        "index_base": 1,
        "columns": [list(column.row_indices) for column in block],
        "coefficients": [list(column.coefficients) for column in block],
    }


def _format_grade_for_text(grade: Sequence[GradeCoordinate]) -> str:
    return "(" + ", ".join(str(coordinate) for coordinate in grade) + ")"


def _format_summand_list_for_text(kind: str, grades: Sequence[Grade]) -> str:
    if not grades:
        return "0"
    return " + ".join(f"{kind}{_format_grade_for_text(grade)}" for grade in grades)


def _format_matrix_block_for_text(
    *,
    name: str,
    block: Sequence[Scc2020MatrixColumn],
    row_grades: Sequence[Grade],
    column_grades: Sequence[Grade],
) -> list[str]:
    lines = [
        f"  {name}",
        f"    rows: {_format_grade_list_for_text(row_grades)}",
        f"    columns: {_format_grade_list_for_text(column_grades)}",
        "    matrix:",
    ]
    dense_rows = _matrix_block_dense_rows(block, row_count=len(row_grades))
    if not dense_rows:
        lines.append("      []")
    else:
        for row in dense_rows:
            lines.append("      [" + " ".join(row) + "]")
    return lines


def _format_grade_list_for_text(grades: Sequence[Grade]) -> str:
    if not grades:
        return "[]"
    return "[" + ", ".join(_format_grade_for_text(grade) for grade in grades) + "]"


def _matrix_block_dense_rows(
    block: Sequence[Scc2020MatrixColumn],
    *,
    row_count: int,
) -> list[list[str]]:
    if row_count == 0:
        return []
    dense = [["0" for _column in block] for _row in range(row_count)]
    for column_index, column in enumerate(block):
        for row_index, coefficient in zip(column.row_indices, column.coefficients):
            dense[row_index - 1][column_index] = coefficient or "1"
    return dense


def _resolution_blocks_to_scc2020_text(
    *,
    num_parameters: int,
    field: Optional[str],
    generator_grades: Sequence[Sequence[Grade]],
    matrices: Sequence[Sequence[Scc2020MatrixColumn]],
    comments: Sequence[str] = (),
) -> str:
    sizes = " ".join(str(len(term_grades)) for term_grades in generator_grades)
    lines = [
        "scc2020",
        str(num_parameters),
        sizes,
    ]
    if field is not None:
        lines.append(f"--field {field}")
    lines.extend(f"# {comment}" for comment in comments)

    for block_index, block in enumerate(matrices, start=1):
        lines.append(f"# matrix block {block_index}")
        lines.extend(_scc2020_column_line(column) for column in block)

    lines.append("# Row grades for final term")
    if generator_grades:
        lines.extend(_scc2020_grade_line(grade) for grade in generator_grades[-1])
    return "\n".join(lines) + "\n"


def _scc2020_column_line(column: Scc2020MatrixColumn) -> str:
    line = _scc2020_grade_line(column.grade) + " ;"
    entries = [
        _scc2020_entry(row_index, coefficient)
        for row_index, coefficient in zip(column.row_indices, column.coefficients)
        if coefficient != "0"
    ]
    if entries:
        line += " " + " ".join(entries)
    return line


def _scc2020_grade_line(grade: Sequence[GradeCoordinate]) -> str:
    return " ".join(_scc2020_coordinate(coordinate) for coordinate in grade)


def _scc2020_entry(row_index: int, coefficient: str) -> str:
    if coefficient in {"", "1"}:
        return str(row_index)
    return f"{row_index}:{coefficient}"


def _scc2020_coordinate(coordinate: GradeCoordinate) -> str:
    if isinstance(coordinate, float) and coordinate.is_integer():
        return str(int(coordinate))
    return str(coordinate)


def compute_finite_grid_projective_resolution(
    input_path: PathLike,
    *,
    input_kind: Union[InputKind, str] = InputKind.SCC2020,
    output_dir: PathLike = "outputs/resolution_computation",
    output_prefix: Optional[str] = None,
    twopac_binary: PathLike = "2pac",
    target_dim: int = 1,
    samples: Optional[int] = None,
    chunkdim: Optional[int] = None,
    chain_chunk: bool = False,
    flip_grading: bool = False,
    sfd: Optional[str] = None,
    truncate: Optional[float] = None,
    timeout: Optional[float] = None,
    verbose: bool = False,
    keep_zero_extended_resolution: bool = False,
    save_json: bool = True,
) -> FiniteGridConeResult:
    """
    Compute a finite-grid projective/free resolution via finite-boundary coning.

    This workflow does not modify 2pac source code. It builds a temporary
    finite-boundary-coned scc2020 chain complex in Python, calls the installed
    2pac executable on that temporary file, and writes both JSON and
    scc2020-style txt finite-grid outputs, with optional JSON sidecars.
    """
    kind = InputKind(input_kind)
    if kind == InputKind.FIREP:
        raise UnsupportedComputationError(
            "Finite-grid cone uses 2pac input formats; firep is not supported."
        )
    return run_finite_grid_cone_workflow(
        input_path,
        input_kind=kind.value,
        output_dir=output_dir,
        output_prefix=output_prefix,
        twopac_binary=twopac_binary,
        target_dim=target_dim,
        save_injective=False,
        samples=samples,
        chunkdim=chunkdim,
        chain_chunk=chain_chunk,
        flip_grading=flip_grading,
        sfd=sfd,
        truncate=truncate,
        timeout=timeout,
        verbose=verbose,
        keep_zero_extended_resolution=keep_zero_extended_resolution,
        save_json=save_json,
    )


def compute_finite_grid_injective_coresolution(
    input_path: PathLike,
    *,
    input_kind: Union[InputKind, str] = InputKind.SCC2020,
    output_dir: PathLike = "outputs/resolution_computation",
    output_prefix: Optional[str] = None,
    twopac_binary: PathLike = "2pac",
    target_dim: int = 1,
    samples: Optional[int] = None,
    chunkdim: Optional[int] = None,
    chain_chunk: bool = False,
    flip_grading: bool = False,
    sfd: Optional[str] = None,
    truncate: Optional[float] = None,
    timeout: Optional[float] = None,
    verbose: bool = False,
    keep_zero_extended_resolution: bool = False,
    save_json: bool = True,
) -> FiniteGridConeResult:
    """
    Compute finite-grid projective/free and injective outputs via boundary coning.

    The injective coresolution is read from the zero-extended free resolution
    using the Nakayama shift, and finite-grid injective summands are written as
    ``Q_G``.
    """
    kind = InputKind(input_kind)
    if kind == InputKind.FIREP:
        raise UnsupportedComputationError(
            "Finite-grid cone uses 2pac input formats; firep is not supported."
        )
    return run_finite_grid_cone_workflow(
        input_path,
        input_kind=kind.value,
        output_dir=output_dir,
        output_prefix=output_prefix,
        twopac_binary=twopac_binary,
        target_dim=target_dim,
        save_injective=True,
        samples=samples,
        chunkdim=chunkdim,
        chain_chunk=chain_chunk,
        flip_grading=flip_grading,
        sfd=sfd,
        truncate=truncate,
        timeout=timeout,
        verbose=verbose,
        keep_zero_extended_resolution=keep_zero_extended_resolution,
        save_json=save_json,
    )


def compute_minimal_projective_resolution(
    input_path: PathLike,
    *,
    input_kind: Union[InputKind, str] = InputKind.SCC2020,
    backend: Union[Backend, str] = Backend.AUTO,
    output_path: Optional[PathLike] = None,
    twopac_binary: PathLike = "2pac",
    maxdim: int = 2,
    timeout: Optional[float] = None,
    verbose: bool = False,
) -> ExternalComputationResult:
    """Compute a minimal projective/free resolution using 2pac."""
    kind = InputKind(input_kind)
    selected_backend = _select_backend(kind, Backend(backend), twopac_binary)

    if selected_backend != Backend.TWOPAC:
        raise UnsupportedComputationError(f"Unsupported backend: {selected_backend}")
    return compute_twopac_resolution(
        input_path,
        input_kind=kind,
        output_path=output_path,
        twopac_binary=twopac_binary,
        maxdim=maxdim,
        timeout=timeout,
        verbose=verbose,
    )


def compute_minimal_injective_resolution(
    input_path: PathLike,
    *,
    dual_input_path: Optional[PathLike] = None,
    dualize_input: Optional[Callable[[Path], PathLike]] = None,
    input_kind: Union[InputKind, str] = InputKind.SCC2020,
    backend: Union[Backend, str] = Backend.AUTO,
    output_path: Optional[PathLike] = None,
    twopac_binary: PathLike = "2pac",
    maxdim: int = 2,
    timeout: Optional[float] = None,
    verbose: bool = False,
    samples: Optional[int] = None,
    chunkdim: Optional[int] = None,
    chain_chunk: bool = False,
    cone: Optional[int] = None,
    cone_first_parameter: Optional[int] = None,
    flip_grading: bool = False,
    sfd: Optional[str] = None,
    truncate: Optional[float] = None,
    extra_args: Optional[Iterable[str]] = None,
) -> ExternalComputationResult:
    """
    Compute an injective coresolution workflow result.

    With 2pac and no explicit dual input, this computes the selected homology
    module and saves its full minimal free resolution. For a finite-total-
    dimension module, use
    ``corollary14_injective_coresolution_from_free_resolution`` on that output
    to read the injective coresolution from Bauer-Lenzen-Lesnick Corollary 14.

    For modules with essential or coordinate-unbounded classes, pass ``cone``
    and/or ``cone_first_parameter`` so 2pac cones off the corresponding
    parameter directions before saving the homology resolution. The saved file
    may contain infinite grades; these are boundary-at-infinity labels in the
    coned/compactified convention.

    If an explicit dual/opposite input is provided, this wrapper computes a
    2pac projective/free resolution of that dual input and labels the result as
    injective source data for downstream interpretation.
    """
    if dual_input_path is not None and dualize_input is not None:
        raise ValueError("Use either dual_input_path or dualize_input, not both.")

    source_path = Path(input_path)
    kind = InputKind(input_kind)
    selected_backend = _select_backend(kind, Backend(backend), twopac_binary)

    if dual_input_path is None and dualize_input is None:
        return compute_twopac_injective_coresolution(
            source_path,
            input_kind=kind,
            output_path=output_path,
            twopac_binary=twopac_binary,
            maxdim=maxdim,
            samples=samples,
            chunkdim=chunkdim,
            chain_chunk=chain_chunk,
            cone=cone,
            cone_first_parameter=cone_first_parameter,
            flip_grading=flip_grading,
            sfd=sfd,
            truncate=truncate,
            timeout=timeout,
            verbose=verbose,
            extra_args=extra_args,
        )

    if dual_input_path is not None:
        computation_input = Path(dual_input_path)
    elif dualize_input is not None:
        computation_input = Path(dualize_input(source_path))

    result = compute_minimal_projective_resolution(
        computation_input,
        input_kind=kind,
        backend=backend,
        output_path=output_path,
        twopac_binary=twopac_binary,
        maxdim=maxdim,
        timeout=timeout,
        verbose=verbose,
    )

    return ExternalComputationResult(
        backend=result.backend,
        resolution_kind=ResolutionKind.INJECTIVE,
        input_kind=result.input_kind,
        input_path=source_path,
        output_path=result.output_path,
        output_format=result.output_format,
        command=result.command,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        elapsed_seconds=result.elapsed_seconds,
    )


def compute_twopac_injective_coresolution(
    input_path: PathLike,
    *,
    input_kind: Union[InputKind, str] = InputKind.SCC2020,
    output_path: Optional[PathLike] = None,
    twopac_binary: PathLike = "2pac",
    maxdim: int = 2,
    samples: Optional[int] = None,
    chunkdim: Optional[int] = None,
    chain_chunk: bool = False,
    cone: Optional[int] = None,
    cone_first_parameter: Optional[int] = None,
    flip_grading: bool = False,
    sfd: Optional[str] = None,
    truncate: Optional[float] = None,
    timeout: Optional[float] = None,
    verbose: bool = False,
    extra_args: Optional[Iterable[str]] = None,
) -> ExternalComputationResult:
    """
    Compute source data for an injective coresolution using 2pac homology.

    The saved scc2020 file is 2pac's minimal free resolution
    ``F_n -> ... -> F_0 -> M`` of the selected persistent homology module.
    For finite-total-dimension modules, Corollary 14 of
    Bauer-Lenzen-Lesnick reads this same matrix data as the injective
    coresolution ``Q^q = nu(F_{n-q})<epsilon>``. Use
    ``corollary14_injective_coresolution_from_free_resolution`` to get the
    shifted injective generator grades.

    If the selected homology module is not bounded at infinity, use 2pac's
    coning options. In the command line these are ``--cone`` for the second
    parameter and ``--Cone`` for the first parameter. Infinite grades in the
    output record those coned directions.
    """
    output_file = _resolve_output_path(output_path, suffix=".2pac.injective.scc2020")
    result = compute_twopac_resolution(
        input_path,
        input_kind=input_kind,
        output_path=output_file,
        twopac_binary=twopac_binary,
        algorithm=TwoPacAlgorithm.HOMOLOGY,
        maxdim=maxdim,
        samples=samples,
        chunkdim=chunkdim,
        chain_chunk=chain_chunk,
        cone=cone,
        cone_first_parameter=cone_first_parameter,
        flip_grading=flip_grading,
        sfd=sfd,
        truncate=truncate,
        timeout=timeout,
        verbose=verbose,
        extra_args=extra_args,
    )

    return ExternalComputationResult(
        backend=result.backend,
        resolution_kind=ResolutionKind.INJECTIVE,
        input_kind=result.input_kind,
        input_path=result.input_path,
        output_path=result.output_path,
        output_format=result.output_format,
        command=result.command,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        elapsed_seconds=result.elapsed_seconds,
    )


def compute_twopac_resolution(
    input_path: PathLike,
    *,
    input_kind: Union[InputKind, str] = InputKind.SCC2020,
    output_path: Optional[PathLike] = None,
    twopac_binary: PathLike = "2pac",
    algorithm: Union[TwoPacAlgorithm, str] = TwoPacAlgorithm.HOMOLOGY,
    maxdim: int = 2,
    samples: Optional[int] = None,
    chunkdim: Optional[int] = None,
    chain_chunk: bool = False,
    cone: Optional[int] = None,
    cone_first_parameter: Optional[int] = None,
    flip_grading: bool = False,
    sfd: Optional[str] = None,
    truncate: Optional[float] = None,
    timeout: Optional[float] = None,
    verbose: bool = False,
    extra_args: Optional[Iterable[str]] = None,
) -> ExternalComputationResult:
    """Call 2pac and save a top-dimensional free resolution in scc2020 format."""
    kind = InputKind(input_kind)
    algo = TwoPacAlgorithm(algorithm)
    input_file = _require_existing_file(input_path)
    output_file = _resolve_output_path(output_path, suffix=".2pac.resolution.scc2020")
    binary = _resolve_executable(twopac_binary)

    command = [
        binary,
        "-f",
        str(input_file),
        "--save-resolution-scc",
        str(output_file),
        "--dim",
        str(maxdim),
    ]

    if kind == InputKind.SCC2020:
        command.append("--scc-input")
    elif kind == InputKind.TWOPAC_MATRIX:
        command.append("--matrix-input")
    elif kind == InputKind.TWOPAC_CLIQUE:
        command.append("--clique-input")
    elif kind == InputKind.TWOPAC_FUNCTION_RIPS:
        pass
    else:
        raise UnsupportedComputationError(
            f"2pac supports scc2020, matrix, clique, and function-Rips input, not {kind.value}."
        )

    if algo == TwoPacAlgorithm.HOMOLOGY:
        command.append("--nocohomology")
    elif algo == TwoPacAlgorithm.COHOMOLOGY:
        command.append("--nohomology")
    elif algo == TwoPacAlgorithm.BOTH:
        pass

    if samples is not None:
        command.extend(["--samples", str(samples)])
    if chunkdim is not None:
        command.extend(["--chunkdim", str(chunkdim)])
    if chain_chunk:
        command.append("--chainchunk")
    if cone is not None:
        command.extend(["--cone", str(cone)])
    if cone_first_parameter is not None:
        command.extend(["--Cone", str(cone_first_parameter)])
    if flip_grading:
        command.append("--flip-grading")
    if sfd is not None:
        command.extend(["--sfd", sfd])
    if truncate is not None:
        command.extend(["--truncate", str(truncate)])
    if extra_args:
        command.extend(str(arg) for arg in extra_args)

    completed, elapsed = _run_command(command, timeout=timeout)
    return ExternalComputationResult(
        backend=Backend.TWOPAC,
        resolution_kind=ResolutionKind.PROJECTIVE,
        input_kind=kind,
        input_path=input_file,
        output_path=output_file,
        output_format="scc2020",
        command=tuple(command),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        elapsed_seconds=elapsed,
    )


def check_twopac_input(
    input_path: PathLike,
    *,
    input_kind: Union[InputKind, str] = InputKind.SCC2020,
    twopac_binary: PathLike = "2pac",
    timeout: Optional[float] = None,
) -> subprocess.CompletedProcess[str]:
    """Run 2pac's input checker."""
    kind = InputKind(input_kind)
    input_file = _require_existing_file(input_path)
    binary = _resolve_executable(twopac_binary)

    command = [binary, "-f", str(input_file), "--check"]
    if kind == InputKind.SCC2020:
        command.append("--scc-input")
    elif kind == InputKind.TWOPAC_MATRIX:
        command.append("--matrix-input")
    elif kind == InputKind.TWOPAC_CLIQUE:
        command.append("--clique-input")
    elif kind == InputKind.TWOPAC_FUNCTION_RIPS:
        pass
    else:
        raise UnsupportedComputationError(f"2pac cannot check {kind.value} input.")

    completed, _elapsed = _run_command(command, timeout=timeout)
    return completed


def _select_backend(
    input_kind: InputKind,
    backend: Backend,
    twopac_binary: PathLike,
) -> Backend:
    if input_kind == InputKind.FIREP:
        raise UnsupportedComputationError(
            "firep input is not supported by the 2pac-only resolution_compt.py. "
            "Convert the input to scc2020."
        )

    if backend == Backend.TWOPAC:
        return backend
    if backend != Backend.AUTO:
        raise UnsupportedComputationError(f"Unsupported backend in 2pac-only module: {backend.value}.")

    if is_executable_available(twopac_binary):
        return Backend.TWOPAC

    raise ExternalToolNotFoundError(
        "2pac is required for matrix, clique, and function-Rips input."
    )


def _require_existing_file(path: PathLike) -> Path:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Input path is not a file: {file_path}")
    return file_path


def _resolve_output_path(output_path: Optional[PathLike], *, suffix: str) -> Path:
    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        return output_file

    handle = tempfile.NamedTemporaryFile(prefix="presentation_", suffix=suffix, delete=False)
    try:
        return Path(handle.name)
    finally:
        handle.close()


def _run_command(
    command: Sequence[str],
    *,
    timeout: Optional[float] = None,
) -> tuple[subprocess.CompletedProcess[str], float]:
    start = time.monotonic()
    completed = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    elapsed = time.monotonic() - start

    if completed.returncode != 0:
        command_text = " ".join(command)
        raise ExternalToolError(
            f"External command failed with return code {completed.returncode}: "
            f"{command_text}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    return completed, elapsed
