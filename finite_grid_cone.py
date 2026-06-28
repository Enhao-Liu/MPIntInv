"""
Finite-grid boundary coning without modifying 2pac.

This module treats 2pac as an external executable. It can parse a complete
scc2020 chain complex, build the finite-boundary coned chain complex over
GF(2), call 2pac on the temporary coned scc2020 file, and extract finite-grid
projective and injective outputs.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Iterable, Optional, Sequence, Union


PathLike = Union[str, Path]
Grade = tuple[int | float, int | float]


@dataclass(frozen=True)
class SccColumn:
    """One sparse GF(2) column with zero-based row indices."""

    grade: Grade
    rows: tuple[int, ...]


@dataclass(frozen=True)
class ChainMatrix:
    """A graded sparse boundary matrix C_q -> C_{q-1}."""

    row_grades: tuple[Grade, ...]
    column_grades: tuple[Grade, ...]
    columns: tuple[tuple[int, ...], ...]

    @property
    def rows(self) -> int:
        return len(self.row_grades)

    @property
    def cols(self) -> int:
        return len(self.column_grades)


@dataclass(frozen=True)
class MaterializedChainComplex:
    """A finite graded chain complex, indexed by homological dimension."""

    term_grades: tuple[tuple[Grade, ...], ...]
    boundaries: tuple[Optional[ChainMatrix], ...]
    field: str = "GF(2)"
    path: Optional[Path] = None

    @property
    def max_dim(self) -> int:
        return len(self.term_grades) - 1


@dataclass(frozen=True)
class GridCompression:
    """Exact coordinate compression to [0,n] x [0,m]."""

    x_values: tuple[int | float, ...]
    y_values: tuple[int | float, ...]

    @property
    def n(self) -> int:
        return len(self.x_values) - 1

    @property
    def m(self) -> int:
        return len(self.y_values) - 1

    @property
    def upper(self) -> Grade:
        return (self.n, self.m)

    @property
    def coned_upper(self) -> Grade:
        return (self.n + 1, self.m + 1)

    def compress_grade(self, grade: Grade) -> Grade:
        try:
            return (self.x_values.index(grade[0]), self.y_values.index(grade[1]))
        except ValueError as exc:
            raise ValueError(f"Grade {grade} was not present in the compression data.") from exc

    def to_dict(self) -> dict:
        return {
            "original_x_values": [_json_number(value) for value in self.x_values],
            "original_y_values": [_json_number(value) for value in self.y_values],
            "n": self.n,
            "m": self.m,
            "lower": [0, 0],
            "upper": [self.n, self.m],
            "artificial_right_coordinate": self.n + 1,
            "artificial_top_coordinate": self.m + 1,
            "artificial_corner_coordinate": [self.n + 1, self.m + 1],
        }


@dataclass(frozen=True)
class CoresolutionData:
    """Finite-grid injective coresolution Q^0 -> ... -> Q^n."""

    terms: tuple[tuple[Grade, ...], ...]
    matrices: tuple[ChainMatrix, ...]
    field: str
    epsilon: Grade


@dataclass(frozen=True)
class FiniteGridConeResult:
    """Paths and structured data produced by the finite-grid cone workflow."""

    input_path: Path
    input_kind: str
    target_dim: int
    compression: GridCompression
    coned_complex_path: Path
    zero_extended_resolution_path: Optional[Path]
    metadata_path: Optional[Path]
    projective_json_path: Optional[Path]
    projective_txt_path: Path
    injective_json_path: Optional[Path]
    injective_txt_path: Optional[Path]
    projective_resolution: MaterializedChainComplex
    injective_coresolution: Optional[CoresolutionData]
    commands: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class _SccHeader:
    num_parameters: int
    sizes: tuple[int, ...]
    field: str
    reverse_coordinates: tuple[int, ...]
    data_lines: tuple[str, ...]


@dataclass(frozen=True)
class _BasisDescriptor:
    kind: int
    source_dim: int
    source_index: int
    grade: Grade


def run_finite_grid_cone_workflow(
    input_path: PathLike,
    *,
    input_kind: str = "scc2020",
    output_dir: PathLike = "outputs/resolution_computation",
    output_prefix: Optional[str] = None,
    twopac_binary: PathLike = "2pac",
    target_dim: int = 1,
    save_injective: bool = True,
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
    Compute finite-grid projective/free and optional injective outputs.

    Complete scc2020 inputs are parsed directly. Other 2pac-supported input
    kinds are first exported with ``--save-complex-scc``; current 2pac releases
    do not include final row grades in that export, so this path requires the
    export to be complete.

    The finite-boundary cone uses the reduced convention in degree zero. To
    keep 2pac unmodified, the wrapper calls 2pac's cohomology pipeline
    (``--nohomology``), which is the stable CLI path for saving the corresponding
    homology free resolution of the coned complex.
    """
    source_path = _require_existing_file(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix or source_path.stem

    binary = _resolve_executable(twopac_binary)
    commands: list[tuple[str, ...]] = []

    if input_kind == "scc2020":
        source_complex_path = source_path
    else:
        source_complex_path = out_dir / f"{prefix}.materialized_complex.txt"
        command = _twopac_save_complex_command(
            binary=binary,
            input_path=source_path,
            input_kind=input_kind,
            output_path=source_complex_path,
            maxdim=target_dim + 1,
            samples=samples,
            flip_grading=flip_grading,
            sfd=sfd,
            truncate=truncate,
        )
        _run_command(command, timeout=timeout, verbose=verbose)
        commands.append(tuple(command))

    chain_complex = read_scc2020_chain_complex(source_complex_path)
    compression = compute_grid_compression(chain_complex)
    compressed = compress_chain_complex(chain_complex, compression)
    coned = finite_boundary_cone(compressed, compression, target_dim=target_dim)

    coned_complex_path = out_dir / f"{prefix}.finite_grid_coned_complex.txt"
    write_scc2020_chain_complex(
        coned,
        coned_complex_path,
        comments=(
            "finite_grid_coned_chain_complex",
            f"original_grid_upper: {compression.n} {compression.m}",
            f"coned_grid_upper: {compression.n + 1} {compression.m + 1}",
        ),
    )

    metadata_path = None
    if save_json:
        metadata_path = out_dir / f"{prefix}.finite_grid_metadata.json"
        metadata_path.write_text(json.dumps(compression.to_dict(), indent=2) + "\n", encoding="utf-8")

    zero_extended_resolution_path = (
        out_dir / f"{prefix}.finite_grid_zero_extended_free_resolution.txt"
    )
    resolution_command = [
        binary,
        "-f",
        str(coned_complex_path),
        "--scc-input",
        "--nohomology",
        "--dim",
        str(target_dim),
        "--save-resolution-scc",
        str(zero_extended_resolution_path),
    ]
    if chunkdim is not None:
        resolution_command.extend(["--chunkdim", str(chunkdim)])
    if chain_chunk:
        resolution_command.append("--chainchunk")
    _run_command(resolution_command, timeout=timeout, verbose=verbose)
    commands.append(tuple(resolution_command))

    zero_extended_resolution = read_scc2020_chain_complex(zero_extended_resolution_path)
    saved_zero_extended_resolution_path: Optional[Path] = zero_extended_resolution_path
    if not keep_zero_extended_resolution:
        zero_extended_resolution_path.unlink(missing_ok=True)
        saved_zero_extended_resolution_path = None

    projective = restrict_projective_resolution_to_grid(zero_extended_resolution, compression)
    projective_json_path = None
    projective_txt_path = out_dir / f"{prefix}.finite_grid.projective_resolution.txt"
    if save_json:
        projective_json_path = out_dir / f"{prefix}.finite_grid.projective_resolution.json"
        write_projective_resolution_json(projective, compression, projective_json_path)
    write_projective_resolution_text(projective, compression, projective_txt_path)

    injective = None
    injective_json_path = None
    injective_txt_path = None
    if save_injective:
        injective = finite_grid_injective_from_free_resolution(zero_extended_resolution, compression)
        injective_txt_path = out_dir / f"{prefix}.finite_grid.injective_coresolution.txt"
        if save_json:
            injective_json_path = out_dir / f"{prefix}.finite_grid.injective_coresolution.json"
            write_injective_coresolution_json(injective, compression, injective_json_path)
        write_injective_coresolution_text(injective, compression, injective_txt_path)

    return FiniteGridConeResult(
        input_path=source_path,
        input_kind=input_kind,
        target_dim=target_dim,
        compression=compression,
        coned_complex_path=coned_complex_path,
        zero_extended_resolution_path=saved_zero_extended_resolution_path,
        metadata_path=metadata_path,
        projective_json_path=projective_json_path,
        projective_txt_path=projective_txt_path,
        injective_json_path=injective_json_path,
        injective_txt_path=injective_txt_path,
        projective_resolution=projective,
        injective_coresolution=injective,
        commands=tuple(commands),
    )


def read_scc2020_chain_complex(
    path: PathLike,
    *,
    require_final_grade_block: bool = True,
) -> MaterializedChainComplex:
    """Read a complete scc2020 chain complex into low-to-high dimension order."""
    source_path = Path(path)
    header = _read_scc2020_header(source_path)
    if header.num_parameters != 2:
        raise ValueError("Finite-grid cone currently supports exactly two parameters.")

    max_dim = len(header.sizes) - 1
    expected_without_final = sum(header.sizes[:-1])
    expected_with_final = sum(header.sizes)
    actual = len(header.data_lines)
    if actual not in {expected_without_final, expected_with_final}:
        raise ValueError(
            f"Unexpected scc2020 data line count in {source_path}: got {actual}, "
            f"expected {expected_without_final} without final grades or {expected_with_final} with them."
        )
    if require_final_grade_block and actual != expected_with_final:
        raise ValueError(
            "The finite-grid cone workflow needs the final row-grade block. "
            f"The file {source_path} has no final grade block."
        )

    term_grades: list[Optional[tuple[Grade, ...]]] = [None] * (max_dim + 1)
    raw_columns_by_dim: dict[int, tuple[tuple[int, ...], ...]] = {}
    offset = 0

    for block_index, source_size in enumerate(header.sizes[:-1]):
        source_dim = max_dim - block_index
        target_size = header.sizes[block_index + 1]
        grades: list[Grade] = []
        columns: list[tuple[int, ...]] = []
        for _local_column in range(source_size):
            grade, rows = _parse_scc2020_column(
                header.data_lines[offset],
                header.num_parameters,
                target_size,
                header.reverse_coordinates,
            )
            grades.append(grade)
            columns.append(rows)
            offset += 1
        term_grades[source_dim] = tuple(grades)
        raw_columns_by_dim[source_dim] = tuple(columns)

    if actual == expected_with_final:
        final_grades = []
        for _ in range(header.sizes[-1]):
            final_grades.append(
                _parse_scc2020_grade_line(
                    header.data_lines[offset],
                    header.num_parameters,
                    header.reverse_coordinates,
                )
            )
            offset += 1
        term_grades[0] = tuple(final_grades)
    elif term_grades[0] is None:
        term_grades[0] = tuple()

    finalized_terms: list[tuple[Grade, ...]] = []
    for dim, grades in enumerate(term_grades):
        if grades is None:
            raise ValueError(f"Missing term grades for chain dimension {dim}.")
        finalized_terms.append(grades)

    boundaries: list[Optional[ChainMatrix]] = [None] * (max_dim + 1)
    for source_dim, columns in raw_columns_by_dim.items():
        target_dim = source_dim - 1
        boundaries[source_dim] = ChainMatrix(
            row_grades=finalized_terms[target_dim],
            column_grades=finalized_terms[source_dim],
            columns=columns,
        )

    complex_ = MaterializedChainComplex(
        term_grades=tuple(finalized_terms),
        boundaries=tuple(boundaries),
        field=header.field,
        path=source_path,
    )
    validate_chain_complex(complex_)
    return complex_


def compute_grid_compression(complex_: MaterializedChainComplex) -> GridCompression:
    """Compute exact coordinate compression from all basis grades."""
    x_values: set[int | float] = set()
    y_values: set[int | float] = set()
    for term in complex_.term_grades:
        for grade in term:
            _require_finite_grade(grade)
            x_values.add(grade[0])
            y_values.add(grade[1])
    if not x_values or not y_values:
        raise ValueError("Cannot compress an empty set of grades.")
    return GridCompression(
        x_values=tuple(sorted(x_values)),
        y_values=tuple(sorted(y_values)),
    )


def compress_chain_complex(
    complex_: MaterializedChainComplex,
    compression: GridCompression,
) -> MaterializedChainComplex:
    """Replace all grades by their integer compression ranks."""
    compressed_terms = tuple(
        tuple(compression.compress_grade(grade) for grade in term)
        for term in complex_.term_grades
    )
    boundaries: list[Optional[ChainMatrix]] = [None] * len(complex_.boundaries)
    for dim, matrix in enumerate(complex_.boundaries):
        if matrix is None:
            continue
        boundaries[dim] = ChainMatrix(
            row_grades=compressed_terms[dim - 1],
            column_grades=compressed_terms[dim],
            columns=matrix.columns,
        )
    compressed = MaterializedChainComplex(
        term_grades=compressed_terms,
        boundaries=tuple(boundaries),
        field=complex_.field,
        path=complex_.path,
    )
    validate_chain_complex(compressed)
    return compressed


def finite_boundary_cone(
    complex_: MaterializedChainComplex,
    compression: GridCompression,
    *,
    target_dim: int,
) -> MaterializedChainComplex:
    """Build C_hat_q = C_q + C_{q-1}^x + C_{q-1}^y + C_{q-2}^{xy}."""
    max_coned_dim = target_dim + 1
    unsorted_terms: list[list[_BasisDescriptor]] = []
    for q in range(max_coned_dim + 1):
        descriptors: list[_BasisDescriptor] = []
        descriptors.extend(_basis_descriptors(complex_, q, 0, compression))
        descriptors.extend(_basis_descriptors(complex_, q - 1, 1, compression))
        descriptors.extend(_basis_descriptors(complex_, q - 1, 2, compression))
        descriptors.extend(_basis_descriptors(complex_, q - 2, 3, compression))
        unsorted_terms.append(descriptors)

    unsorted_boundaries: list[Optional[tuple[tuple[int, ...], ...]]] = [None]
    for q in range(1, max_coned_dim + 1):
        unsorted_boundaries.append(_build_coned_boundary_columns(complex_, compression, q))

    orders: list[list[int]] = [
        sorted(range(len(term)), key=lambda index, term=term: _descriptor_sort_key(term[index]))
        for term in unsorted_terms
    ]
    old_to_new = [
        {old_index: new_index for new_index, old_index in enumerate(order)}
        for order in orders
    ]
    sorted_terms = tuple(
        tuple(term[old_index].grade for old_index in order)
        for term, order in zip(unsorted_terms, orders)
    )

    sorted_boundaries: list[Optional[ChainMatrix]] = [None] * (max_coned_dim + 1)
    for q in range(1, max_coned_dim + 1):
        raw_columns = unsorted_boundaries[q]
        assert raw_columns is not None
        sorted_columns = []
        row_map = old_to_new[q - 1]
        for old_column_index in orders[q]:
            rows = sorted(row_map[row] for row in raw_columns[old_column_index])
            sorted_columns.append(tuple(rows))
        sorted_boundaries[q] = ChainMatrix(
            row_grades=sorted_terms[q - 1],
            column_grades=sorted_terms[q],
            columns=tuple(sorted_columns),
        )

    coned = MaterializedChainComplex(
        term_grades=sorted_terms,
        boundaries=tuple(sorted_boundaries),
        field=complex_.field,
    )
    validate_chain_complex(coned)
    return coned


def restrict_projective_resolution_to_grid(
    resolution: MaterializedChainComplex,
    compression: GridCompression,
) -> MaterializedChainComplex:
    """Keep only free summands whose grades lie in the original finite grid."""
    keep = tuple(
        tuple(index for index, grade in enumerate(term) if _grade_in_grid(grade, compression))
        for term in resolution.term_grades
    )
    restricted_terms = tuple(
        tuple(term[index] for index in term_keep)
        for term, term_keep in zip(resolution.term_grades, keep)
    )
    boundaries: list[Optional[ChainMatrix]] = [None] * len(resolution.boundaries)
    for dim, matrix in enumerate(resolution.boundaries):
        if matrix is None:
            continue
        boundaries[dim] = _restrict_matrix(
            matrix,
            row_keep=keep[dim - 1],
            col_keep=keep[dim],
            row_grades=restricted_terms[dim - 1],
            column_grades=restricted_terms[dim],
        )
    restricted = MaterializedChainComplex(
        term_grades=restricted_terms,
        boundaries=tuple(boundaries),
        field=resolution.field,
        path=resolution.path,
    )
    validate_chain_complex(restricted)
    return restricted


def finite_grid_injective_from_free_resolution(
    resolution: MaterializedChainComplex,
    compression: GridCompression,
    *,
    epsilon: Grade = (1, 1),
) -> CoresolutionData:
    """Read the finite-grid injective coresolution from a full free resolution."""
    n = resolution.max_dim
    shifted_terms = tuple(
        tuple(_shift_grade(grade, epsilon) for grade in resolution.term_grades[n - q])
        for q in range(n + 1)
    )
    keep = tuple(
        tuple(index for index, grade in enumerate(term) if _grade_in_grid(grade, compression))
        for term in shifted_terms
    )
    restricted_terms = tuple(
        tuple(term[index] for index in term_keep)
        for term, term_keep in zip(shifted_terms, keep)
    )

    matrices: list[ChainMatrix] = []
    for q in range(n):
        free_boundary_dim = n - q
        matrix = resolution.boundaries[free_boundary_dim]
        if matrix is None:
            raise ValueError(f"Missing free-resolution matrix in dimension {free_boundary_dim}.")
        matrices.append(
            _restrict_matrix(
                matrix,
                row_keep=keep[q + 1],
                col_keep=keep[q],
                row_grades=restricted_terms[q + 1],
                column_grades=restricted_terms[q],
            )
        )

    _validate_coresolution_sequence(tuple(matrices))
    return CoresolutionData(
        terms=restricted_terms,
        matrices=tuple(matrices),
        field=resolution.field,
        epsilon=epsilon,
    )


def write_scc2020_chain_complex(
    complex_: MaterializedChainComplex,
    path: PathLike,
    *,
    comments: Sequence[str] = (),
) -> Path:
    """Write a chain complex in scc2020 sparse-column text format."""
    terms_high_to_low = tuple(reversed(complex_.term_grades))
    matrices_high_to_low = tuple(
        complex_.boundaries[dim]
        for dim in range(complex_.max_dim, 0, -1)
        if complex_.boundaries[dim] is not None
    )
    return _write_scc2020_blocks(
        terms_high_to_low,
        matrices_high_to_low,
        path,
        field=complex_.field,
        comments=comments,
    )


def write_projective_resolution_text(
    resolution: MaterializedChainComplex,
    compression: GridCompression,
    path: PathLike,
) -> Path:
    """Write finite-grid projective/free output as readable scc2020-style txt."""
    terms_high_to_low = tuple(reversed(resolution.term_grades))
    matrices_high_to_low = tuple(
        resolution.boundaries[dim]
        for dim in range(resolution.max_dim, 0, -1)
        if resolution.boundaries[dim] is not None
    )
    return _write_scc2020_blocks(
        terms_high_to_low,
        matrices_high_to_low,
        path,
        field=resolution.field,
        comments=(
            "finite_grid_projective_resolution",
            "summand_kind: P_G",
            "minimal_expected: true",
            "minimal_verified_by_2pac_before_grid_extraction: true",
            f"grid min: 0 0",
            f"grid max: {compression.n} {compression.m}",
        ),
    )


def write_injective_coresolution_text(
    coresolution: CoresolutionData,
    compression: GridCompression,
    path: PathLike,
) -> Path:
    """Write finite-grid injective output as readable scc2020-style txt."""
    return _write_scc2020_blocks(
        coresolution.terms,
        coresolution.matrices,
        path,
        field=coresolution.field,
        comments=(
            "finite_grid_injective_coresolution",
            "summand_kind: Q_G",
            "minimal_expected: true",
            f"epsilon: {coresolution.epsilon[0]} {coresolution.epsilon[1]}",
            f"grid min: 0 0",
            f"grid max: {compression.n} {compression.m}",
            "semantic note: matrix blocks are ordered b0, b1, ... for Q^0 -> Q^1 -> ... .",
        ),
    )


def write_projective_resolution_json(
    resolution: MaterializedChainComplex,
    compression: GridCompression,
    path: PathLike,
) -> Path:
    """Write finite-grid projective/free output as structured JSON."""
    data = {
        "type": "finite_grid_projective_resolution",
        "minimal_expected": True,
        "field": resolution.field,
        "grid": compression.to_dict(),
        "terms": [
            {
                "degree": degree,
                "summand_kind": "P_G",
                "grades": [_json_grade(grade) for grade in resolution.term_grades[degree]],
            }
            for degree in range(resolution.max_dim + 1)
        ],
        "differentials": [
            _matrix_to_json(
                name=f"R{degree - 1}",
                from_degree=degree,
                to_degree=degree - 1,
                matrix=resolution.boundaries[degree],
            )
            for degree in range(1, resolution.max_dim + 1)
        ],
    }
    return _write_json(path, data)


def write_injective_coresolution_json(
    coresolution: CoresolutionData,
    compression: GridCompression,
    path: PathLike,
) -> Path:
    """Write finite-grid injective output as structured JSON."""
    data = {
        "type": "finite_grid_injective_coresolution",
        "minimal_expected": True,
        "field": coresolution.field,
        "epsilon": _json_grade(coresolution.epsilon),
        "grid": compression.to_dict(),
        "terms": [
            {
                "degree": degree,
                "summand_kind": "Q_G",
                "grades": [_json_grade(grade) for grade in grades],
            }
            for degree, grades in enumerate(coresolution.terms)
        ],
        "differentials": [
            _matrix_to_json(
                name=f"B{degree}",
                from_degree=degree,
                to_degree=degree + 1,
                matrix=matrix,
            )
            for degree, matrix in enumerate(coresolution.matrices)
        ],
    }
    return _write_json(path, data)


def validate_chain_complex(complex_: MaterializedChainComplex) -> None:
    """Check matrix shapes, grading, and d^2=0 over GF(2)."""
    for dim, matrix in enumerate(complex_.boundaries):
        if dim == 0:
            if matrix is not None:
                raise ValueError("Boundary in dimension 0 must be absent.")
            continue
        if matrix is None:
            if complex_.term_grades[dim]:
                raise ValueError(f"Missing boundary matrix in dimension {dim}.")
            continue
        if matrix.row_grades != complex_.term_grades[dim - 1]:
            raise ValueError(f"Boundary row grades do not match C_{dim - 1}.")
        if matrix.column_grades != complex_.term_grades[dim]:
            raise ValueError(f"Boundary column grades do not match C_{dim}.")
        if len(matrix.columns) != len(matrix.column_grades):
            raise ValueError(f"Boundary in dimension {dim} has wrong column count.")
        for column_index, rows in enumerate(matrix.columns):
            if len(set(rows)) != len(rows):
                raise ValueError(f"Repeated row in boundary d_{dim}, column {column_index}.")
            for row in rows:
                if row < 0 or row >= len(matrix.row_grades):
                    raise ValueError(f"Row {row} outside d_{dim} row range.")
                if not _grade_leq(matrix.row_grades[row], matrix.column_grades[column_index]):
                    raise ValueError(
                        f"Boundary d_{dim} is not grade-compatible at column {column_index}, row {row}."
                    )

    matrices = tuple(matrix for matrix in complex_.boundaries[1:] if matrix is not None)
    _validate_matrix_sequence(matrices)


def _read_scc2020_header(path: Path) -> _SccHeader:
    records = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        content = raw_line.partition("#")[0].strip()
        if content:
            records.append(content)
    if len(records) < 3 or records[0] != "scc2020":
        raise ValueError(f"{path} is not a valid scc2020 file.")
    num_parameters = int(records[1])
    index = 2
    reverse_coordinates: tuple[int, ...] = ()
    field = "GF(2)"
    sizes: Optional[tuple[int, ...]] = None
    if records[index].startswith("--"):
        while index < len(records) and records[index].startswith("--"):
            reverse_coordinates, field = _parse_header_option(
                records[index],
                reverse_coordinates,
                field,
            )
            index += 1
        sizes = tuple(int(token) for token in records[index].split())
        index += 1
    else:
        sizes = tuple(int(token) for token in records[index].split())
        index += 1
        while index < len(records) and records[index].startswith("--"):
            reverse_coordinates, field = _parse_header_option(
                records[index],
                reverse_coordinates,
                field,
            )
            index += 1
    if field != "GF(2)":
        raise ValueError(f"Only GF(2) is supported, got {field}.")
    return _SccHeader(
        num_parameters=num_parameters,
        sizes=sizes,
        field=field,
        reverse_coordinates=reverse_coordinates,
        data_lines=tuple(records[index:]),
    )


def _parse_header_option(
    line: str,
    reverse_coordinates: tuple[int, ...],
    field: str,
) -> tuple[tuple[int, ...], str]:
    tokens = line.split()
    if tokens[0] == "--reverse":
        return tuple(int(token) for token in tokens[1:]), field
    if tokens[0] == "--field":
        return reverse_coordinates, " ".join(tokens[1:])
    return reverse_coordinates, field


def _parse_scc2020_column(
    line: str,
    num_parameters: int,
    target_size: int,
    reverse_coordinates: Sequence[int],
) -> tuple[Grade, tuple[int, ...]]:
    tokens = line.replace(";", " ; ").split()
    grade = _parse_scc2020_grade(tokens[:num_parameters], reverse_coordinates)
    entries = tokens[num_parameters:]
    if entries and entries[0] == ";":
        entries = entries[1:]
    rows: set[int] = set()
    for entry in entries:
        if entry == ";":
            continue
        row_token, separator, coeff_token = entry.partition(":")
        row = int(row_token) - 1
        coeff = int(coeff_token) if separator else 1
        if row < 0 or row >= target_size:
            raise ValueError(f"Row index {row + 1} is outside 1..{target_size}.")
        if coeff % 2:
            if row in rows:
                rows.remove(row)
            else:
                rows.add(row)
    return grade, tuple(sorted(rows))


def _parse_scc2020_grade_line(
    line: str,
    num_parameters: int,
    reverse_coordinates: Sequence[int],
) -> Grade:
    tokens = line.split()
    if len(tokens) != num_parameters:
        raise ValueError(f"Expected {num_parameters} grade coordinates, got {line!r}.")
    return _parse_scc2020_grade(tokens, reverse_coordinates)


def _parse_scc2020_grade(tokens: Sequence[str], reverse_coordinates: Sequence[int]) -> Grade:
    values = [_parse_coordinate(token) for token in tokens]
    for coordinate_index in reverse_coordinates:
        index = coordinate_index - 1
        values[index] = -values[index]
    return (values[0], values[1])


def _parse_coordinate(token: str) -> int | float:
    lowered = token.lower()
    if lowered in {"inf", "+inf", "infinity", "+infinity", "-inf", "-infinity"}:
        raise ValueError(f"Finite-grid compression does not accept infinite coordinate {token!r}.")
    try:
        return int(token)
    except ValueError:
        value = float(token)
        if value.is_integer():
            return int(value)
        return value


def _basis_descriptors(
    complex_: MaterializedChainComplex,
    source_dim: int,
    kind: int,
    compression: GridCompression,
) -> list[_BasisDescriptor]:
    if source_dim < 0 or source_dim > complex_.max_dim:
        return []
    descriptors = []
    for source_index, grade in enumerate(complex_.term_grades[source_dim]):
        if kind == 0:
            new_grade = grade
        elif kind == 1:
            new_grade = (compression.n + 1, grade[1])
        elif kind == 2:
            new_grade = (grade[0], compression.m + 1)
        elif kind == 3:
            new_grade = (compression.n + 1, compression.m + 1)
        else:
            raise AssertionError(f"Unknown basis descriptor kind: {kind}")
        descriptors.append(_BasisDescriptor(kind, source_dim, source_index, new_grade))
    return descriptors


def _build_coned_boundary_columns(
    complex_: MaterializedChainComplex,
    compression: GridCompression,
    q: int,
) -> tuple[tuple[int, ...], ...]:
    n_q = _term_size(complex_, q)
    n_q_1 = _term_size(complex_, q - 1)
    n_q_2 = _term_size(complex_, q - 2)
    n_q_3 = _term_size(complex_, q - 3)

    row_orig_offset = 0
    row_x_offset = n_q_1
    row_y_offset = n_q_1 + n_q_2
    row_xy_offset = n_q_1 + 2 * n_q_2
    row_count = n_q_1 + 2 * n_q_2 + n_q_3

    columns: list[tuple[int, ...]] = []

    d_q = _boundary_columns(complex_, q)
    d_q_1 = _boundary_columns(complex_, q - 1)
    d_q_2 = _boundary_columns(complex_, q - 2)

    for sigma in range(n_q):
        rows = _toggle_rows(row_orig_offset + row for row in d_q[sigma])
        columns.append(_checked_rows(rows, row_count))

    for tau in range(n_q_1):
        rows = {row_orig_offset + tau}
        rows = _toggle_rows((row_x_offset + row for row in d_q_1[tau]), rows)
        columns.append(_checked_rows(rows, row_count))

    for tau in range(n_q_1):
        rows = {row_orig_offset + tau}
        rows = _toggle_rows((row_y_offset + row for row in d_q_1[tau]), rows)
        columns.append(_checked_rows(rows, row_count))

    for upsilon in range(n_q_2):
        rows = {row_x_offset + upsilon, row_y_offset + upsilon}
        rows = _toggle_rows((row_xy_offset + row for row in d_q_2[upsilon]), rows)
        columns.append(_checked_rows(rows, row_count))

    expected_cols = n_q + 2 * n_q_1 + n_q_2
    if len(columns) != expected_cols:
        raise AssertionError(f"Expected {expected_cols} columns in d_hat_{q}, got {len(columns)}.")
    return tuple(columns)


def _term_size(complex_: MaterializedChainComplex, dim: int) -> int:
    if dim < 0 or dim > complex_.max_dim:
        return 0
    return len(complex_.term_grades[dim])


def _boundary_columns(complex_: MaterializedChainComplex, dim: int) -> tuple[tuple[int, ...], ...]:
    if dim < 0 or dim > complex_.max_dim:
        return tuple()
    if dim == 0:
        return tuple(() for _ in range(_term_size(complex_, dim)))
    matrix = complex_.boundaries[dim]
    if matrix is None:
        return tuple(() for _ in range(_term_size(complex_, dim)))
    return matrix.columns


def _toggle_rows(
    rows: Iterable[int],
    initial: Optional[set[int]] = None,
) -> set[int]:
    out = set() if initial is None else set(initial)
    for row in rows:
        if row in out:
            out.remove(row)
        else:
            out.add(row)
    return out


def _checked_rows(rows: set[int], row_count: int) -> tuple[int, ...]:
    for row in rows:
        if row < 0 or row >= row_count:
            raise ValueError(f"Constructed row {row} outside row count {row_count}.")
    return tuple(sorted(rows))


def _descriptor_sort_key(descriptor: _BasisDescriptor) -> tuple[float, float, int, int, int]:
    grade = descriptor.grade
    return (float(grade[1]), float(grade[0]), descriptor.kind, descriptor.source_dim, descriptor.source_index)


def _restrict_matrix(
    matrix: ChainMatrix,
    *,
    row_keep: Sequence[int],
    col_keep: Sequence[int],
    row_grades: Sequence[Grade],
    column_grades: Sequence[Grade],
) -> ChainMatrix:
    row_map = {old_index: new_index for new_index, old_index in enumerate(row_keep)}
    columns = []
    for old_column in col_keep:
        rows = tuple(
            row_map[row]
            for row in matrix.columns[old_column]
            if row in row_map
        )
        columns.append(tuple(sorted(rows)))
    return ChainMatrix(
        row_grades=tuple(row_grades),
        column_grades=tuple(column_grades),
        columns=tuple(columns),
    )


def _validate_matrix_sequence(matrices: Sequence[ChainMatrix]) -> None:
    for first, second in zip(matrices, matrices[1:]):
        if first.column_grades != second.row_grades:
            raise ValueError("Adjacent matrix terms do not have matching grades.")
        if not _composition_is_zero(first, second):
            raise ValueError("Adjacent matrix composition is nonzero.")


def _validate_coresolution_sequence(matrices: Sequence[ChainMatrix]) -> None:
    for first, second in zip(matrices, matrices[1:]):
        if first.row_grades != second.column_grades:
            raise ValueError("Adjacent coresolution matrix terms do not have matching grades.")
        if not _composition_is_zero(second, first):
            raise ValueError("Adjacent coresolution matrix composition is nonzero.")


def _composition_is_zero(first: ChainMatrix, second: ChainMatrix) -> bool:
    """Return whether first * second is zero, with first after second."""
    for second_column in second.columns:
        rows: set[int] = set()
        for middle_row in second_column:
            for target_row in first.columns[middle_row]:
                if target_row in rows:
                    rows.remove(target_row)
                else:
                    rows.add(target_row)
        if rows:
            return False
    return True


def _grade_leq(left: Grade, right: Grade) -> bool:
    return left[0] <= right[0] and left[1] <= right[1]


def _grade_in_grid(grade: Grade, compression: GridCompression) -> bool:
    return 0 <= grade[0] <= compression.n and 0 <= grade[1] <= compression.m


def _shift_grade(grade: Grade, epsilon: Grade) -> Grade:
    return (_normalize_number(grade[0] - epsilon[0]), _normalize_number(grade[1] - epsilon[1]))


def _require_finite_grade(grade: Grade) -> None:
    for coordinate in grade:
        if not isinstance(coordinate, (int, float)) or not math.isfinite(float(coordinate)):
            raise ValueError(f"Finite-grid compression requires finite numeric grades, got {grade}.")


def _write_scc2020_blocks(
    terms: Sequence[Sequence[Grade]],
    matrices: Sequence[ChainMatrix],
    path: PathLike,
    *,
    field: str,
    comments: Sequence[str] = (),
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "scc2020",
        "2",
        " ".join(str(len(term)) for term in terms),
        f"--field {field}",
    ]
    lines.extend(f"# {comment}" for comment in comments)
    for block_index, matrix in enumerate(matrices, start=1):
        lines.append(f"# matrix block {block_index}")
        for grade, rows in zip(matrix.column_grades, matrix.columns):
            line = f"{_format_grade(grade)} ;"
            if rows:
                line += " " + " ".join(str(row + 1) for row in rows)
            lines.append(line)
    lines.append("# Row grades for final term")
    if terms:
        lines.extend(_format_grade(grade) for grade in terms[-1])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _matrix_to_json(
    *,
    name: str,
    from_degree: int,
    to_degree: int,
    matrix: Optional[ChainMatrix],
) -> dict:
    if matrix is None:
        columns: list[list[int]] = []
    else:
        columns = [list(column) for column in matrix.columns]
    return {
        "name": name,
        "from_degree": from_degree,
        "to_degree": to_degree,
        "matrix_format": "sparse_columns",
        "index_base": 0,
        "columns": columns,
    }


def _write_json(path: PathLike, data: dict) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return out


def _format_grade(grade: Grade) -> str:
    return f"{_format_number(grade[0])} {_format_number(grade[1])}"


def _format_number(value: int | float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _json_grade(grade: Grade) -> list[int | float]:
    return [_json_number(grade[0]), _json_number(grade[1])]


def _json_number(value: int | float) -> int | float:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _normalize_number(value: int | float) -> int | float:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _twopac_save_complex_command(
    *,
    binary: str,
    input_path: Path,
    input_kind: str,
    output_path: Path,
    maxdim: int,
    samples: Optional[int],
    flip_grading: bool,
    sfd: Optional[str],
    truncate: Optional[float],
) -> list[str]:
    command = [
        binary,
        "-f",
        str(input_path),
        "--save-complex-scc",
        str(output_path),
        "--dim",
        str(maxdim),
    ]
    if input_kind == "scc2020":
        command.append("--scc-input")
    elif input_kind == "2pac-matrix":
        command.append("--matrix-input")
    elif input_kind == "2pac-clique":
        command.append("--clique-input")
    elif input_kind == "2pac-function-rips":
        pass
    else:
        raise ValueError(f"Unsupported 2pac input kind for finite-grid cone: {input_kind}.")
    if samples is not None:
        command.extend(["--samples", str(samples)])
    if flip_grading:
        command.append("--flip-grading")
    if sfd is not None:
        command.extend(["--sfd", sfd])
    if truncate is not None:
        command.extend(["--truncate", str(truncate)])
    return command


def _run_command(
    command: Sequence[str],
    *,
    timeout: Optional[float],
    verbose: bool,
) -> subprocess.CompletedProcess[str]:
    started = time.monotonic()
    completed = subprocess.run(
        list(command),
        check=False,
        capture_output=not verbose,
        text=True,
        timeout=timeout,
    )
    _elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(
            "External command failed with exit code "
            f"{completed.returncode}: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def _resolve_executable(binary: PathLike) -> str:
    binary_path = Path(binary)
    if binary_path.parent != Path("."):
        if binary_path.exists() and os.access(binary_path, os.X_OK):
            return str(binary_path)
        raise FileNotFoundError(f"Executable not found: {binary_path}")
    resolved = which(str(binary))
    if resolved is not None:
        return resolved
    for bin_dir in (Path(sys.executable).parent, Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin")):
        candidate = bin_dir / str(binary)
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise FileNotFoundError(f"Executable not found: {binary}")


def _require_existing_file(path: PathLike) -> Path:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Input path is not a file: {file_path}")
    return file_path
