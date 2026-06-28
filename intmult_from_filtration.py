"""Interval multiplicities from finite-grid filtration outputs.

This module implements the second step after interval candidate filtering:
evaluate the interval multiplicity formula only on retained candidates.  The
candidate generation code stays in ``finite_grid_interval_candidates.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from finite_grid_interval_candidates import (
    CandidateParseError,
    GridBounds,
    GridPoint,
    IntervalCandidate,
    IntervalCandidateComputation,
    PathLike,
    _interval_column_ranges,
    _parse_points_field,
    _parse_resolution_txt,
    _require_compressed_grid,
)
from utils import Interval


@dataclass(frozen=True)
class ProjectivePresentationData:
    """The minimal finite-grid projective presentation matrix ``P_1 -> P_0``."""

    bounds: GridBounds
    p1_grades: tuple[GridPoint, ...]
    p0_grades: tuple[GridPoint, ...]
    matrix_columns: tuple[tuple[int, ...], ...]
    field: str = "GF(2)"
    projective_path: Optional[Path] = None

    def __post_init__(self) -> None:
        p1_grades = _normalize_points_with_multiplicity(self.p1_grades)
        p0_grades = _normalize_points_with_multiplicity(self.p0_grades)
        if len(self.matrix_columns) != len(p1_grades):
            raise ValueError(
                "The P_1 -> P_0 matrix must have one sparse column per P_1 summand; "
                f"got {len(self.matrix_columns)} columns and {len(p1_grades)} P_1 grades."
            )
        normalized_columns: list[tuple[int, ...]] = []
        for column_index, rows in enumerate(self.matrix_columns, start=1):
            row_tuple = tuple(sorted(set(int(row) for row in rows)))
            for row in row_tuple:
                if row < 0 or row >= len(p0_grades):
                    raise ValueError(
                        f"Column {column_index} has row index {row}, outside 0..{len(p0_grades) - 1}."
                    )
            normalized_columns.append(row_tuple)
        for field_name, grades in (("p1_grades", p1_grades), ("p0_grades", p0_grades)):
            for grade in grades:
                if not self.bounds.contains(grade):
                    raise ValueError(f"{field_name} contains {grade}, outside {self.bounds}.")
        if self.field != "GF(2)":
            raise ValueError(f"Interval multiplicity computation currently supports GF(2), got {self.field}.")
        object.__setattr__(self, "p1_grades", p1_grades)
        object.__setattr__(self, "p0_grades", p0_grades)
        object.__setattr__(self, "matrix_columns", tuple(normalized_columns))


@dataclass(frozen=True)
class InjectiveCopresentationData:
    """The minimal finite-grid injective copresentation matrix ``Q^0 -> Q^1``."""

    bounds: GridBounds
    q0_grades: tuple[GridPoint, ...]
    q1_grades: tuple[GridPoint, ...]
    matrix_columns: tuple[tuple[int, ...], ...]
    field: str = "GF(2)"
    injective_path: Optional[Path] = None

    def __post_init__(self) -> None:
        q0_grades = _normalize_points_with_multiplicity(self.q0_grades)
        q1_grades = _normalize_points_with_multiplicity(self.q1_grades)
        if len(self.matrix_columns) != len(q0_grades):
            raise ValueError(
                "The Q^0 -> Q^1 matrix must have one sparse column per Q^0 summand; "
                f"got {len(self.matrix_columns)} columns and {len(q0_grades)} Q^0 grades."
            )
        normalized_columns: list[tuple[int, ...]] = []
        for column_index, rows in enumerate(self.matrix_columns, start=1):
            row_tuple = tuple(sorted(set(int(row) for row in rows)))
            for row in row_tuple:
                if row < 0 or row >= len(q1_grades):
                    raise ValueError(
                        f"Column {column_index} has row index {row}, outside 0..{len(q1_grades) - 1}."
                    )
            normalized_columns.append(row_tuple)
        for field_name, grades in (("q0_grades", q0_grades), ("q1_grades", q1_grades)):
            for grade in grades:
                if not self.bounds.contains(grade):
                    raise ValueError(f"{field_name} contains {grade}, outside {self.bounds}.")
        if self.field != "GF(2)":
            raise ValueError(f"Interval multiplicity computation currently supports GF(2), got {self.field}.")
        object.__setattr__(self, "q0_grades", q0_grades)
        object.__setattr__(self, "q1_grades", q1_grades)
        object.__setattr__(self, "matrix_columns", tuple(normalized_columns))


@dataclass(frozen=True)
class IntervalMultiplicity:
    """A retained candidate with nonzero computed interval multiplicity."""

    interval_index: int
    candidate: IntervalCandidate
    multiplicity: int
    rank_with_g3: int
    rank_without_g3: int


@dataclass(frozen=True)
class IntervalMultiplicityComputation:
    """Result returned by interval multiplicity computations."""

    presentation: ProjectivePresentationData | InjectiveCopresentationData
    candidate_count: int
    nonzero_multiplicities: tuple[IntervalMultiplicity, ...]
    output_path: Optional[Path] = None
    skipped: bool = False
    message: str = ""
    source_kind: str = "projective_presentation"
    source_path: Optional[Path] = None


@dataclass(frozen=True)
class IntervalMultiplicityComparison:
    """Comparison of projective-presentation and injective-copresentation results."""

    projective_result: IntervalMultiplicityComputation
    injective_result: IntervalMultiplicityComputation
    consistent: bool
    mismatches: tuple[tuple[int, int, int, IntervalCandidate], ...]
    output_path: Optional[Path] = None


def load_projective_presentation_data(projective_txt: PathLike) -> ProjectivePresentationData:
    """Load the minimal finite-grid projective presentation matrix ``P_1 -> P_0``."""

    projective = _parse_resolution_txt(projective_txt)
    _require_compressed_grid(projective)
    if len(projective.generator_grades) < 2 or len(projective.matrix_columns) < 1:
        raise CandidateParseError(f"{projective.path} does not contain a P_1 -> P_0 matrix block.")
    return ProjectivePresentationData(
        bounds=GridBounds(projective.grid_max),
        p1_grades=projective.generator_grades[-2],
        p0_grades=projective.generator_grades[-1],
        matrix_columns=projective.matrix_columns[-1],
        field=projective.field,
        projective_path=projective.path,
    )


def load_injective_copresentation_data(injective_txt: PathLike) -> InjectiveCopresentationData:
    """Load the minimal finite-grid injective copresentation matrix ``Q^0 -> Q^1``."""

    injective = _parse_resolution_txt(injective_txt)
    _require_compressed_grid(injective)
    if len(injective.generator_grades) < 2 or len(injective.matrix_columns) < 1:
        raise CandidateParseError(f"{injective.path} does not contain a Q^0 -> Q^1 matrix block.")
    return InjectiveCopresentationData(
        bounds=GridBounds(injective.grid_max),
        q0_grades=injective.generator_grades[0],
        q1_grades=injective.generator_grades[1],
        matrix_columns=injective.matrix_columns[0],
        field=injective.field,
        injective_path=injective.path,
    )


def compute_interval_multiplicities_from_candidate_result(
    candidate_result: IntervalCandidateComputation,
    output_path: Optional[PathLike] = None,
) -> IntervalMultiplicityComputation:
    """Compute interval multiplicities for the candidates retained in Step 1."""

    projective_path = candidate_result.support_data.projective_path
    if projective_path is None:
        raise ValueError("candidate_result does not record a projective resolution path.")
    return compute_interval_multiplicities(
        projective_path,
        candidate_result.candidates,
        output_path=output_path,
    )


def compute_interval_multiplicities_from_injective_candidate_result(
    candidate_result: IntervalCandidateComputation,
    injective_txt: Optional[PathLike] = None,
    output_path: Optional[PathLike] = None,
) -> IntervalMultiplicityComputation:
    """Compute interval multiplicities from the injective copresentation in Step 1."""

    injective_path = Path(injective_txt) if injective_txt is not None else candidate_result.support_data.injective_path
    if injective_path is None:
        raise ValueError("candidate_result does not record an injective coresolution path.")
    return compute_interval_multiplicities_from_injective_copresentation(
        injective_path,
        candidate_result.candidates,
        output_path=output_path,
    )


def compute_interval_multiplicities(
    projective_txt: PathLike,
    candidates: Iterable[IntervalCandidate],
    output_path: Optional[PathLike] = None,
) -> IntervalMultiplicityComputation:
    """Compute nonzero interval multiplicities from retained interval candidates.

    The implementation follows Theorem ``formula-pp(M)-icp(ass)`` and the fast
    block-matrix convention in Proposition ``fast-way-block-matrices`` from the
    local manuscript.  Only nonzero multiplicities are stored in the returned
    result and in the optional txt report.
    """

    presentation = load_projective_presentation_data(projective_txt)
    candidate_tuple = tuple(candidates)
    output = Path(output_path) if output_path is not None else _default_multiplicity_output_path(projective_txt)

    if not candidate_tuple:
        result = IntervalMultiplicityComputation(
            presentation=presentation,
            candidate_count=0,
            nonzero_multiplicities=(),
            output_path=output,
            skipped=True,
            message="No interval candidates were retained in Step 1; the persistence module has no interval summand candidates.",
            source_kind="projective_presentation",
            source_path=presentation.projective_path,
        )
        write_interval_multiplicities_text(result, output)
        return result

    nonzero: list[IntervalMultiplicity] = []
    for interval_index, candidate in enumerate(candidate_tuple, start=1):
        multiplicity, rank_with_g3, rank_without_g3 = _interval_multiplicity(candidate, presentation)
        if multiplicity:
            nonzero.append(
                IntervalMultiplicity(
                    interval_index=interval_index,
                    candidate=candidate,
                    multiplicity=multiplicity,
                    rank_with_g3=rank_with_g3,
                    rank_without_g3=rank_without_g3,
                )
            )

    message = (
        f"Computed multiplicities for {len(candidate_tuple)} interval candidates; "
        f"{len(nonzero)} have nonzero multiplicity."
    )
    if not nonzero:
        message += " No interval summand was detected among the retained candidates."
    result = IntervalMultiplicityComputation(
        presentation=presentation,
        candidate_count=len(candidate_tuple),
        nonzero_multiplicities=tuple(nonzero),
        output_path=output,
        skipped=False,
        message=message,
        source_kind="projective_presentation",
        source_path=presentation.projective_path,
    )
    write_interval_multiplicities_text(result, output)
    return result


def compute_interval_multiplicities_from_injective_copresentation(
    injective_txt: PathLike,
    candidates: Iterable[IntervalCandidate],
    output_path: Optional[PathLike] = None,
) -> IntervalMultiplicityComputation:
    """Compute nonzero interval multiplicities from an injective copresentation.

    This implements the dual formula in Theorem ``formula-icp(M)-pp(ass)``.
    The block ranks are built directly from the finite-grid copresentation
    matrix ``Q^0 -> Q^1`` over GF(2).
    """

    copresentation = load_injective_copresentation_data(injective_txt)
    candidate_tuple = tuple(candidates)
    output = (
        Path(output_path)
        if output_path is not None
        else _default_injective_multiplicity_output_path(injective_txt)
    )

    if not candidate_tuple:
        result = IntervalMultiplicityComputation(
            presentation=copresentation,
            candidate_count=0,
            nonzero_multiplicities=(),
            output_path=output,
            skipped=True,
            message="No interval candidates were retained in Step 1; the persistence module has no interval summand candidates.",
            source_kind="injective_copresentation",
            source_path=copresentation.injective_path,
        )
        write_interval_multiplicities_text(result, output)
        return result

    nonzero: list[IntervalMultiplicity] = []
    for interval_index, candidate in enumerate(candidate_tuple, start=1):
        multiplicity, rank_with_g3, rank_without_g3 = _interval_multiplicity_from_injective(
            candidate,
            copresentation,
        )
        if multiplicity:
            nonzero.append(
                IntervalMultiplicity(
                    interval_index=interval_index,
                    candidate=candidate,
                    multiplicity=multiplicity,
                    rank_with_g3=rank_with_g3,
                    rank_without_g3=rank_without_g3,
                )
            )

    message = (
        f"Computed multiplicities from injective copresentation for {len(candidate_tuple)} "
        f"interval candidates; {len(nonzero)} have nonzero multiplicity."
    )
    if not nonzero:
        message += " No interval summand was detected among the retained candidates."
    result = IntervalMultiplicityComputation(
        presentation=copresentation,
        candidate_count=len(candidate_tuple),
        nonzero_multiplicities=tuple(nonzero),
        output_path=output,
        skipped=False,
        message=message,
        source_kind="injective_copresentation",
        source_path=copresentation.injective_path,
    )
    write_interval_multiplicities_text(result, output)
    return result


def compare_interval_multiplicity_methods(
    candidate_result: IntervalCandidateComputation,
    *,
    projective_output_path: Optional[PathLike] = None,
    injective_output_path: Optional[PathLike] = None,
    comparison_output_path: Optional[PathLike] = None,
) -> IntervalMultiplicityComparison:
    """Compute multiplicities both ways and compare every candidate multiplicity."""

    projective_path = candidate_result.support_data.projective_path
    injective_path = candidate_result.support_data.injective_path
    if projective_path is None:
        raise ValueError("candidate_result does not record a projective resolution path.")
    if injective_path is None:
        raise ValueError("candidate_result does not record an injective coresolution path.")

    projective_result = compute_interval_multiplicities(
        projective_path,
        candidate_result.candidates,
        output_path=projective_output_path,
    )
    injective_result = compute_interval_multiplicities_from_injective_copresentation(
        injective_path,
        candidate_result.candidates,
        output_path=injective_output_path,
    )
    comparison_output = Path(comparison_output_path) if comparison_output_path is not None else None
    return compare_interval_multiplicity_results(
        projective_result,
        injective_result,
        candidates=candidate_result.candidates,
        output_path=comparison_output,
    )


def compare_interval_multiplicity_results(
    projective_result: IntervalMultiplicityComputation,
    injective_result: IntervalMultiplicityComputation,
    *,
    candidates: Sequence[IntervalCandidate],
    output_path: Optional[PathLike] = None,
) -> IntervalMultiplicityComparison:
    """Compare two interval multiplicity computations candidate-by-candidate."""

    projective_by_index = {
        record.interval_index: record.multiplicity for record in projective_result.nonzero_multiplicities
    }
    injective_by_index = {
        record.interval_index: record.multiplicity for record in injective_result.nonzero_multiplicities
    }
    mismatches = []
    for interval_index, candidate in enumerate(candidates, start=1):
        projective_multiplicity = projective_by_index.get(interval_index, 0)
        injective_multiplicity = injective_by_index.get(interval_index, 0)
        if projective_multiplicity != injective_multiplicity:
            mismatches.append(
                (
                    interval_index,
                    projective_multiplicity,
                    injective_multiplicity,
                    candidate,
                )
            )

    result = IntervalMultiplicityComparison(
        projective_result=projective_result,
        injective_result=injective_result,
        consistent=not mismatches,
        mismatches=tuple(mismatches),
        output_path=Path(output_path) if output_path is not None else None,
    )
    if result.output_path is not None:
        write_interval_multiplicity_comparison_text(result, result.output_path)
    return result


def load_interval_multiplicities_text(path: PathLike) -> IntervalMultiplicityComputation:
    """Load a txt file written by ``write_interval_multiplicities_text``.

    The saved file stores only nonzero intervals.  This loader reconstructs the
    compact interval candidates needed for downstream previews and
    visualization; it does not recompute any rank formula.
    """

    input_path = Path(path)
    raw_lines = input_path.read_text(encoding="utf-8").splitlines()
    header: dict[str, str] = {}
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("#"):
            break
        content = stripped[1:].strip()
        key, separator, value = content.partition(":")
        if separator:
            header[key.strip()] = value.strip()

    try:
        bounds = GridBounds(_parse_grid_max_header(header["grid max"]))
    except KeyError as exc:
        raise CandidateParseError(f"Missing grid max header in {input_path}.") from exc

    source_kind = header.get("source_kind", "projective_presentation")
    source_path = _optional_path_from_header(header.get("source_path"))
    presentation = _empty_presentation_for_loaded_multiplicities(source_kind, bounds, source_path)

    records: list[IntervalMultiplicity] = []
    current: dict[str, str] = {}

    def flush_current() -> None:
        if not current:
            return
        if "interval_index" not in current:
            raise CandidateParseError(f"Malformed interval record in {input_path}: missing interval index.")
        src = _parse_points_field(current.get("src", "-"))
        snk = _parse_points_field(current.get("snk", "-"))
        column_ranges = _interval_column_ranges(src, snk, bounds)
        if column_ranges is None:
            raise CandidateParseError(f"Interval in {input_path} has an empty hull: src={src}, snk={snk}.")
        candidate = IntervalCandidate(
            interval=Interval(src, snk),
            src=src,
            snk=snk,
            src_1_circ=_parse_points_field(current.get("src_1_circ", "-")),
            src_uuset=_parse_points_field(current.get("src_uuset", "-")),
            snk_1_circ=_parse_points_field(current.get("snk_1_circ", "-")),
            snk_ddset=_parse_points_field(current.get("snk_ddset", "-")),
            column_ranges=column_ranges,
            points=None,
        )
        records.append(
            IntervalMultiplicity(
                interval_index=int(current["interval_index"]),
                candidate=candidate,
                multiplicity=int(current.get("multiplicity", "0")),
                rank_with_g3=int(current.get("rank_with_g3", "0")),
                rank_without_g3=int(current.get("rank_without_g3", "0")),
            )
        )

    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("interval "):
            flush_current()
            current = {"interval_index": line.split(maxsplit=1)[1].strip()}
            continue
        key, separator, value = line.partition(":")
        if separator:
            current[key.strip()] = value.strip()
    flush_current()

    return IntervalMultiplicityComputation(
        presentation=presentation,
        candidate_count=int(header.get("candidate_intervals", len(records))),
        nonzero_multiplicities=tuple(records),
        output_path=input_path,
        skipped=_parse_bool_header(header.get("skipped", "False")),
        message=header.get("message", ""),
        source_kind=source_kind,
        source_path=source_path,
    )


def write_interval_multiplicities_text(
    result: IntervalMultiplicityComputation,
    path: PathLike,
) -> Path:
    """Write only nonzero interval multiplicities to a human-readable txt file."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    source = result.presentation
    lines = [
        "# finite_grid_interval_multiplicities",
        f"# source_kind: {result.source_kind}",
        f"# source_path: {result.source_path}",
        "# grid min: 0 0",
        f"# grid max: {_format_point_plain(source.bounds.max_grade)}",
        f"# candidate_intervals: {result.candidate_count}",
        f"# nonzero_intervals: {len(result.nonzero_multiplicities)}",
        f"# skipped: {result.skipped}",
        f"# message: {result.message}",
    ]

    for record in result.nonzero_multiplicities:
        candidate = record.candidate
        lines.extend(
            [
                "",
                f"interval {record.interval_index}",
                f"multiplicity: {record.multiplicity}",
                f"size: {candidate.size}",
                f"src: {_format_points(candidate.src)}",
                f"snk: {_format_points(candidate.snk)}",
                f"src_1_circ: {_format_points(candidate.src_1_circ)}",
                f"src_uuset: {_format_points(candidate.src_uuset)}",
                f"snk_1_circ: {_format_points(candidate.snk_1_circ)}",
                f"snk_ddset: {_format_points(candidate.snk_ddset)}",
            ]
        )

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def write_interval_multiplicity_comparison_text(
    result: IntervalMultiplicityComparison,
    path: PathLike,
) -> Path:
    """Write a summary comparing projective and injective multiplicity outputs."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# finite_grid_interval_multiplicity_comparison",
        f"# projective_output: {result.projective_result.output_path}",
        f"# injective_output: {result.injective_result.output_path}",
        f"# candidate_intervals: {result.projective_result.candidate_count}",
        f"# projective_nonzero: {len(result.projective_result.nonzero_multiplicities)}",
        f"# injective_nonzero: {len(result.injective_result.nonzero_multiplicities)}",
        f"# consistent: {result.consistent}",
        f"# mismatches: {len(result.mismatches)}",
    ]
    for interval_index, projective_multiplicity, injective_multiplicity, candidate in result.mismatches:
        lines.extend(
            [
                "",
                f"interval {interval_index}",
                f"projective_multiplicity: {projective_multiplicity}",
                f"injective_multiplicity: {injective_multiplicity}",
                f"src: {_format_points(candidate.src)}",
                f"snk: {_format_points(candidate.snk)}",
            ]
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


@dataclass(frozen=True)
class _CoefficientMatrix:
    row_labels: tuple[GridPoint, ...]
    col_labels: tuple[GridPoint, ...]
    rows: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.rows) != len(self.row_labels):
            raise ValueError("Coefficient matrix row count does not match row labels.")
        column_count = len(self.col_labels)
        for row in self.rows:
            if row < 0 or row.bit_length() > column_count:
                raise ValueError("Coefficient matrix row contains a column outside the label range.")


def _interval_multiplicity(
    candidate: IntervalCandidate,
    presentation: ProjectivePresentationData,
) -> tuple[int, int, int]:
    if len(candidate.column_ranges) != presentation.bounds.max_grade[0] + 1:
        raise ValueError("Candidate column ranges do not match the projective presentation grid.")

    g1 = _g1_coefficient_matrix(candidate)
    g2 = _g2_coefficient_matrix(candidate)
    g3 = _g3_coefficient_matrix(candidate)
    rank_with_g3 = _multiplicity_block_rank(presentation, g1, g2, g3, include_g3=True)
    rank_without_g3 = _multiplicity_block_rank(presentation, g1, g2, g3, include_g3=False)
    multiplicity = rank_with_g3 - rank_without_g3
    if multiplicity < 0:
        raise ValueError(
            "Computed a negative interval multiplicity. This indicates an inconsistent block construction."
        )
    return multiplicity, rank_with_g3, rank_without_g3


def _interval_multiplicity_from_injective(
    candidate: IntervalCandidate,
    copresentation: InjectiveCopresentationData,
) -> tuple[int, int, int]:
    if len(candidate.column_ranges) != copresentation.bounds.max_grade[0] + 1:
        raise ValueError("Candidate column ranges do not match the injective copresentation grid.")

    g1 = _g1_coefficient_matrix(candidate)
    g2 = _g2_coefficient_matrix(candidate)
    g3 = _g3_coefficient_matrix(candidate)
    rank_with_g3 = _injective_multiplicity_block_rank(copresentation, g1, g2, g3, include_g3=True)
    rank_without_g3 = _injective_multiplicity_block_rank(copresentation, g1, g2, g3, include_g3=False)
    multiplicity = rank_with_g3 - rank_without_g3
    if multiplicity < 0:
        raise ValueError(
            "Computed a negative interval multiplicity from injective data. "
            "This indicates an inconsistent block construction."
        )
    return multiplicity, rank_with_g3, rank_without_g3


def _g1_coefficient_matrix(candidate: IntervalCandidate) -> _CoefficientMatrix:
    row_labels = candidate.src_1_circ + candidate.src_uuset
    col_labels = candidate.src
    rows: list[int] = []

    for row_label in candidate.src_1_circ:
        row_bits = 0
        for col_index, source in enumerate(col_labels):
            if _leq(source, row_label):
                row_bits ^= 1 << col_index
        rows.append(row_bits)

    for row_label in candidate.src_uuset:
        choice = _choose_source_for_proper_source(row_label, col_labels)
        rows.append(1 << col_labels.index(choice))

    return _CoefficientMatrix(row_labels=row_labels, col_labels=col_labels, rows=tuple(rows))


def _g2_coefficient_matrix(candidate: IntervalCandidate) -> _CoefficientMatrix:
    row_labels = candidate.snk
    col_labels = candidate.snk_ddset + candidate.snk_1_circ
    rows = [0 for _ in row_labels]

    for col_index, col_label in enumerate(candidate.snk_ddset):
        choice = _choose_sink_for_proper_sink(col_label, row_labels)
        rows[row_labels.index(choice)] ^= 1 << col_index

    sink_1_offset = len(candidate.snk_ddset)
    for offset, col_label in enumerate(candidate.snk_1_circ):
        col_index = sink_1_offset + offset
        for row_index, sink in enumerate(row_labels):
            if _leq(col_label, sink):
                rows[row_index] ^= 1 << col_index

    return _CoefficientMatrix(row_labels=row_labels, col_labels=col_labels, rows=tuple(rows))


def _g3_coefficient_matrix(candidate: IntervalCandidate) -> _CoefficientMatrix:
    row_labels = candidate.snk
    col_labels = candidate.src
    rows = [0 for _ in row_labels]
    sink, source = _choose_sink_source_path(candidate)
    rows[row_labels.index(sink)] = 1 << col_labels.index(source)
    return _CoefficientMatrix(row_labels=row_labels, col_labels=col_labels, rows=tuple(rows))


def _choose_source_for_proper_source(
    proper_source: GridPoint,
    sources: Sequence[GridPoint],
) -> GridPoint:
    for source in sources:
        if _leq(source, proper_source):
            return source
    raise ValueError(f"No source in {sources!r} is <= proper source {proper_source}.")


def _choose_sink_for_proper_sink(
    proper_sink: GridPoint,
    sinks: Sequence[GridPoint],
) -> GridPoint:
    for sink in sinks:
        if _leq(proper_sink, sink):
            return sink
    raise ValueError(f"No sink in {sinks!r} is >= proper sink {proper_sink}.")


def _choose_sink_source_path(candidate: IntervalCandidate) -> tuple[GridPoint, GridPoint]:
    for sink in candidate.snk:
        for source in candidate.src:
            if _leq(source, sink):
                return sink, source
    raise ValueError(f"Interval candidate has no comparable source-sink pair: {candidate!r}.")


def _multiplicity_block_rank(
    presentation: ProjectivePresentationData,
    g1: _CoefficientMatrix,
    g2: _CoefficientMatrix,
    g3: _CoefficientMatrix,
    *,
    include_g3: bool,
) -> int:
    x_grades = presentation.p0_grades
    y_grades = presentation.p1_grades
    top_labels = g1.row_labels
    bottom_labels = g2.row_labels

    top_row_count = len(x_grades) * len(top_labels)
    bottom_row_count = len(x_grades) * len(bottom_labels)
    row_count = top_row_count + bottom_row_count

    group_1_cols = len(x_grades) * len(g1.col_labels)
    group_2_cols = len(x_grades) * len(g2.col_labels)
    group_3_cols = len(y_grades) * len(top_labels)
    group_4_cols = len(y_grades) * len(bottom_labels)
    group_1_offset = 0
    group_2_offset = group_1_offset + group_1_cols
    group_3_offset = group_2_offset + group_2_cols
    group_4_offset = group_3_offset + group_3_cols

    rows = [0 for _ in range(row_count)]
    _add_pprime_g_block(rows, 0, group_1_offset, x_grades, g1)
    _add_pprime_alpha_block(rows, 0, group_3_offset, top_labels, presentation)
    if include_g3:
        _add_pprime_g_block(rows, top_row_count, group_1_offset, x_grades, g3)
    _add_pprime_g_block(rows, top_row_count, group_2_offset, x_grades, g2)
    _add_pprime_alpha_block(rows, top_row_count, group_4_offset, bottom_labels, presentation)
    return _gf2_rank_from_rows(rows)


def _injective_multiplicity_block_rank(
    copresentation: InjectiveCopresentationData,
    g1: _CoefficientMatrix,
    g2: _CoefficientMatrix,
    g3: _CoefficientMatrix,
    *,
    include_g3: bool,
) -> int:
    x_grades = copresentation.q0_grades
    y_grades = copresentation.q1_grades
    source_labels = g1.col_labels
    top_labels = g1.row_labels
    sink_labels = g2.row_labels
    bottom_labels = g2.col_labels

    top_row_count = len(x_grades) * len(source_labels)
    bottom_row_count = len(x_grades) * len(bottom_labels)
    row_count = top_row_count + bottom_row_count

    group_1_cols = len(x_grades) * len(top_labels)
    group_2_cols = len(x_grades) * len(sink_labels)
    group_3_cols = len(y_grades) * len(source_labels)
    group_4_cols = len(y_grades) * len(bottom_labels)
    group_1_offset = 0
    group_2_offset = group_1_offset + group_1_cols
    group_3_offset = group_2_offset + group_2_cols
    group_4_offset = group_3_offset + group_3_cols

    rows = [0 for _ in range(row_count)]
    _add_p_g_block(rows, 0, group_1_offset, x_grades, g1)
    if include_g3:
        _add_p_g_block(rows, 0, group_2_offset, x_grades, g3)
    _add_p_alpha_block(rows, 0, group_3_offset, source_labels, copresentation)
    _add_p_g_block(rows, top_row_count, group_2_offset, x_grades, g2)
    _add_p_alpha_block(rows, top_row_count, group_4_offset, bottom_labels, copresentation)
    return _gf2_rank_from_rows(rows)


def _add_pprime_g_block(
    rows: list[int],
    row_offset: int,
    column_offset: int,
    x_grades: Sequence[GridPoint],
    matrix: _CoefficientMatrix,
) -> None:
    row_stride = len(matrix.row_labels)
    column_stride = len(matrix.col_labels)
    if not row_stride or not column_stride:
        return
    for x_index, x_grade in enumerate(x_grades):
        block_row_offset = row_offset + x_index * row_stride
        block_column_offset = column_offset + x_index * column_stride
        for local_row, row_bits in enumerate(matrix.rows):
            value = row_bits
            while value:
                low_bit = value & -value
                local_col = low_bit.bit_length() - 1
                if _leq(x_grade, matrix.col_labels[local_col]):
                    rows[block_row_offset + local_row] ^= 1 << (block_column_offset + local_col)
                value ^= low_bit


def _add_p_g_block(
    rows: list[int],
    row_offset: int,
    column_offset: int,
    x_grades: Sequence[GridPoint],
    matrix: _CoefficientMatrix,
) -> None:
    row_stride = len(matrix.col_labels)
    column_stride = len(matrix.row_labels)
    if not row_stride or not column_stride:
        return
    for x_index, x_grade in enumerate(x_grades):
        block_row_offset = row_offset + x_index * row_stride
        block_column_offset = column_offset + x_index * column_stride
        for local_col, row_bits in enumerate(matrix.rows):
            if not _leq(matrix.row_labels[local_col], x_grade):
                continue
            value = row_bits
            while value:
                low_bit = value & -value
                local_row = low_bit.bit_length() - 1
                rows[block_row_offset + local_row] ^= 1 << (block_column_offset + local_col)
                value ^= low_bit


def _add_pprime_alpha_block(
    rows: list[int],
    row_offset: int,
    column_offset: int,
    target_labels: Sequence[GridPoint],
    presentation: ProjectivePresentationData,
) -> None:
    label_count = len(target_labels)
    if not label_count:
        return
    for y_index, y_grade in enumerate(presentation.p1_grades):
        active_labels = [
            label_index for label_index, target_label in enumerate(target_labels) if _leq(y_grade, target_label)
        ]
        if not active_labels:
            continue
        for x_index in presentation.matrix_columns[y_index]:
            row_base = row_offset + x_index * label_count
            column_base = column_offset + y_index * label_count
            for label_index in active_labels:
                rows[row_base + label_index] ^= 1 << (column_base + label_index)


def _add_p_alpha_block(
    rows: list[int],
    row_offset: int,
    column_offset: int,
    target_labels: Sequence[GridPoint],
    copresentation: InjectiveCopresentationData,
) -> None:
    label_count = len(target_labels)
    if not label_count:
        return
    for x_index, y_rows in enumerate(copresentation.matrix_columns):
        row_base = row_offset + x_index * label_count
        for y_index in y_rows:
            y_grade = copresentation.q1_grades[y_index]
            column_base = column_offset + y_index * label_count
            for label_index, target_label in enumerate(target_labels):
                if _leq(target_label, y_grade):
                    rows[row_base + label_index] ^= 1 << (column_base + label_index)


def _gf2_rank_from_rows(rows: Iterable[int]) -> int:
    basis = _GF2IncrementalRank()
    for row in rows:
        basis.add(row)
    return basis.rank


class _GF2IncrementalRank:
    """Incremental row-echelon rank for integer bit-vectors over GF(2)."""

    def __init__(self) -> None:
        self._basis_by_pivot: dict[int, int] = {}
        self.rank = 0

    def add(self, vector: int) -> bool:
        value = vector
        while value:
            pivot = value.bit_length() - 1
            basis_vector = self._basis_by_pivot.get(pivot)
            if basis_vector is None:
                self._basis_by_pivot[pivot] = value
                self.rank += 1
                return True
            value ^= basis_vector
        return False


def _default_multiplicity_output_path(projective_txt: PathLike) -> Path:
    path = Path(projective_txt)
    prefix = _output_prefix_from_resolution_path(path, ".finite_grid.projective_resolution.txt")
    return Path("outputs") / "interval_multiplicities" / f"{prefix}.interval_multiplicities.projective.txt"


def _default_injective_multiplicity_output_path(injective_txt: PathLike) -> Path:
    path = Path(injective_txt)
    prefix = _output_prefix_from_resolution_path(path, ".finite_grid.injective_coresolution.txt")
    return Path("outputs") / "interval_multiplicities" / f"{prefix}.interval_multiplicities.injective.txt"


def _output_prefix_from_resolution_path(path: Path, suffix: str) -> str:
    name = path.name
    prefix = name[: -len(suffix)] if name.endswith(suffix) else path.stem
    return prefix[: -len(".scc2020")] if prefix.endswith(".scc2020") else prefix


def _parse_grid_max_header(value: str) -> GridPoint:
    tokens = value.split()
    if len(tokens) != 2:
        raise CandidateParseError(f"Expected grid max header with two integers, got {value!r}.")
    return (int(tokens[0]), int(tokens[1]))


def _optional_path_from_header(value: Optional[str]) -> Optional[Path]:
    if value is None or value in {"", "None"}:
        return None
    return Path(value)


def _parse_bool_header(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise CandidateParseError(f"Expected boolean header value, got {value!r}.")


def _empty_presentation_for_loaded_multiplicities(
    source_kind: str,
    bounds: GridBounds,
    source_path: Optional[Path],
) -> ProjectivePresentationData | InjectiveCopresentationData:
    if source_kind == "injective_copresentation":
        return InjectiveCopresentationData(
            bounds=bounds,
            q0_grades=(),
            q1_grades=(),
            matrix_columns=(),
            injective_path=source_path,
        )
    return ProjectivePresentationData(
        bounds=bounds,
        p1_grades=(),
        p0_grades=(),
        matrix_columns=(),
        projective_path=source_path,
    )


def _as_grid_point(point: Sequence[int]) -> GridPoint:
    if len(point) != 2:
        raise ValueError(f"Expected a 2D grid point, got {point!r}.")
    return (int(point[0]), int(point[1]))


def _normalize_points_with_multiplicity(points: Iterable[Sequence[int]]) -> tuple[GridPoint, ...]:
    return tuple(_as_grid_point(point) for point in points)


def _leq(left: GridPoint, right: GridPoint) -> bool:
    return left[0] <= right[0] and left[1] <= right[1]


def _format_point_plain(point: GridPoint) -> str:
    return f"{point[0]} {point[1]}"


def _format_point(point: GridPoint) -> str:
    return f"({point[0]}, {point[1]})"


def _format_points(points: Sequence[GridPoint]) -> str:
    if not points:
        return "-"
    return " ".join(_format_point(point) for point in points)
