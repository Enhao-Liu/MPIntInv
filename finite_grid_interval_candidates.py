"""Candidate interval filtering for compressed finite-grid modules.

This module implements the first, conservative filtering step for interval
multiplicity computations. It reads the finite-grid projective and injective
outputs produced by Scheme 1, extracts the support data used in the critical
set criteria, and keeps only intervals that can still have nonzero interval
multiplicity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence

from intappx import (
    get_prop_src_snk_Rfree,
    get_prop_src_snk_from_extrema_Rfree,
    get_src_snk_Rfree,
    int_hull_Rfree,
)
from utils import Interval, Representation

GridPoint = tuple[int, int]
PathLike = str | Path
ZeroPairOracle = Callable[[GridPoint, GridPoint], bool]
POINT_PATTERN = re.compile(r"\((-?\d+),\s*(-?\d+)\)")


class CandidateParseError(ValueError):
    """Raised when finite-grid resolution txt input cannot be parsed."""


class CandidateGenerationLimitError(RuntimeError):
    """Raised when candidate generation exceeds an explicit safety limit."""


@dataclass(frozen=True)
class GridBounds:
    """Compressed 2D grid bounds with implicit minimum (0, 0)."""

    max_grade: GridPoint

    def __post_init__(self) -> None:
        max_grade = _as_grid_point(self.max_grade)
        if max_grade[0] < 0 or max_grade[1] < 0:
            raise ValueError(f"Grid max must be nonnegative, got {max_grade}.")
        object.__setattr__(self, "max_grade", max_grade)

    @property
    def min_grade(self) -> GridPoint:
        return (0, 0)

    @property
    def dimensions(self) -> tuple[int, int]:
        return (self.max_grade[0] + 1, self.max_grade[1] + 1)

    def contains(self, point: GridPoint) -> bool:
        point = _as_grid_point(point)
        return 0 <= point[0] <= self.max_grade[0] and 0 <= point[1] <= self.max_grade[1]

    def representation(self) -> Representation:
        return Representation(list(self.dimensions))


@dataclass(frozen=True)
class ResolutionSupportData:
    """Supports needed for the crt0 and 2D-grid crt1 interval criteria."""

    bounds: GridBounds
    top_p0_support: tuple[GridPoint, ...]
    top_p1_support: tuple[GridPoint, ...]
    soc_q0_support: tuple[GridPoint, ...]
    soc_q1_support: tuple[GridPoint, ...]
    projective_path: Optional[Path] = None
    injective_path: Optional[Path] = None

    def __post_init__(self) -> None:
        for field_name in (
            "top_p0_support",
            "top_p1_support",
            "soc_q0_support",
            "soc_q1_support",
        ):
            points = _normalize_points(getattr(self, field_name))
            for point in points:
                if not self.bounds.contains(point):
                    raise ValueError(f"{field_name} contains {point}, outside {self.bounds}.")
            object.__setattr__(self, field_name, points)

    def representation(self) -> Representation:
        return self.bounds.representation()


@dataclass(frozen=True)
class IntervalCandidate:
    """A candidate interval with cached criterion data."""

    interval: Interval
    src: tuple[GridPoint, ...]
    snk: tuple[GridPoint, ...]
    src_1_circ: tuple[GridPoint, ...]
    src_uuset: tuple[GridPoint, ...]
    snk_1_circ: tuple[GridPoint, ...]
    snk_ddset: tuple[GridPoint, ...]
    column_ranges: tuple[Optional[tuple[int, int]], ...]
    points: Optional[tuple[GridPoint, ...]] = None

    @property
    def size(self) -> int:
        return _column_ranges_size(self.column_ranges)


@dataclass(frozen=True)
class IntervalCandidateComputation:
    """Result returned by compute_interval_candidates."""

    support_data: ResolutionSupportData
    candidates: tuple[IntervalCandidate, ...]
    generated_count: int
    retained_count: int
    criterion: str
    generation_mode: str
    socle_filter_r: Optional[int] = None
    radical_filter_r: Optional[int] = None
    output_path: Optional[Path] = None


@dataclass(frozen=True)
class SubmoduleIntervalFamily:
    """A socle-layer constrained family of interval submodules.

    The sinks are a possible ``max(I)``.  Each source candidate ``a`` satisfies
    the top restriction ``a in supp(soc^(h_I(a)+1) M / soc^h_I(a) M)`` and has
    the property that the rectangles from ``a`` to all comparable sinks stay
    inside the socle-layer allowed region.  Concrete intervals in the family are
    obtained by choosing a canonical connected source antichain from these
    candidates.
    """

    sinks: tuple[GridPoint, ...]
    source_candidates: tuple[GridPoint, ...]


@dataclass(frozen=True)
class SubmoduleIntervalFamilyComputation:
    """Socle-layer interval families for possible submodules ``V_I -> M``."""

    bounds: GridBounds
    r: int
    sink_antichain_count: int
    families: tuple[SubmoduleIntervalFamily, ...]
    output_path: Optional[Path] = None

    @property
    def retained_family_count(self) -> int:
        return len(self.families)


@dataclass(frozen=True)
class RankFilteredSubmoduleInterval:
    """A concrete interval passing the injective-copresentation rank test."""

    family_index: int
    interval: "ConcreteSubmoduleInterval"


@dataclass(frozen=True)
class ConcreteSubmoduleInterval:
    """A concrete source/sink interval with only the data needed downstream."""

    src: tuple[GridPoint, ...]
    snk: tuple[GridPoint, ...]
    column_ranges: tuple[Optional[tuple[int, int]], ...]

    @property
    def interval(self) -> Interval:
        return Interval(self.src, self.snk)

    @property
    def size(self) -> int:
        return _column_ranges_size(self.column_ranges)


@dataclass(frozen=True)
class RankFilteredSubmoduleIntervalComputation:
    """Concrete submodule candidates retained by the rank necessary test."""

    bounds: GridBounds
    source_mode: str
    tested_intervals: int
    retained_intervals: tuple[RankFilteredSubmoduleInterval, ...]
    retained_interval_count: Optional[int] = None
    output_path: Optional[Path] = None

    @property
    def retained_count(self) -> int:
        if self.retained_interval_count is not None:
            return self.retained_interval_count
        return len(self.retained_intervals)


@dataclass(frozen=True)
class ProjectivePresentationData:
    """The first finite-grid projective presentation matrix P_1 -> P_0."""

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
            raise ValueError(f"Higher radical support computation currently supports GF(2), got {self.field}.")
        object.__setattr__(self, "p1_grades", p1_grades)
        object.__setattr__(self, "p0_grades", p0_grades)
        object.__setattr__(self, "matrix_columns", tuple(normalized_columns))


@dataclass(frozen=True)
class InjectiveCopresentationData:
    """The first finite-grid injective copresentation matrix Q^0 -> Q^1."""

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
            raise ValueError(f"Higher socle support computation currently supports GF(2), got {self.field}.")
        object.__setattr__(self, "q0_grades", q0_grades)
        object.__setattr__(self, "q1_grades", q1_grades)
        object.__setattr__(self, "matrix_columns", tuple(normalized_columns))


@dataclass(frozen=True)
class SocleLayerSupport:
    """Support and pointwise dimensions for one socle layer."""

    layer: int
    support: tuple[GridPoint, ...]
    dimensions: tuple[tuple[GridPoint, int], ...]

    def dimension_dict(self) -> dict[GridPoint, int]:
        return dict(self.dimensions)


@dataclass(frozen=True)
class SocleLayerSupportComputation:
    """Socle-layer supports computed from an injective copresentation."""

    copresentation: InjectiveCopresentationData
    max_layer: int
    layers: tuple[SocleLayerSupport, ...]

    def layer(self, layer: int) -> SocleLayerSupport:
        for support in self.layers:
            if support.layer == layer:
                return support
        raise KeyError(f"No socle layer {layer} in this computation.")


@dataclass(frozen=True)
class RadicalLayerSupport:
    """Support and pointwise dimensions for one radical layer."""

    layer: int
    support: tuple[GridPoint, ...]
    dimensions: tuple[tuple[GridPoint, int], ...]

    def dimension_dict(self) -> dict[GridPoint, int]:
        return dict(self.dimensions)


@dataclass(frozen=True)
class RadicalLayerSupportComputation:
    """Radical-layer supports computed from a projective presentation."""

    presentation: ProjectivePresentationData
    max_layer: int
    layers: tuple[RadicalLayerSupport, ...]

    def layer(self, layer: int) -> RadicalLayerSupport:
        for support in self.layers:
            if support.layer == layer:
                return support
        raise KeyError(f"No radical layer {layer} in this computation.")


@dataclass(frozen=True)
class _ParsedResolution:
    path: Path
    num_parameters: int
    generator_sizes: tuple[int, ...]
    generator_grades: tuple[tuple[GridPoint, ...], ...]
    matrix_columns: tuple[tuple[tuple[int, ...], ...], ...]
    grid_min: GridPoint
    grid_max: GridPoint
    field: str = "GF(2)"


@dataclass(frozen=True)
class _ColumnRangeSummary:
    column_ranges: tuple[Optional[tuple[int, int]], ...]
    sources: tuple[GridPoint, ...]
    sinks: tuple[GridPoint, ...]
    connected: bool


@dataclass(frozen=True)
class _ColumnRangeBounds:
    min_x: int
    max_x: int
    min_y: int
    max_y: int


def load_resolution_support_data(
    projective_txt: PathLike,
    injective_txt: PathLike,
) -> ResolutionSupportData:
    """Load crt support data from Scheme 1 finite-grid txt outputs."""

    projective = _parse_resolution_txt(projective_txt)
    injective = _parse_resolution_txt(injective_txt)

    _require_compressed_grid(projective)
    _require_compressed_grid(injective)
    if projective.grid_max != injective.grid_max:
        raise CandidateParseError(
            f"Projective grid max {projective.grid_max} does not match "
            f"injective grid max {injective.grid_max}."
        )

    if len(projective.generator_grades) < 1:
        raise CandidateParseError("Projective resolution has no terms.")
    if len(injective.generator_grades) < 1:
        raise CandidateParseError("Injective coresolution has no terms.")

    return ResolutionSupportData(
        bounds=GridBounds(projective.grid_max),
        top_p0_support=projective.generator_grades[-1],
        top_p1_support=projective.generator_grades[-2]
        if len(projective.generator_grades) >= 2
        else (),
        soc_q0_support=injective.generator_grades[0],
        soc_q1_support=injective.generator_grades[1]
        if len(injective.generator_grades) >= 2
        else (),
        projective_path=projective.path,
        injective_path=injective.path,
    )


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


def compute_radical_layer_supports_from_projective_txt(
    projective_txt: PathLike,
    max_layer: int,
    *,
    min_layer: int = 1,
    stop_at_empty: bool = False,
    support_only: bool = False,
) -> RadicalLayerSupportComputation:
    """Compute radical-layer supports directly from a finite-grid projective txt file."""

    presentation = load_projective_presentation_data(projective_txt)
    return compute_radical_layer_supports(
        presentation,
        max_layer,
        min_layer=min_layer,
        stop_at_empty=stop_at_empty,
        support_only=support_only,
    )


def compute_radical_layer_supports(
    presentation: ProjectivePresentationData,
    max_layer: int,
    *,
    min_layer: int = 1,
    stop_at_empty: bool = False,
    support_only: bool = False,
) -> RadicalLayerSupportComputation:
    """Compute supports of ``rad^(j-1) M / rad^j M`` from ``P_1 -> P_0``.

    This follows the local formula from the Radical layers section:

    ``dim(rad^r M / rad^(r+1) M)_x =
    |L_r(x)| - (rank D_{R_<r+1(x), C(x)} - rank D_{R_<r(x), C(x)})``.

    Public layer labels are 1-based, so layer ``j`` stores the formula with
    internal radical depth ``r = j - 1``.  The largest possible public layer on
    a grid with upper grade ``(n, m)`` is ``n + m + 1``.

    For each grid point, rows of ``P_0`` are added by increasing distance from
    their projective grade to the point, and an incremental GF(2) row-rank basis
    records the rank increase of each distance shell.
    """

    if min_layer < 1:
        raise ValueError("min_layer must be at least 1.")
    max_possible_layer = presentation.bounds.max_grade[0] + presentation.bounds.max_grade[1] + 1
    if max_layer < min_layer:
        raise ValueError(f"max_layer={max_layer} is smaller than min_layer={min_layer}.")
    if max_layer > max_possible_layer:
        raise ValueError(
            f"max_layer={max_layer} exceeds n+m+1={max_possible_layer} for grid max "
            f"{presentation.bounds.max_grade}."
        )

    if support_only:
        layer_supports: list[set[GridPoint]] = [set() for _ in range(max_layer + 1)]
        layer_dimensions: list[dict[GridPoint, int]] = []
    else:
        layer_supports = []
        layer_dimensions = [dict() for _ in range(max_layer + 1)]

    row_column_masks = _row_column_masks(len(presentation.p0_grades), presentation.matrix_columns)
    max_x, max_y = presentation.bounds.max_grade
    max_depth = max_layer - 1

    for x0 in range(max_x + 1):
        for x1 in range(max_y + 1):
            point = (x0, x1)
            active_column_mask = 0
            for column_index, grade in enumerate(presentation.p1_grades):
                if _leq(grade, point):
                    active_column_mask |= 1 << column_index

            events: dict[int, list[int]] = {}
            for row_index, grade in enumerate(presentation.p0_grades):
                if not _leq(grade, point):
                    continue
                depth = _l1_distance(grade, point)
                if depth <= max_depth:
                    events.setdefault(depth, []).append(row_index)
            if not events:
                continue

            basis = _GF2IncrementalRank()
            previous_rank = 0
            for depth in range(0, max_depth + 1):
                rows_at_layer = events.get(depth, ())
                for row_index in rows_at_layer:
                    basis.add(row_column_masks[row_index] & active_column_mask)
                rank_increase = basis.rank - previous_rank
                layer_dimension = len(rows_at_layer) - rank_increase
                layer = depth + 1
                if layer_dimension > 0 and layer >= min_layer:
                    if support_only:
                        layer_supports[layer].add(point)
                    else:
                        layer_dimensions[layer][point] = layer_dimension
                previous_rank = basis.rank

    final_max_layer = max_layer
    if stop_at_empty:
        for layer in range(min_layer, max_layer + 1):
            layer_is_empty = not layer_supports[layer] if support_only else not layer_dimensions[layer]
            if layer_is_empty:
                final_max_layer = layer
                break

    if support_only:
        layers = tuple(
            RadicalLayerSupport(
                layer=layer,
                support=tuple(sorted(layer_supports[layer])),
                dimensions=(),
            )
            for layer in range(min_layer, final_max_layer + 1)
        )
    else:
        layers = tuple(
            RadicalLayerSupport(
                layer=layer,
                support=tuple(sorted(layer_dimensions[layer])),
                dimensions=tuple(sorted(layer_dimensions[layer].items())),
            )
            for layer in range(min_layer, final_max_layer + 1)
        )
    return RadicalLayerSupportComputation(
        presentation=presentation,
        max_layer=final_max_layer,
        layers=layers,
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


def compute_socle_layer_supports_from_injective_txt(
    injective_txt: PathLike,
    max_layer: int,
    *,
    min_layer: int = 1,
    stop_at_empty: bool = False,
    support_only: bool = False,
) -> SocleLayerSupportComputation:
    """Compute socle-layer supports directly from a finite-grid injective txt file.

    ``max_layer`` is the largest layer to compute.  On a grid with upper
    coordinate ``(n, m)``, the allowed maximum is ``n + m + 1``.  Set
    ``min_layer=2`` when the ordinary socle support from ``Q^0`` is already
    displayed elsewhere.
    """

    copresentation = load_injective_copresentation_data(injective_txt)
    return compute_socle_layer_supports(
        copresentation,
        max_layer,
        min_layer=min_layer,
        stop_at_empty=stop_at_empty,
        support_only=support_only,
    )


def compute_socle_layer_supports(
    copresentation: InjectiveCopresentationData,
    max_layer: int,
    *,
    min_layer: int = 1,
    stop_at_empty: bool = False,
    support_only: bool = False,
) -> SocleLayerSupportComputation:
    """Compute supports of ``soc^j M / soc^{j-1} M`` from ``Q^0 -> Q^1``.

    The implementation follows the local formula in Section 3.1 of the
    referenced manuscript:

    ``dim (soc^r M)_x = |C_r(x)| - rank C_{R(x), C_r(x)}``.

    For each grid point, active columns are added in increasing layer order and
    an incremental GF(2) row-echelon basis is maintained.  This avoids rebuilding
    and reranking a submatrix for every ``(x, r)`` pair.

    Set ``stop_at_empty=True`` to keep layers only through the first empty layer.
    Set ``support_only=True`` when downstream code only needs membership in the
    layer support; this skips storing pointwise dimensions.
    """

    if min_layer < 1:
        raise ValueError("min_layer must be at least 1.")
    max_possible_layer = copresentation.bounds.max_grade[0] + copresentation.bounds.max_grade[1] + 1
    if max_layer < min_layer:
        raise ValueError(f"max_layer={max_layer} is smaller than min_layer={min_layer}.")
    if max_layer > max_possible_layer:
        raise ValueError(
            f"max_layer={max_layer} exceeds n+m+1={max_possible_layer} for grid max "
            f"{copresentation.bounds.max_grade}."
        )

    if support_only:
        layer_supports: list[set[GridPoint]] = [set() for _ in range(max_layer + 1)]
        layer_dimensions: list[dict[GridPoint, int]] = []
    else:
        layer_supports = []
        layer_dimensions = [dict() for _ in range(max_layer + 1)]
    column_masks = _column_row_masks(copresentation.matrix_columns)
    max_x, max_y = copresentation.bounds.max_grade

    for x0 in range(max_x + 1):
        for x1 in range(max_y + 1):
            point = (x0, x1)
            row_mask = _active_row_mask(point, copresentation.q1_grades)
            events: dict[int, list[int]] = {}
            for column_index, grade in enumerate(copresentation.q0_grades):
                if not _leq(point, grade):
                    continue
                layer = _l1_distance(point, grade) + 1
                if layer <= max_layer:
                    events.setdefault(layer, []).append(column_index)
            if not events:
                continue

            basis = _GF2IncrementalRank()
            active_columns = 0
            previous_socle_dimension = 0
            for layer in sorted(events):
                for column_index in events[layer]:
                    active_columns += 1
                    basis.add(column_masks[column_index] & row_mask)
                socle_dimension = active_columns - basis.rank
                layer_dimension = socle_dimension - previous_socle_dimension
                if layer_dimension > 0 and layer >= min_layer:
                    if support_only:
                        layer_supports[layer].add(point)
                    else:
                        layer_dimensions[layer][point] = layer_dimension
                previous_socle_dimension = socle_dimension

    final_max_layer = max_layer
    if stop_at_empty:
        for layer in range(min_layer, max_layer + 1):
            layer_is_empty = not layer_supports[layer] if support_only else not layer_dimensions[layer]
            if layer_is_empty:
                final_max_layer = layer
                break

    if support_only:
        layers = tuple(
            SocleLayerSupport(
                layer=layer,
                support=tuple(sorted(layer_supports[layer])),
                dimensions=(),
            )
            for layer in range(min_layer, final_max_layer + 1)
        )
    else:
        layers = tuple(
            SocleLayerSupport(
                layer=layer,
                support=tuple(sorted(layer_dimensions[layer])),
                dimensions=tuple(sorted(layer_dimensions[layer].items())),
            )
            for layer in range(min_layer, final_max_layer + 1)
        )
    return SocleLayerSupportComputation(
        copresentation=copresentation,
        max_layer=final_max_layer,
        layers=layers,
    )


def compute_module_support(copresentation: InjectiveCopresentationData) -> tuple[GridPoint, ...]:
    """Return ``supp M`` from an injective copresentation ``Q^0 -> Q^1``.

    At each grid point ``x`` this computes
    ``dim M_x = |A(x)| - rank C_{B(x), A(x)}`` over GF(2), where
    ``A(x) = {j | x <= a_j}`` and ``B(x) = {i | x <= b_i}``.
    """

    context = _InjectiveRankContext(copresentation)
    support = []
    max_x, max_y = copresentation.bounds.max_grade
    for x_coord in range(max_x + 1):
        for y_coord in range(max_y + 1):
            point = (x_coord, y_coord)
            row_mask = _active_grade_mask(point, context.q1_active_masks_by_x)
            column_mask = _active_grade_mask(point, context.q0_active_masks_by_x)
            if column_mask.bit_count() - _restricted_column_rank(row_mask, column_mask, context) > 0:
                support.append(point)
    return tuple(support)


def module_support_from_socle_layers(
    module_socle_layer_supports: Mapping[int, Iterable[GridPoint]],
) -> tuple[GridPoint, ...]:
    """Return the union of already computed socle-layer supports."""

    support: set[GridPoint] = set()
    for points in module_socle_layer_supports.values():
        support.update(_normalize_points(points))
    return tuple(sorted(support))


def compute_reachability_regions(
    copresentation: InjectiveCopresentationData,
    omegas: Optional[Iterable[GridPoint]] = None,
    *,
    module_support: Optional[Iterable[GridPoint]] = None,
) -> dict[GridPoint, tuple[GridPoint, ...]]:
    """Compute projected-kernel reachability regions ``R_omega``.

    For ``omega`` in the selected column labels, this returns the points ``x``
    with ``x <= omega`` and nonzero projection

    ``ker C_{B(x), A(x)} -> k^{A_omega}``.

    The rank criterion used is

    ``rank C_{B(x), A(x)} < rank C_{B(x), A(x) \\ A_omega} + |A_omega|``.
    """

    context = _InjectiveRankContext(copresentation)
    omega_tuple = _normalize_points(
        omegas if omegas is not None else context.q0_point_masks
    )
    if module_support is not None:
        support_points = _normalize_points(module_support)
        points_by_omega = {
            omega: tuple(point for point in support_points if _leq(point, omega))
            for omega in omega_tuple
        }
    else:
        points_by_omega = {
            omega: tuple(
                (x_coord, y_coord)
                for x_coord in range(omega[0] + 1)
                for y_coord in range(omega[1] + 1)
            )
            for omega in omega_tuple
        }

    regions: dict[GridPoint, tuple[GridPoint, ...]] = {}
    for omega in omega_tuple:
        omega_mask = context.q0_point_masks.get(omega, 0)
        if not omega_mask:
            regions[omega] = ()
            continue
        reachable = []
        for point in points_by_omega[omega]:
            row_mask = _active_grade_mask(point, context.q1_active_masks_by_x)
            column_mask = _active_grade_mask(point, context.q0_active_masks_by_x)
            if not column_mask & omega_mask:
                continue
            rank_all = _restricted_column_rank(row_mask, column_mask, context)
            rank_without = _restricted_column_rank(row_mask, column_mask & ~omega_mask, context)
            if rank_all < rank_without + omega_mask.bit_count():
                reachable.append(point)
        regions[omega] = tuple(reachable)
    return regions


def build_interval_candidate(interval: Interval, bounds: GridBounds) -> IntervalCandidate:
    """Build cached finite-grid interval data from a utils.Interval."""

    points = interval_points(interval, bounds)
    if not points:
        raise ValueError("An interval candidate must have a nonempty hull.")
    if not is_interval_points(points, bounds):
        raise ValueError("The candidate hull is not a connected convex interval.")

    src, snk = interval_sources_sinks(points, bounds)
    normalized_interval = Interval(src, snk)
    src_uuset, snk_ddset = proper_sources_sinks_from_extrema(src, snk, bounds)
    column_ranges = _interval_column_ranges(src, snk, bounds)
    if column_ranges is None:
        raise ValueError("An interval candidate must have a nonempty hull.")
    return IntervalCandidate(
        interval=normalized_interval,
        src=src,
        snk=snk,
        src_1_circ=src_1_circ(src),
        src_uuset=src_uuset,
        snk_1_circ=snk_1_circ(snk),
        snk_ddset=snk_ddset,
        column_ranges=column_ranges,
        points=None,
    )


def interval_points(interval: Interval, bounds: GridBounds) -> tuple[GridPoint, ...]:
    """Return the sorted hull points of a utils.Interval in the compressed grid."""

    return _normalize_points(int_hull_Rfree(interval, _gridsize(bounds)))


def interval_sources_sinks(
    points: Iterable[GridPoint],
    bounds: GridBounds,
) -> tuple[tuple[GridPoint, ...], tuple[GridPoint, ...]]:
    """Return source and sink antichains for a point set."""

    point_tuple = _normalize_points(points)
    if not point_tuple:
        return (), ()
    src, snk = get_src_snk_Rfree(list(point_tuple))
    return _normalize_points(src), _normalize_points(snk)


def proper_sources_sinks(
    points: Iterable[GridPoint],
    bounds: GridBounds,
) -> tuple[tuple[GridPoint, ...], tuple[GridPoint, ...]]:
    """Return src(uuset I) and snk(ddset I) for an interval point set."""

    point_tuple = _normalize_points(points)
    if not point_tuple:
        return (), ()
    src_uuset, snk_ddset = get_prop_src_snk_Rfree(list(point_tuple), _gridsize(bounds))
    return _normalize_points(src_uuset), _normalize_points(snk_ddset)


def proper_sources_sinks_from_extrema(
    sources: Iterable[GridPoint],
    sinks: Iterable[GridPoint],
    bounds: GridBounds,
) -> tuple[tuple[GridPoint, ...], tuple[GridPoint, ...]]:
    """Return src(uuset I) and snk(ddset I) directly from source/sink antichains."""

    source_tuple = _normalize_points(sources)
    sink_tuple = _normalize_points(sinks)
    if len(bounds.dimensions) == 2:
        return _proper_sources_sinks_from_extrema_2d(source_tuple, sink_tuple, bounds)

    src_uuset, snk_ddset = get_prop_src_snk_from_extrema_Rfree(
        list(source_tuple),
        list(sink_tuple),
        _gridsize(bounds),
    )
    return _normalize_points(src_uuset), _normalize_points(snk_ddset)


def src_1_circ(
    sources: Iterable[GridPoint],
    rep: Optional[Representation] = None,
) -> tuple[GridPoint, ...]:
    """Return adjacent joins of sources ordered by increasing second coordinate."""

    ordered = sorted(_normalize_points(sources), key=lambda point: (point[1], point[0]))
    return _normalize_points(
        _join_grid(left, right)
        for left, right in zip(ordered, ordered[1:])
    )


def snk_1_circ(
    sinks: Iterable[GridPoint],
    rep: Optional[Representation] = None,
) -> tuple[GridPoint, ...]:
    """Return adjacent meets of sinks ordered by increasing second coordinate."""

    ordered = sorted(_normalize_points(sinks), key=lambda point: (point[1], point[0]))
    return _normalize_points(
        _meet_grid(left, right)
        for left, right in zip(ordered, ordered[1:])
    )


def is_interval_points(points: Iterable[GridPoint], bounds: GridBounds) -> bool:
    """Return True when points form a nonempty connected convex interval."""

    point_tuple = _normalize_points(points)
    if not point_tuple:
        return False
    src, snk = interval_sources_sinks(point_tuple, bounds)
    interval = Interval(src, snk)
    return set(point_tuple) == set(interval_points(interval, bounds)) and _is_connected_points(point_tuple)


def passes_crt0(
    interval: Interval | IntervalCandidate,
    support_data: ResolutionSupportData,
) -> bool:
    """Check the crt0 necessary criterion."""

    candidate = _ensure_candidate(interval, support_data)
    return set(candidate.src).issubset(support_data.top_p0_support) and set(
        candidate.snk
    ).issubset(support_data.soc_q0_support)


def passes_crt1_2d(
    interval: Interval | IntervalCandidate,
    support_data: ResolutionSupportData,
) -> bool:
    """Check the specialized 2D-grid crt1 necessary criterion."""

    candidate = _ensure_candidate(interval, support_data)
    if not passes_crt0(candidate, support_data):
        return False
    source_constraint = set(candidate.src_1_circ).union(candidate.src_uuset)
    sink_constraint = set(candidate.snk_1_circ).union(candidate.snk_ddset)
    return source_constraint.issubset(support_data.top_p1_support) and sink_constraint.issubset(
        support_data.soc_q1_support
    )


def passes_zero_pair(
    interval: Interval | IntervalCandidate,
    oracle: Optional[ZeroPairOracle],
    support_data: Optional[ResolutionSupportData] = None,
) -> bool:
    """Check the optional zero-pair criterion when an oracle is available."""

    if oracle is None:
        return True
    if isinstance(interval, IntervalCandidate):
        candidate = interval
    elif support_data is not None:
        candidate = build_interval_candidate(interval, support_data.bounds)
    else:
        raise ValueError("support_data is required for non-cached intervals.")

    for source in candidate.src:
        for sink in candidate.snk:
            if _leq(source, sink) and not oracle(source, sink):
                return False
    return True


def filter_interval_candidates(
    intervals: Iterable[Interval | IntervalCandidate | tuple[Iterable[GridPoint], Iterable[GridPoint]]],
    support_data: ResolutionSupportData,
    *,
    criterion: str = "crt1_2d",
    zero_pair_oracle: Optional[ZeroPairOracle] = None,
) -> list[IntervalCandidate]:
    """Filter supplied intervals by a conservative necessary criterion."""

    retained = []
    for interval_like in intervals:
        candidate = _ensure_candidate(interval_like, support_data)
        if _passes_criterion(candidate, support_data, criterion) and passes_zero_pair(
            candidate,
            zero_pair_oracle,
        ):
            retained.append(candidate)
    return retained


def enumerate_rectangular_intervals(bounds: GridBounds) -> list[IntervalCandidate]:
    """Generate all rectangular intervals in the compressed grid."""

    candidates = []
    max_x, max_y = bounds.max_grade
    for x0 in range(max_x + 1):
        for y0 in range(max_y + 1):
            for x1 in range(x0, max_x + 1):
                for y1 in range(y0, max_y + 1):
                    candidates.append(build_interval_candidate(Interval([(x0, y0)], [(x1, y1)]), bounds))
    return candidates


def enumerate_intervals_from_supports(
    bounds: GridBounds,
    support_data: ResolutionSupportData,
    *,
    criterion: str = "crt1_2d",
    max_antichains: Optional[int] = 200000,
    max_candidates: Optional[int] = 200000,
    socle_layer_supports: Optional[Mapping[int, Iterable[GridPoint]]] = None,
    socle_filter_r: Optional[int] = None,
    radical_layer_supports: Optional[Mapping[int, Iterable[GridPoint]]] = None,
    radical_filter_r: Optional[int] = None,
) -> list[IntervalCandidate]:
    """Generate and filter intervals from P0-source and Q0-sink supports."""

    candidates, _generated_count = _enumerate_intervals_from_supports_with_stats(
        bounds,
        support_data,
        criterion=criterion,
        max_antichains=max_antichains,
        max_candidates=max_candidates,
        socle_layer_supports=socle_layer_supports,
        socle_filter_r=socle_filter_r,
        radical_layer_supports=radical_layer_supports,
        radical_filter_r=radical_filter_r,
    )
    return candidates


def compute_interval_candidates(
    projective_txt: PathLike,
    injective_txt: PathLike,
    output_path: Optional[PathLike] = None,
    *,
    criterion: str = "crt1_2d",
    generation_mode: str = "support_antichains",
    max_antichains: Optional[int] = 200000,
    max_candidates: Optional[int] = 200000,
    point_display_limit: Optional[int] = None,
    include_size: bool = False,
    socle_layer_supports: Optional[Mapping[int, Iterable[GridPoint]]] = None,
    socle_filter_r: Optional[int] = None,
    radical_layer_supports: Optional[Mapping[int, Iterable[GridPoint]]] = None,
    radical_filter_r: Optional[int] = None,
) -> IntervalCandidateComputation:
    """Compute interval candidates and optionally write a txt report."""

    support_data = load_resolution_support_data(projective_txt, injective_txt)
    if generation_mode == "support_antichains":
        candidates, generated_count = _enumerate_intervals_from_supports_with_stats(
            support_data.bounds,
            support_data,
            criterion=criterion,
            max_antichains=max_antichains,
            max_candidates=max_candidates,
            socle_layer_supports=socle_layer_supports,
            socle_filter_r=socle_filter_r,
            radical_layer_supports=radical_layer_supports,
            radical_filter_r=radical_filter_r,
        )
    elif generation_mode == "rectangles":
        rectangles = enumerate_rectangular_intervals(support_data.bounds)
        generated_count = len(rectangles)
        if max_candidates is not None and generated_count > max_candidates:
            raise CandidateGenerationLimitError(
                f"Generated {generated_count} rectangles, exceeding max_candidates={max_candidates}."
            )
        candidates = filter_interval_candidates(rectangles, support_data, criterion=criterion)
        normalized_socle_supports, effective_socle_r = _normalize_optional_socle_filter(
            socle_layer_supports,
            socle_filter_r,
        )
        normalized_radical_supports, effective_radical_r = _normalize_optional_radical_filter(
            radical_layer_supports,
            radical_filter_r,
        )
        if (
            (normalized_socle_supports is not None and effective_socle_r is not None)
            or (normalized_radical_supports is not None and effective_radical_r is not None)
        ):
            candidates = [
                candidate
                for candidate in candidates
                if _passes_layer_filters_from_ranges(
                    candidate.src,
                    candidate.snk,
                    candidate.column_ranges,
                    normalized_socle_supports,
                    effective_socle_r,
                    normalized_radical_supports,
                    effective_radical_r,
                )
            ]
    else:
        raise ValueError(f"Unknown generation_mode: {generation_mode!r}.")

    result = IntervalCandidateComputation(
        support_data=support_data,
        candidates=tuple(candidates),
        generated_count=generated_count,
        retained_count=len(candidates),
        criterion=criterion,
        generation_mode=generation_mode,
        socle_filter_r=_effective_socle_filter_r(socle_layer_supports, socle_filter_r),
        radical_filter_r=_effective_radical_filter_r(radical_layer_supports, radical_filter_r),
        output_path=Path(output_path) if output_path is not None else _default_output_path(projective_txt),
    )
    if result.output_path is not None:
        write_interval_candidates_text(
            result,
            result.output_path,
            point_display_limit=point_display_limit,
            include_size=include_size,
        )
    return result


def load_interval_candidates_text(
    path: PathLike,
    *,
    projective_txt: Optional[PathLike] = None,
    injective_txt: Optional[PathLike] = None,
) -> IntervalCandidateComputation:
    """Load a candidate txt file written by ``write_interval_candidates_text``."""

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

    projective_path = Path(projective_txt) if projective_txt is not None else Path(header["projective_resolution"])
    injective_path = Path(injective_txt) if injective_txt is not None else Path(header["injective_coresolution"])
    support_data = load_resolution_support_data(projective_path, injective_path)

    candidates = []
    current: dict[str, str] = {}

    def flush_current() -> None:
        if not current:
            return
        src = _parse_points_field(current.get("src", "-"))
        snk = _parse_points_field(current.get("snk", "-"))
        column_ranges = _interval_column_ranges(src, snk, support_data.bounds)
        if column_ranges is None:
            raise CandidateParseError(f"Candidate in {input_path} has an empty interval: src={src}, snk={snk}.")
        candidates.append(
            IntervalCandidate(
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
        )

    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("interval "):
            flush_current()
            current = {}
            continue
        key, separator, value = line.partition(":")
        if separator:
            current[key.strip()] = value.strip()
    flush_current()

    return IntervalCandidateComputation(
        support_data=support_data,
        candidates=tuple(candidates),
        generated_count=int(header.get("generated_intervals", len(candidates))),
        retained_count=int(header.get("retained_intervals", len(candidates))),
        criterion=header.get("criterion", "unknown"),
        generation_mode=header.get("generation_mode", "unknown"),
        socle_filter_r=int(header["socle_filter_r"]) if "socle_filter_r" in header else None,
        radical_filter_r=int(header["radical_filter_r"]) if "radical_filter_r" in header else None,
        output_path=input_path,
    )


def write_interval_candidates_text(
    result: IntervalCandidateComputation,
    path: PathLike,
    *,
    point_display_limit: Optional[int] = None,
    include_size: bool = False,
) -> Path:
    """Write candidate intervals as a human-readable txt file.

    By default this writes only extrema and criterion support data.  Internal
    interval points and interval sizes are intentionally omitted because large
    grids make those fields bulky and downstream computations reconstruct size
    from the compact column-range representation when needed.
    """

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    support = result.support_data
    lines = [
        "# finite_grid_interval_candidates",
        f"# projective_resolution: {support.projective_path}",
        f"# injective_coresolution: {support.injective_path}",
        "# grid min: 0 0",
        f"# grid max: {_format_point_plain(support.bounds.max_grade)}",
        f"# criterion: {result.criterion}",
        f"# generation_mode: {result.generation_mode}",
        f"# generated_intervals: {result.generated_count}",
        f"# retained_intervals: {result.retained_count}",
    ]
    if result.socle_filter_r is not None:
        lines.append(f"# socle_filter_r: {result.socle_filter_r}")
    if result.radical_filter_r is not None:
        lines.append(f"# radical_filter_r: {result.radical_filter_r}")

    for index, candidate in enumerate(result.candidates, start=1):
        lines.extend(
            [
                "",
                f"interval {index}",
                f"src: {_format_points(candidate.src)}",
                f"snk: {_format_points(candidate.snk)}",
                f"src_1_circ: {_format_points(candidate.src_1_circ)}",
                f"src_uuset: {_format_points(candidate.src_uuset)}",
                f"snk_1_circ: {_format_points(candidate.snk_1_circ)}",
                f"snk_ddset: {_format_points(candidate.snk_ddset)}",
            ]
        )
        if include_size:
            lines.append(f"size: {candidate.size}")
        if point_display_limit is not None and candidate.size <= point_display_limit:
            points = candidate.points
            if points is None:
                points = interval_points(candidate.interval, support.bounds)
            lines.append(f"points: {_format_points(points)}")
        elif point_display_limit is not None:
            lines.append(f"points: omitted; size exceeds {point_display_limit}")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def interval_socle_layer_support(
    candidate: IntervalCandidate,
    layer: int,
) -> tuple[GridPoint, ...]:
    """Return ``supp(soc^layer V_I / soc^(layer-1) V_I)``.

    This is the set of grid points ``x`` in the interval with ``h_I(x)=layer-1``
    from formula ``eqn:soc-layer-interval``.
    """

    if layer < 1:
        raise ValueError("layer must be at least 1.")
    supports = interval_socle_layer_supports_up_to(candidate, layer)
    return supports.get(layer, ())


def interval_socle_layer_supports_up_to(
    candidate: IntervalCandidate,
    max_layer: int,
) -> dict[int, tuple[GridPoint, ...]]:
    """Return interval socle-layer supports for layers ``1..max_layer``."""

    if max_layer < 1:
        raise ValueError("max_layer must be at least 1.")
    by_height = _interval_height_supports_up_to(candidate, max_layer - 1)
    return {
        layer: tuple(sorted(by_height.get(layer - 1, ())))
        for layer in range(1, max_layer + 1)
    }


def passes_crit_r_socle_layers(
    candidate: IntervalCandidate,
    module_socle_layer_supports: dict[int, frozenset[GridPoint]] | dict[int, set[GridPoint]],
    r: int,
) -> bool:
    """Return whether an interval satisfies the socle-layer support criterion.

    The layer cap ``r`` is 1-based: layer ``j`` means
    ``soc^j V_I / soc^(j-1) V_I``.
    """

    if r < 1:
        raise ValueError("r must be at least 1.")
    interval_layers = _interval_height_supports_up_to(candidate, r - 1)
    for height, points in interval_layers.items():
        module_support = module_socle_layer_supports.get(height + 1, frozenset())
        if not points.issubset(module_support):
            return False
    return True


def interval_radical_layer_support(
    candidate: IntervalCandidate,
    layer: int,
) -> tuple[GridPoint, ...]:
    """Return ``supp(rad^(layer-1) V_I / rad^layer V_I)``."""

    if layer < 1:
        raise ValueError("layer must be at least 1.")
    supports = interval_radical_layer_supports_up_to(candidate, layer)
    return supports.get(layer, ())


def interval_radical_layer_supports_up_to(
    candidate: IntervalCandidate,
    max_layer: int,
) -> dict[int, tuple[GridPoint, ...]]:
    """Return interval radical-layer supports for layers ``1..max_layer``."""

    if max_layer < 1:
        raise ValueError("max_layer must be at least 1.")
    by_depth = _interval_depth_supports_up_to(candidate, max_layer - 1)
    return {
        layer: tuple(sorted(by_depth.get(layer - 1, ())))
        for layer in range(1, max_layer + 1)
    }


def passes_radical_layer_supports(
    candidate: IntervalCandidate,
    module_radical_layer_supports: dict[int, frozenset[GridPoint]] | dict[int, set[GridPoint]],
    r: int,
) -> bool:
    """Return whether an interval satisfies radical-layer support inclusion."""

    if r < 1:
        raise ValueError("r must be at least 1.")
    interval_layers = _interval_depth_supports_up_to(candidate, r - 1)
    for depth, points in interval_layers.items():
        module_support = module_radical_layer_supports.get(depth + 1, frozenset())
        if not points.issubset(module_support):
            return False
    return True


def generate_submodule_interval_families_from_socle_layers(
    bounds: GridBounds,
    module_socle_layer_supports: dict[int, frozenset[GridPoint]] | dict[int, set[GridPoint]],
    *,
    r: int,
    reachability_regions: Optional[Mapping[GridPoint, Iterable[GridPoint]]] = None,
    module_support: Optional[Iterable[GridPoint]] = None,
    output_path: Optional[PathLike] = None,
) -> SubmoduleIntervalFamilyComputation:
    """Generate socle-constrained interval families for submodules ``V_I -> M``.

    This uses only the supports of ``soc^k M / soc^(k-1) M``.  The first layer
    supplies possible sink antichains ``max(I)``.  For each sink antichain, the
    formula ``eqn:soc-layer-interval`` determines which grid points may occur at
    each height ``h_I(x)``.  The source candidates are then cut down by the
    top-restriction criterion for minimal points.

    If ``reachability_regions`` is supplied, a point ``x`` is allowed for a
    sink antichain ``W`` only when ``x in R_omega`` for every ``omega in W`` with
    ``x <= omega``.  This implements the projected-kernel reachability pruning
    criterion before concrete source antichains are enumerated.

    The returned objects are families, not fully enumerated source antichains,
    because without projective/top constraints the number of concrete intervals
    can be exponential.
    """

    if r < 0:
        raise ValueError("r must be nonnegative.")

    normalized_supports = {
        int(layer): set(_normalize_points(points))
        for layer, points in module_socle_layer_supports.items()
    }
    reachability_sets = _normalize_reachability_regions(reachability_regions)
    module_support_set = (
        set(_normalize_points(module_support)) if module_support is not None else None
    )
    sink_antichains = _generate_antichains(normalized_supports.get(1, ()), None)

    families: list[SubmoduleIntervalFamily] = []
    for sinks in sink_antichains:
        reachability_allowed = None
        if reachability_sets is not None:
            _require_reachability_regions_for_sinks(sinks, reachability_sets)
            reachability_allowed = lambda point, sinks=sinks: _point_reachable_to_sinks(
                point,
                sinks,
                reachability_sets,
            )
            if module_support_set is not None and not _sinks_share_reachability_component(
                sinks,
                module_support_set,
                reachability_allowed,
            ):
                continue
        source_candidates = _source_candidates_for_submodule_family(
            sinks,
            bounds,
            normalized_supports,
            r,
            point_allowed=reachability_allowed,
        )
        if source_candidates:
            families.append(
                SubmoduleIntervalFamily(
                    sinks=sinks,
                    source_candidates=source_candidates,
                )
            )

    result = SubmoduleIntervalFamilyComputation(
        bounds=bounds,
        r=r,
        sink_antichain_count=len(sink_antichains),
        families=tuple(families),
        output_path=Path(output_path) if output_path is not None else None,
    )
    if output_path is not None:
        write_submodule_interval_families_text(result, output_path)
    return result


def write_submodule_interval_families_text(
    result: SubmoduleIntervalFamilyComputation,
    path: PathLike,
) -> Path:
    """Write socle-constrained submodule interval families."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# relative_betti_submodule_interval_families",
        "# source: socle_layers_only",
        "# criterion: crit_r_socle_layers",
        f"# r: {result.r}",
        "# grid min: 0 0",
        f"# grid max: {_format_point_plain(result.bounds.max_grade)}",
        f"# sink_antichains: {result.sink_antichain_count}",
        f"# retained_families: {result.retained_family_count}",
        "# note: source_candidates satisfy the top-restriction criterion for possible minimal points.",
    ]

    for index, family in enumerate(result.families, start=1):
        lines.extend(
            [
                "",
                f"family {index}",
                f"snk: {_format_points(family.sinks)}",
                f"source_candidate_ranges: {_format_point_column_ranges(family.source_candidates)}",
            ]
        )

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def filter_submodule_families_by_injective_rank(
    family_result: SubmoduleIntervalFamilyComputation,
    copresentation: InjectiveCopresentationData,
    *,
    source_mode: str = "single_source",
    output_path: Optional[PathLike] = None,
    retain_intervals: Optional[bool] = None,
) -> RankFilteredSubmoduleIntervalComputation:
    """Expand submodule families and apply the injective rank necessary test.

    ``source_mode="single_source"`` expands one-source intervals only and is the
    safest large-grid default.  ``source_mode="minimal_covers"`` expands source
    antichains that cover the chosen sinks minimally.  ``source_mode="all_antichains"``
    is complete but can be exponentially large.

    If ``output_path`` is supplied, retained intervals are streamed to disk by
    default instead of kept in memory.  Pass ``retain_intervals=True`` to also
    keep them in the returned object.
    """

    if family_result.bounds != copresentation.bounds:
        raise ValueError(
            f"family bounds {family_result.bounds} do not match copresentation bounds {copresentation.bounds}."
        )
    if source_mode not in {"single_source", "minimal_covers", "all_antichains"}:
        raise ValueError(f"Unknown source_mode: {source_mode!r}.")

    if retain_intervals is None:
        retain_intervals = output_path is None

    rank_context = _InjectiveRankContext(copresentation)
    retained: list[RankFilteredSubmoduleInterval] = []
    tested = 0
    retained_count = 0
    output = Path(output_path) if output_path is not None else None
    body_path: Optional[Path] = None
    body_file = None
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        body_path = output.with_name(output.name + ".body.tmp")
        body_file = body_path.open("w", encoding="utf-8")

    try:
        for family_index, family in enumerate(family_result.families, start=1):
            for candidate in _iter_concrete_intervals_from_family(
                family,
                family_result.bounds,
                source_mode=source_mode,
            ):
                tested += 1
                if not _passes_injective_rank_test(candidate, rank_context):
                    continue
                retained_count += 1
                retained_interval = RankFilteredSubmoduleInterval(
                    family_index=family_index,
                    interval=candidate,
                )
                if retain_intervals:
                    retained.append(retained_interval)
                if body_file is not None:
                    body_file.write(
                        "\n".join(
                            _rank_filtered_interval_lines(retained_count, retained_interval)
                        )
                        + "\n"
                    )
    finally:
        if body_file is not None:
            body_file.close()

    result = RankFilteredSubmoduleIntervalComputation(
        bounds=family_result.bounds,
        source_mode=source_mode,
        tested_intervals=tested,
        retained_intervals=tuple(retained),
        retained_interval_count=retained_count,
        output_path=output,
    )
    if output is not None and body_path is not None:
        _write_rank_filtered_submodule_intervals_from_body(result, output, body_path)
    return result


def write_rank_filtered_submodule_intervals_text(
    result: RankFilteredSubmoduleIntervalComputation,
    path: PathLike,
) -> Path:
    """Write concrete intervals retained by the injective rank test."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = _rank_filtered_header_lines(result)

    for index, retained in enumerate(result.retained_intervals, start=1):
        lines.extend(_rank_filtered_interval_lines(index, retained))

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _rank_filtered_header_lines(
    result: RankFilteredSubmoduleIntervalComputation,
) -> list[str]:
    return [
        "# relative_betti_rank_filtered_submodule_intervals",
        "# source: submodule_interval_families",
        "# criterion: injective_rank_necessary_test",
        f"# source_mode: {result.source_mode}",
        "# grid min: 0 0",
        f"# grid max: {_format_point_plain(result.bounds.max_grade)}",
        f"# tested_intervals: {result.tested_intervals}",
        f"# retained_intervals: {result.retained_count}",
    ]


def _rank_filtered_interval_lines(
    index: int,
    retained: RankFilteredSubmoduleInterval,
) -> list[str]:
    candidate = retained.interval
    return [
        "",
        f"interval {index}",
        f"family_index: {retained.family_index}",
        f"src: {_format_points(candidate.src)}",
        f"snk: {_format_points(candidate.snk)}",
    ]


def _write_rank_filtered_submodule_intervals_from_body(
    result: RankFilteredSubmoduleIntervalComputation,
    output: Path,
    body_path: Path,
) -> None:
    with output.open("w", encoding="utf-8") as output_file:
        output_file.write("\n".join(_rank_filtered_header_lines(result)) + "\n")
        with body_path.open("r", encoding="utf-8") as body_file:
            for chunk in iter(lambda: body_file.read(1024 * 1024), ""):
                output_file.write(chunk)
    body_path.unlink(missing_ok=True)


def _enumerate_intervals_from_supports_with_stats(
    bounds: GridBounds,
    support_data: ResolutionSupportData,
    *,
    criterion: str,
    max_antichains: Optional[int],
    max_candidates: Optional[int],
    socle_layer_supports: Optional[Mapping[int, Iterable[GridPoint]]] = None,
    socle_filter_r: Optional[int] = None,
    radical_layer_supports: Optional[Mapping[int, Iterable[GridPoint]]] = None,
    radical_filter_r: Optional[int] = None,
) -> tuple[list[IntervalCandidate], int]:
    if bounds != support_data.bounds:
        raise ValueError(f"bounds {bounds} do not match support data bounds {support_data.bounds}.")

    source_adjacent_allowed: Optional[Callable[[GridPoint, GridPoint], bool]] = None
    sink_adjacent_allowed: Optional[Callable[[GridPoint, GridPoint], bool]] = None
    if criterion in {"crt1", "crt1_2d"}:
        top_p1_support = set(support_data.top_p1_support)
        soc_q1_support = set(support_data.soc_q1_support)
        source_adjacent_allowed = lambda left, right: _join_grid(left, right) in top_p1_support
        sink_adjacent_allowed = lambda left, right: _meet_grid(left, right) in soc_q1_support

    source_antichains = _generate_antichains(
        support_data.top_p0_support,
        max_antichains,
        adjacent_pair_allowed=source_adjacent_allowed,
    )
    sink_antichains = _generate_antichains(
        support_data.soc_q0_support,
        max_antichains,
        adjacent_pair_allowed=sink_adjacent_allowed,
    )
    normalized_socle_supports, effective_socle_r = _normalize_optional_socle_filter(
        socle_layer_supports,
        socle_filter_r,
    )
    normalized_radical_supports, effective_radical_r = _normalize_optional_radical_filter(
        radical_layer_supports,
        radical_filter_r,
    )

    retained: list[IntervalCandidate] = []
    seen_intervals: set[tuple[tuple[GridPoint, ...], tuple[GridPoint, ...]]] = set()
    generated_count = 0
    for sources in source_antichains:
        for sinks in sink_antichains:
            if not _source_sink_pair_feasible(sources, sinks):
                continue
            generated_count += 1
            if max_candidates is not None and generated_count > max_candidates:
                raise CandidateGenerationLimitError(
                    f"Candidate generation exceeded max_candidates={max_candidates}."
                )
            summary = _interval_column_range_summary(sources, sinks, bounds)
            if summary is None:
                continue
            column_ranges = summary.column_ranges
            if not summary.connected or summary.sources != tuple(sources) or summary.sinks != tuple(sinks):
                continue
            if not _passes_layer_filters_from_ranges(
                sources,
                sinks,
                column_ranges,
                normalized_socle_supports,
                effective_socle_r,
                normalized_radical_supports,
                effective_radical_r,
            ):
                continue
            interval_key = (sources, sinks)
            if interval_key in seen_intervals:
                continue
            seen_intervals.add(interval_key)
            candidate = _candidate_from_extrema_if_passes(
                sources,
                sinks,
                column_ranges,
                bounds,
                support_data,
                criterion,
            )
            if candidate is not None:
                retained.append(candidate)
    return retained, generated_count


def _candidate_from_extrema(
    src: tuple[GridPoint, ...],
    snk: tuple[GridPoint, ...],
    column_ranges: tuple[Optional[tuple[int, int]], ...],
    bounds: GridBounds,
) -> IntervalCandidate:
    src_uuset, snk_ddset = proper_sources_sinks_from_extrema(src, snk, bounds)
    return IntervalCandidate(
        interval=Interval(src, snk),
        src=src,
        snk=snk,
        src_1_circ=src_1_circ(src),
        src_uuset=src_uuset,
        snk_1_circ=snk_1_circ(snk),
        snk_ddset=snk_ddset,
        column_ranges=column_ranges,
        points=None,
    )


def _candidate_from_extrema_if_passes(
    src: tuple[GridPoint, ...],
    snk: tuple[GridPoint, ...],
    column_ranges: tuple[Optional[tuple[int, int]], ...],
    bounds: GridBounds,
    support_data: ResolutionSupportData,
    criterion: str,
) -> Optional[IntervalCandidate]:
    if criterion == "crt0":
        if not set(src).issubset(support_data.top_p0_support) or not set(snk).issubset(
            support_data.soc_q0_support
        ):
            return None
    elif criterion in {"crt1", "crt1_2d"}:
        if not set(src).issubset(support_data.top_p0_support) or not set(snk).issubset(
            support_data.soc_q0_support
        ):
            return None
        src_1 = src_1_circ(src)
        if not set(src_1).issubset(support_data.top_p1_support):
            return None
        snk_1 = snk_1_circ(snk)
        if not set(snk_1).issubset(support_data.soc_q1_support):
            return None
        src_uuset, snk_ddset = proper_sources_sinks_from_extrema(src, snk, bounds)
        if not set(src_uuset).issubset(support_data.top_p1_support):
            return None
        if not set(snk_ddset).issubset(support_data.soc_q1_support):
            return None
        return IntervalCandidate(
            interval=Interval(src, snk),
            src=src,
            snk=snk,
            src_1_circ=src_1,
            src_uuset=src_uuset,
            snk_1_circ=snk_1,
            snk_ddset=snk_ddset,
            column_ranges=column_ranges,
            points=None,
        )
    else:
        raise ValueError(f"Unknown criterion: {criterion!r}.")

    return _candidate_from_extrema(src, snk, column_ranges, bounds)


def _normalize_optional_socle_filter(
    socle_layer_supports: Optional[Mapping[int, Iterable[GridPoint]]],
    socle_filter_r: Optional[int],
) -> tuple[Optional[dict[int, set[GridPoint]]], Optional[int]]:
    if socle_layer_supports is None:
        if socle_filter_r is not None:
            raise ValueError("socle_filter_r requires socle_layer_supports.")
        return None, None

    normalized = {
        int(layer): set(_normalize_points(points))
        for layer, points in socle_layer_supports.items()
    }
    effective_r = _effective_socle_filter_r(normalized, socle_filter_r)
    if effective_r is not None and effective_r < 1:
        raise ValueError("socle_filter_r must be at least 1.")
    return normalized, effective_r


def _effective_socle_filter_r(
    socle_layer_supports: Optional[Mapping[int, Iterable[GridPoint]]],
    socle_filter_r: Optional[int],
) -> Optional[int]:
    if socle_layer_supports is None:
        return None
    if socle_filter_r is not None:
        return int(socle_filter_r)
    layers = [int(layer) for layer in socle_layer_supports]
    return max(layers) if layers else None


def _normalize_optional_radical_filter(
    radical_layer_supports: Optional[Mapping[int, Iterable[GridPoint]]],
    radical_filter_r: Optional[int],
) -> tuple[Optional[dict[int, set[GridPoint]]], Optional[int]]:
    if radical_layer_supports is None:
        if radical_filter_r is not None:
            raise ValueError("radical_filter_r requires radical_layer_supports.")
        return None, None

    normalized = {
        int(layer): set(_normalize_points(points))
        for layer, points in radical_layer_supports.items()
    }
    effective_r = _effective_radical_filter_r(normalized, radical_filter_r)
    if effective_r is not None and effective_r < 1:
        raise ValueError("radical_filter_r must be at least 1.")
    return normalized, effective_r


def _effective_radical_filter_r(
    radical_layer_supports: Optional[Mapping[int, Iterable[GridPoint]]],
    radical_filter_r: Optional[int],
) -> Optional[int]:
    if radical_layer_supports is None:
        return None
    if radical_filter_r is not None:
        return int(radical_filter_r)
    layers = [int(layer) for layer in radical_layer_supports]
    return max(layers) if layers else None


def _passes_crit_r_socle_layers_from_ranges(
    sinks: Sequence[GridPoint],
    column_ranges: Sequence[Optional[tuple[int, int]]],
    module_socle_layer_supports: Mapping[int, set[GridPoint]],
    r: int,
) -> bool:
    if r < 1:
        return True
    bounds = _column_ranges_bounds(column_ranges)
    if bounds is None:
        return True
    for layer in range(1, r + 1):
        if not _passes_socle_layer_from_ranges(
            sinks,
            column_ranges,
            module_socle_layer_supports,
            layer,
            bounds=bounds,
        ):
            return False
    return True


def _passes_layer_filters_from_ranges(
    sources: Sequence[GridPoint],
    sinks: Sequence[GridPoint],
    column_ranges: Sequence[Optional[tuple[int, int]]],
    module_socle_layer_supports: Optional[Mapping[int, set[GridPoint]]],
    socle_filter_r: Optional[int],
    module_radical_layer_supports: Optional[Mapping[int, set[GridPoint]]],
    radical_filter_r: Optional[int],
) -> bool:
    max_layer = max(
        socle_filter_r or 0,
        radical_filter_r or 0,
    )
    if max_layer < 1:
        return True
    bounds = _column_ranges_bounds(column_ranges)
    if bounds is None:
        return True

    for layer in range(1, max_layer + 1):
        checks: list[tuple[int, str]] = []
        if module_socle_layer_supports is not None and socle_filter_r is not None and layer <= socle_filter_r:
            checks.append((len(module_socle_layer_supports.get(layer, ())), "socle"))
        if (
            module_radical_layer_supports is not None
            and radical_filter_r is not None
            and layer <= radical_filter_r
        ):
            checks.append((len(module_radical_layer_supports.get(layer, ())), "radical"))

        for _support_size, filter_name in sorted(checks):
            if filter_name == "socle":
                if not _passes_socle_layer_from_ranges(
                    sinks,
                    column_ranges,
                    module_socle_layer_supports or {},
                    layer,
                    bounds=bounds,
                ):
                    return False
            else:
                if not _passes_radical_layer_from_ranges(
                    sources,
                    column_ranges,
                    module_radical_layer_supports or {},
                    layer,
                    bounds=bounds,
                ):
                    return False
    return True


def _passes_socle_layer_from_ranges(
    sinks: Sequence[GridPoint],
    column_ranges: Sequence[Optional[tuple[int, int]]],
    module_socle_layer_supports: Mapping[int, set[GridPoint]],
    layer: int,
    *,
    bounds: Optional[_ColumnRangeBounds] = None,
) -> bool:
    if layer < 1:
        return True
    height = layer - 1
    allowed = module_socle_layer_supports.get(layer, set())
    if bounds is None:
        bounds = _column_ranges_bounds(column_ranges)
    if bounds is None:
        return True

    seen: set[GridPoint] = set()
    for sink in sinks:
        sink_x, sink_y = sink
        y_start = max(bounds.min_y, sink_y - height)
        y_end = min(sink_y, bounds.max_y)
        x_start = max(bounds.min_x, sink_x - height)
        x_end = min(sink_x, bounds.max_x)
        if y_end - y_start <= x_end - x_start:
            iterator = (
                (sink_x - (height - (sink_y - y_coord)), y_coord)
                for y_coord in range(y_start, y_end + 1)
            )
        else:
            iterator = (
                (x_coord, sink_y - (height - (sink_x - x_coord)))
                for x_coord in range(x_start, x_end + 1)
            )
        for point in iterator:
            if _socle_shell_point_fails(
                point,
                sinks,
                column_ranges,
                allowed,
                layer,
                seen,
            ):
                return False
    return True


def _passes_radical_layer_supports_from_ranges(
    sources: Sequence[GridPoint],
    column_ranges: Sequence[Optional[tuple[int, int]]],
    module_radical_layer_supports: Mapping[int, set[GridPoint]],
    r: int,
) -> bool:
    if r < 1:
        return True
    bounds = _column_ranges_bounds(column_ranges)
    if bounds is None:
        return True
    for layer in range(1, r + 1):
        if not _passes_radical_layer_from_ranges(
            sources,
            column_ranges,
            module_radical_layer_supports,
            layer,
            bounds=bounds,
        ):
            return False
    return True


def _passes_radical_layer_from_ranges(
    sources: Sequence[GridPoint],
    column_ranges: Sequence[Optional[tuple[int, int]]],
    module_radical_layer_supports: Mapping[int, set[GridPoint]],
    layer: int,
    *,
    bounds: Optional[_ColumnRangeBounds] = None,
) -> bool:
    if layer < 1:
        return True
    depth = layer - 1
    allowed = module_radical_layer_supports.get(layer, set())
    if bounds is None:
        bounds = _column_ranges_bounds(column_ranges)
    if bounds is None:
        return True

    seen: set[GridPoint] = set()
    for source in sources:
        source_x, source_y = source
        y_start = max(bounds.min_y, source_y)
        y_end = min(bounds.max_y, source_y + depth)
        x_start = max(bounds.min_x, source_x)
        x_end = min(bounds.max_x, source_x + depth)
        if y_end - y_start <= x_end - x_start:
            iterator = (
                (source_x + (depth - (y_coord - source_y)), y_coord)
                for y_coord in range(y_start, y_end + 1)
            )
        else:
            iterator = (
                (x_coord, source_y + (depth - (x_coord - source_x)))
                for x_coord in range(x_start, x_end + 1)
            )
        for point in iterator:
            if _radical_shell_point_fails(
                point,
                sources,
                column_ranges,
                allowed,
                layer,
                seen,
            ):
                return False
    return True


def _socle_shell_point_fails(
    point: GridPoint,
    sinks: Sequence[GridPoint],
    column_ranges: Sequence[Optional[tuple[int, int]]],
    allowed: set[GridPoint],
    layer: int,
    seen: set[GridPoint],
) -> bool:
    x_coord, _y_coord = point
    if x_coord < 0 or x_coord >= len(column_ranges):
        return False
    if point in seen or not _point_in_column_ranges(point, column_ranges):
        return False
    seen.add(point)
    return _interval_height_from_sinks(point, sinks) == layer - 1 and point not in allowed


def _radical_shell_point_fails(
    point: GridPoint,
    sources: Sequence[GridPoint],
    column_ranges: Sequence[Optional[tuple[int, int]]],
    allowed: set[GridPoint],
    layer: int,
    seen: set[GridPoint],
) -> bool:
    x_coord, _y_coord = point
    if x_coord < 0 or x_coord >= len(column_ranges):
        return False
    if point in seen or not _point_in_column_ranges(point, column_ranges):
        return False
    seen.add(point)
    return _interval_depth_from_sources(point, sources) == layer - 1 and point not in allowed


def _generate_antichains(
    points: Iterable[GridPoint],
    max_antichains: Optional[int],
    *,
    adjacent_pair_allowed: Optional[Callable[[GridPoint, GridPoint], bool]] = None,
) -> tuple[tuple[GridPoint, ...], ...]:
    if max_antichains is not None and max_antichains < 0:
        raise ValueError("max_antichains must be nonnegative.")
    point_tuple = _normalize_points(points)
    antichains: list[tuple[GridPoint, ...]] = []

    def dfs(start_index: int, current: tuple[GridPoint, ...]) -> None:
        for point_index in range(start_index, len(point_tuple)):
            point = point_tuple[point_index]
            if current:
                previous = current[-1]
                if not (previous[0] < point[0] and previous[1] > point[1]):
                    continue
                if adjacent_pair_allowed is not None and not adjacent_pair_allowed(previous, point):
                    continue
            antichain = current + (point,)
            if max_antichains is not None and len(antichains) >= max_antichains:
                raise CandidateGenerationLimitError(
                    f"Antichain generation exceeded max_antichains={max_antichains}."
                )
            antichains.append(antichain)
            dfs(point_index + 1, antichain)

    dfs(0, ())
    return tuple(antichains)


def _source_sink_pair_feasible(sources: Sequence[GridPoint], sinks: Sequence[GridPoint]) -> bool:
    return all(any(_leq(source, sink) for sink in sinks) for source in sources) and all(
        any(_leq(source, sink) for source in sources) for sink in sinks
    )


def _interval_column_ranges(
    sources: Sequence[GridPoint],
    sinks: Sequence[GridPoint],
    bounds: GridBounds,
) -> Optional[tuple[Optional[tuple[int, int]], ...]]:
    summary = _interval_column_range_summary(sources, sinks, bounds)
    return None if summary is None else summary.column_ranges


def _interval_column_range_summary(
    sources: Sequence[GridPoint],
    sinks: Sequence[GridPoint],
    bounds: GridBounds,
) -> Optional[_ColumnRangeSummary]:
    for point in (*sources, *sinks):
        if not bounds.contains(point):
            return None
    width, height = bounds.dimensions
    if height < width:
        return _interval_column_range_summary_by_rows(sources, sinks, bounds)
    return _interval_column_range_summary_by_columns(sources, sinks, bounds)


def _interval_column_range_summary_by_columns(
    sources: Sequence[GridPoint],
    sinks: Sequence[GridPoint],
    bounds: GridBounds,
) -> Optional[_ColumnRangeSummary]:
    source_by_x = sorted(sources)
    sink_by_x = sorted(sinks)
    max_x, _max_y = bounds.max_grade
    breakpoints = {0, max_x + 1}
    breakpoints.update(source[0] for source in source_by_x)
    breakpoints.update(sink[0] + 1 for sink in sink_by_x if sink[0] + 1 <= max_x + 1)
    ordered_breakpoints = sorted(breakpoints)
    segments = []
    for start, end in zip(ordered_breakpoints, ordered_breakpoints[1:]):
        if start >= end:
            continue
        lower_y = _segment_lower_y(start, source_by_x)
        upper_y = _segment_upper_y(start, sink_by_x)
        value = (
            (lower_y, upper_y)
            if lower_y is not None and upper_y is not None and lower_y <= upper_y
            else None
        )
        segments.append((start, end, value))
    return _summary_from_value_segments(segments, max_x)


def _interval_column_range_summary_by_rows(
    sources: Sequence[GridPoint],
    sinks: Sequence[GridPoint],
    bounds: GridBounds,
) -> Optional[_ColumnRangeSummary]:
    max_x, max_y = bounds.max_grade
    row_intervals = []
    for y_coord in range(max_y + 1):
        left_x = None
        for source_x, source_y in sources:
            if source_y <= y_coord:
                left_x = source_x if left_x is None else min(left_x, source_x)
        if left_x is None:
            continue

        right_x = None
        for sink_x, sink_y in sinks:
            if y_coord <= sink_y:
                right_x = sink_x if right_x is None else max(right_x, sink_x)
        if right_x is None or left_x > right_x:
            continue
        row_intervals.append((y_coord, left_x, right_x))

    if not row_intervals:
        return None

    breakpoints = {0, max_x + 1}
    for _y_coord, left_x, right_x in row_intervals:
        breakpoints.add(left_x)
        if right_x + 1 <= max_x + 1:
            breakpoints.add(right_x + 1)

    ordered_breakpoints = sorted(breakpoints)
    segments = []
    for start, end in zip(ordered_breakpoints, ordered_breakpoints[1:]):
        if start >= end:
            continue
        active_rows = [
            y_coord
            for y_coord, left_x, right_x in row_intervals
            if left_x <= start <= right_x
        ]
        value = (min(active_rows), max(active_rows)) if active_rows else None
        segments.append((start, end, value))
    return _summary_from_value_segments(segments, max_x)


def _segment_lower_y(
    x_coord: int,
    sources: Sequence[GridPoint],
) -> Optional[int]:
    lower_y = None
    for source_x, source_y in sources:
        if source_x <= x_coord:
            lower_y = source_y if lower_y is None else min(lower_y, source_y)
    return lower_y


def _segment_upper_y(
    x_coord: int,
    sinks: Sequence[GridPoint],
) -> Optional[int]:
    upper_y = None
    for sink_x, sink_y in sinks:
        if x_coord <= sink_x:
            upper_y = sink_y if upper_y is None else max(upper_y, sink_y)
    return upper_y


def _summary_from_value_segments(
    segments: Sequence[tuple[int, int, Optional[tuple[int, int]]]],
    max_x: int,
) -> Optional[_ColumnRangeSummary]:
    ranges: list[Optional[tuple[int, int]]] = []
    has_nonempty_column = False
    sources = []
    best_lower = None
    sink_candidates = []
    previous_range = None
    connected = True

    current_x = 0
    for start, end, value in segments:
        if start > current_x:
            ranges.extend([None] * (start - current_x))
        length = end - start
        if value is None:
            ranges.extend([None] * length)
        else:
            lower_y, upper_y = value
            ranges.extend([value] * length)
            has_nonempty_column = True
            if best_lower is None or lower_y < best_lower:
                sources.append((start, lower_y))
                best_lower = lower_y
            sink_candidates.append((end - 1, upper_y))
            if previous_range is not None:
                previous_lower, _previous_upper = previous_range
                if previous_lower > upper_y:
                    connected = False
            previous_range = value
        current_x = end
    if current_x <= max_x:
        ranges.extend([None] * (max_x + 1 - current_x))

    if not has_nonempty_column:
        return None

    sinks = []
    best_upper = None
    for x_coord, upper_y in reversed(sink_candidates):
        if best_upper is None or upper_y > best_upper:
            sinks.append((x_coord, upper_y))
            best_upper = upper_y

    return _ColumnRangeSummary(
        column_ranges=tuple(ranges),
        sources=tuple(sources),
        sinks=tuple(sorted(sinks)),
        connected=connected,
    )


def _proper_sources_sinks_from_extrema_2d(
    sources: Sequence[GridPoint],
    sinks: Sequence[GridPoint],
    bounds: GridBounds,
) -> tuple[tuple[GridPoint, ...], tuple[GridPoint, ...]]:
    max_x, max_y = bounds.max_grade
    y_bound = max_y + 1
    raw_segments = _lower_upper_segments(sources, sinks, bounds)

    src_uuset = []
    best_up_y = y_bound
    for start, _end, lower_y, upper_y in raw_segments:
        if lower_y is None:
            continue
        if upper_y is None or lower_y > upper_y:
            threshold = lower_y
        else:
            threshold = upper_y + 1
        if threshold < y_bound and threshold < best_up_y:
            src_uuset.append((start, threshold))
            best_up_y = threshold

    snk_ddset = []
    best_down_y = -1
    for start, end, lower_y, upper_y in reversed(raw_segments):
        if upper_y is None:
            continue
        if lower_y is None or lower_y > upper_y:
            threshold = upper_y
        else:
            threshold = lower_y - 1
        if threshold >= 0 and threshold > best_down_y:
            snk_ddset.append((end - 1, threshold))
            best_down_y = threshold

    return tuple(src_uuset), tuple(sorted(snk_ddset))


def _lower_upper_segments(
    sources: Sequence[GridPoint],
    sinks: Sequence[GridPoint],
    bounds: GridBounds,
) -> tuple[tuple[int, int, Optional[int], Optional[int]], ...]:
    max_x, _max_y = bounds.max_grade
    breakpoints = {0, max_x + 1}
    breakpoints.update(source[0] for source in sources)
    breakpoints.update(sink[0] + 1 for sink in sinks if sink[0] + 1 <= max_x + 1)
    ordered_breakpoints = sorted(breakpoints)
    segments = []
    for start, end in zip(ordered_breakpoints, ordered_breakpoints[1:]):
        if start >= end:
            continue
        segments.append(
            (
                start,
                end,
                _segment_lower_y(start, sources),
                _segment_upper_y(start, sinks),
            )
        )
    return tuple(segments)


def _is_canonical_interval_extrema(
    sources: Sequence[GridPoint],
    sinks: Sequence[GridPoint],
    column_ranges: Sequence[Optional[tuple[int, int]]],
) -> bool:
    if not _is_antichain(sources) or not _is_antichain(sinks):
        return False
    if not _column_ranges_connected(column_ranges):
        return False
    return _column_range_sources(column_ranges) == tuple(sources) and _column_range_sinks(
        column_ranges
    ) == tuple(sinks)


def _column_range_sources(
    column_ranges: Sequence[Optional[tuple[int, int]]],
) -> tuple[GridPoint, ...]:
    sources = []
    best_lower = None
    for x_coord, value in enumerate(column_ranges):
        if value is None:
            continue
        lower_y, _upper_y = value
        if best_lower is None or lower_y < best_lower:
            sources.append((x_coord, lower_y))
            best_lower = lower_y
    return tuple(sources)


def _column_range_sinks(
    column_ranges: Sequence[Optional[tuple[int, int]]],
) -> tuple[GridPoint, ...]:
    sinks = []
    best_upper = None
    for x_coord in range(len(column_ranges) - 1, -1, -1):
        value = column_ranges[x_coord]
        if value is None:
            continue
        _lower_y, upper_y = value
        if best_upper is None or upper_y > best_upper:
            sinks.append((x_coord, upper_y))
            best_upper = upper_y
    return tuple(sorted(sinks))


def _column_ranges_connected(column_ranges: Sequence[Optional[tuple[int, int]]]) -> bool:
    previous_range = None
    has_nonempty_column = False
    for value in column_ranges:
        if value is None:
            continue
        has_nonempty_column = True
        if previous_range is not None:
            previous_lower, _previous_upper = previous_range
            _current_lower, current_upper = value
            if previous_lower > current_upper:
                return False
        previous_range = value
    if not has_nonempty_column:
        return False
    return True


def _points_from_column_ranges(
    column_ranges: Sequence[Optional[tuple[int, int]]],
) -> tuple[GridPoint, ...]:
    points = []
    for x_coord, value in enumerate(column_ranges):
        if value is None:
            continue
        lower_y, upper_y = value
        points.extend((x_coord, y_coord) for y_coord in range(lower_y, upper_y + 1))
    return tuple(points)


def _interval_height_supports_up_to(
    candidate: IntervalCandidate,
    max_height: int,
) -> dict[int, set[GridPoint]]:
    if max_height < 0:
        return {}
    if not candidate.snk:
        return {}

    points_to_check: set[GridPoint] = set()
    for sink in candidate.snk:
        for dx in range(max_height + 1):
            x_coord = sink[0] - dx
            if x_coord < 0 or x_coord >= len(candidate.column_ranges):
                continue
            remaining = max_height - dx
            for dy in range(remaining + 1):
                point = (x_coord, sink[1] - dy)
                if _point_in_column_ranges(point, candidate.column_ranges):
                    points_to_check.add(point)

    supports: dict[int, set[GridPoint]] = {}
    for point in points_to_check:
        height = _interval_height_from_sinks(point, candidate.snk)
        if 0 <= height <= max_height:
            supports.setdefault(height, set()).add(point)
    return supports


def _interval_depth_supports_up_to(
    candidate: IntervalCandidate,
    max_depth: int,
) -> dict[int, set[GridPoint]]:
    if max_depth < 0:
        return {}
    if not candidate.src:
        return {}

    points_to_check: set[GridPoint] = set()
    for source in candidate.src:
        for dx in range(max_depth + 1):
            x_coord = source[0] + dx
            if x_coord < 0 or x_coord >= len(candidate.column_ranges):
                continue
            remaining = max_depth - dx
            for dy in range(remaining + 1):
                point = (x_coord, source[1] + dy)
                if _point_in_column_ranges(point, candidate.column_ranges):
                    points_to_check.add(point)

    supports: dict[int, set[GridPoint]] = {}
    for point in points_to_check:
        depth = _interval_depth_from_sources(point, candidate.src)
        if 0 <= depth <= max_depth:
            supports.setdefault(depth, set()).add(point)
    return supports


def _normalize_reachability_regions(
    reachability_regions: Optional[Mapping[GridPoint, Iterable[GridPoint]]],
) -> Optional[dict[GridPoint, frozenset[GridPoint]]]:
    if reachability_regions is None:
        return None
    return {
        _as_grid_point(omega): frozenset(_normalize_points(points))
        for omega, points in reachability_regions.items()
    }


def _require_reachability_regions_for_sinks(
    sinks: Sequence[GridPoint],
    reachability_regions: Mapping[GridPoint, frozenset[GridPoint]],
) -> None:
    missing = [sink for sink in sinks if sink not in reachability_regions]
    if missing:
        raise ValueError(f"Missing reachability regions for sink grades: {_format_points(missing)}.")


def _point_reachable_to_sinks(
    point: GridPoint,
    sinks: Sequence[GridPoint],
    reachability_regions: Mapping[GridPoint, frozenset[GridPoint]],
) -> bool:
    return all(
        not _leq(point, sink) or point in reachability_regions[sink]
        for sink in sinks
    )


def _sinks_share_reachability_component(
    sinks: Sequence[GridPoint],
    module_support: set[GridPoint],
    point_allowed: Callable[[GridPoint], bool],
) -> bool:
    if not sinks:
        return False
    sink_set = set(sinks)
    if not all(sink in module_support and point_allowed(sink) for sink in sink_set):
        return False

    start = sinks[0]
    visited = {start}
    stack = [start]
    while stack:
        x_coord, y_coord = stack.pop()
        for neighbor in (
            (x_coord - 1, y_coord),
            (x_coord + 1, y_coord),
            (x_coord, y_coord - 1),
            (x_coord, y_coord + 1),
        ):
            if neighbor in visited or neighbor not in module_support:
                continue
            if not point_allowed(neighbor):
                continue
            visited.add(neighbor)
            stack.append(neighbor)
    return sink_set.issubset(visited)


def _source_candidates_for_submodule_family(
    sinks: tuple[GridPoint, ...],
    bounds: GridBounds,
    module_socle_layer_supports: dict[int, set[GridPoint]],
    r: int,
    *,
    point_allowed: Optional[Callable[[GridPoint], bool]] = None,
) -> tuple[GridPoint, ...]:
    allowed_points = _socle_allowed_points_for_sinks(sinks, bounds, module_socle_layer_supports, r)
    if not allowed_points:
        return ()
    if point_allowed is not None:
        allowed_points = {point for point in allowed_points if point_allowed(point)}
        if not allowed_points:
            return ()

    allowed_prefix = _grid_prefix_sum(allowed_points, bounds)
    source_candidates = []
    for point in sorted(allowed_points):
        comparable_sinks = tuple(sink for sink in sinks if _leq(point, sink))
        if not comparable_sinks:
            continue
        if all(_rectangle_is_full(allowed_prefix, point, sink) for sink in comparable_sinks):
            source_candidates.append(point)
    return tuple(source_candidates)


def _iter_concrete_intervals_from_family(
    family: SubmoduleIntervalFamily,
    bounds: GridBounds,
    *,
    source_mode: str,
) -> Iterable[ConcreteSubmoduleInterval]:
    if source_mode == "minimal_covers":
        source_iterable = _generate_minimal_cover_source_antichains(
            family.source_candidates,
            family.sinks,
        )
    elif source_mode == "single_source":
        source_iterable = ((point,) for point in family.source_candidates)
    elif source_mode == "all_antichains":
        source_iterable = _iter_antichains(family.source_candidates)
    else:
        raise ValueError(f"Unknown source_mode: {source_mode!r}.")

    seen: set[tuple[tuple[GridPoint, ...], tuple[GridPoint, ...]]] = set()
    for sources in source_iterable:
        if not _source_sink_pair_feasible(sources, family.sinks):
            continue
        column_ranges = _interval_column_ranges(sources, family.sinks, bounds)
        if column_ranges is None:
            continue
        if not _is_canonical_interval_extrema(sources, family.sinks, column_ranges):
            continue
        key = (sources, family.sinks)
        if key in seen:
            continue
        seen.add(key)
        yield ConcreteSubmoduleInterval(
            src=sources,
            snk=family.sinks,
            column_ranges=column_ranges,
        )


def _generate_minimal_cover_source_antichains(
    source_candidates: Iterable[GridPoint],
    sinks: tuple[GridPoint, ...],
) -> Iterable[tuple[GridPoint, ...]]:
    sink_count = len(sinks)
    if sink_count == 0:
        return

    full_mask = (1 << sink_count) - 1
    point_masks = []
    for point in _normalize_points(source_candidates):
        mask = 0
        point_x, point_y = point
        for sink_index, (sink_x, sink_y) in enumerate(sinks):
            if point_x <= sink_x and point_y <= sink_y:
                mask |= 1 << sink_index
        if mask:
            point_masks.append((point, mask))

    def dfs(start_index: int, current: tuple[GridPoint, ...], coverage: int) -> Iterable[tuple[GridPoint, ...]]:
        if coverage == full_mask:
            if _is_minimal_sink_cover(current, sinks):
                yield current
            return
        if len(current) >= sink_count:
            return

        for point_index in range(start_index, len(point_masks)):
            point, mask = point_masks[point_index]
            new_coverage = coverage | mask
            if new_coverage == coverage:
                continue
            if current:
                previous = current[-1]
                if not (previous[0] < point[0] and previous[1] > point[1]):
                    continue
            yield from dfs(point_index + 1, current + (point,), new_coverage)

    yield from dfs(0, (), 0)


def _iter_antichains(points: Iterable[GridPoint]) -> Iterable[tuple[GridPoint, ...]]:
    point_tuple = _normalize_points(points)

    def dfs(start_index: int, current: tuple[GridPoint, ...]) -> Iterable[tuple[GridPoint, ...]]:
        for point_index in range(start_index, len(point_tuple)):
            point = point_tuple[point_index]
            if current:
                previous = current[-1]
                if not (previous[0] < point[0] and previous[1] > point[1]):
                    continue
            antichain = current + (point,)
            yield antichain
            yield from dfs(point_index + 1, antichain)

    yield from dfs(0, ())


def _is_minimal_sink_cover(
    sources: Sequence[GridPoint],
    sinks: Sequence[GridPoint],
) -> bool:
    if not sources:
        return False
    for source_index, source in enumerate(sources):
        has_unique_sink = False
        for sink in sinks:
            if not _leq(source, sink):
                continue
            if not any(
                other_index != source_index and _leq(other, sink)
                for other_index, other in enumerate(sources)
            ):
                has_unique_sink = True
                break
        if not has_unique_sink:
            return False
    return True


@dataclass
class _InjectiveRankContext:
    copresentation: InjectiveCopresentationData

    def __post_init__(self) -> None:
        self.column_masks = _column_row_masks(self.copresentation.matrix_columns)
        self.q0_masks_by_x = _grade_bit_masks_by_x(
            self.copresentation.q0_grades,
            self.copresentation.bounds,
        )
        self.q1_masks_by_x = _grade_bit_masks_by_x(
            self.copresentation.q1_grades,
            self.copresentation.bounds,
        )
        self.q0_active_masks_by_x = _active_grade_masks_by_x(
            self.copresentation.q0_grades,
            self.copresentation.bounds,
        )
        self.q1_active_masks_by_x = _active_grade_masks_by_x(
            self.copresentation.q1_grades,
            self.copresentation.bounds,
        )
        self.q0_point_masks = _grade_point_masks(self.copresentation.q0_grades)
        self.rank_cache: dict[tuple[int, int], int] = {}
        self.rank_test_cache: dict[tuple[int, int, tuple[GridPoint, ...]], bool] = {}


def _passes_injective_rank_test(
    candidate: ConcreteSubmoduleInterval | IntervalCandidate,
    context: _InjectiveRankContext,
) -> bool:
    row_mask = _grade_mask_in_column_ranges(candidate.column_ranges, context.q1_masks_by_x)
    column_mask = _grade_mask_in_column_ranges(candidate.column_ranges, context.q0_masks_by_x)
    if not column_mask:
        return False

    test_key = (row_mask, column_mask, candidate.snk)
    cached = context.rank_test_cache.get(test_key)
    if cached is not None:
        return cached

    rank_all = _restricted_column_rank(row_mask, column_mask, context)
    for sink in candidate.snk:
        omega_mask = context.q0_point_masks.get(sink, 0) & column_mask
        if not omega_mask:
            context.rank_test_cache[test_key] = False
            return False
        rank_without = _restricted_column_rank(row_mask, column_mask & ~omega_mask, context)
        if rank_all >= rank_without + omega_mask.bit_count():
            context.rank_test_cache[test_key] = False
            return False
    context.rank_test_cache[test_key] = True
    return True


def _restricted_column_rank(
    row_mask: int,
    column_mask: int,
    context: _InjectiveRankContext,
) -> int:
    key = (row_mask, column_mask)
    cached = context.rank_cache.get(key)
    if cached is not None:
        return cached
    basis = _GF2IncrementalRank()
    remaining_columns = column_mask
    while remaining_columns:
        column_bit = remaining_columns & -remaining_columns
        column_index = column_bit.bit_length() - 1
        basis.add(context.column_masks[column_index] & row_mask)
        remaining_columns ^= column_bit
    context.rank_cache[key] = basis.rank
    return basis.rank


def _socle_allowed_points_for_sinks(
    sinks: tuple[GridPoint, ...],
    bounds: GridBounds,
    module_socle_layer_supports: dict[int, set[GridPoint]],
    r: int,
) -> set[GridPoint]:
    allowed_points: set[GridPoint] = set()
    if not sinks:
        return allowed_points

    max_x, max_y = bounds.max_grade
    support_for_layer = module_socle_layer_supports.get
    if len(sinks) == 1:
        sink_x, sink_y = sinks[0]
        for dx in range(r + 1):
            x_coord = sink_x - dx
            if x_coord < 0 or x_coord > max_x:
                continue
            for dy in range(r - dx + 1):
                y_coord = sink_y - dy
                if y_coord < 0:
                    break
                if y_coord > max_y:
                    continue
                height = dx + dy
                point = (x_coord, y_coord)
                if point in support_for_layer(height + 1, ()):
                    allowed_points.add(point)
        return allowed_points

    for sink in sinks:
        sink_x, sink_y = sink
        for dx in range(r + 1):
            x_coord = sink_x - dx
            if x_coord < 0 or x_coord > max_x:
                continue
            for dy in range(r - dx + 1):
                y_coord = sink_y - dy
                if y_coord < 0:
                    break
                if y_coord > max_y:
                    continue
                point = (x_coord, y_coord)
                height = _height_from_sinks_fast(point, sinks)
                if 0 <= height <= r and point in module_socle_layer_supports.get(height + 1, ()):
                    allowed_points.add(point)
    return allowed_points


def _height_from_sinks_fast(
    point: GridPoint,
    sinks: Sequence[GridPoint],
) -> int:
    point_x, point_y = point
    height = -1
    for sink_x, sink_y in sinks:
        if point_x <= sink_x and point_y <= sink_y:
            distance = (sink_x - point_x) + (sink_y - point_y)
            if distance > height:
                height = distance
    return height


def _interval_height_from_sinks(
    point: GridPoint,
    sinks: Sequence[GridPoint],
) -> int:
    height = -1
    for sink in sinks:
        if _leq(point, sink):
            height = max(height, (sink[0] - point[0]) + (sink[1] - point[1]))
    return height


def _interval_depth_from_sources(
    point: GridPoint,
    sources: Sequence[GridPoint],
) -> int:
    depth = -1
    for source in sources:
        if _leq(source, point):
            depth = max(depth, (point[0] - source[0]) + (point[1] - source[1]))
    return depth


def _grid_prefix_sum(
    points: set[GridPoint],
    bounds: GridBounds,
) -> list[list[int]]:
    max_x, max_y = bounds.max_grade
    prefix = [[0] * (max_y + 2) for _ in range(max_x + 2)]
    for x_coord, y_coord in points:
        if bounds.contains((x_coord, y_coord)):
            prefix[x_coord + 1][y_coord + 1] += 1
    for x_coord in range(max_x + 1):
        running = 0
        for y_coord in range(max_y + 1):
            running += prefix[x_coord + 1][y_coord + 1]
            prefix[x_coord + 1][y_coord + 1] = (
                prefix[x_coord][y_coord + 1] + running
            )
    return prefix


def _rectangle_is_full(
    prefix: Sequence[Sequence[int]],
    lower: GridPoint,
    upper: GridPoint,
) -> bool:
    if not _leq(lower, upper):
        return False
    x0, y0 = lower
    x1, y1 = upper
    actual = (
        prefix[x1 + 1][y1 + 1]
        - prefix[x0][y1 + 1]
        - prefix[x1 + 1][y0]
        + prefix[x0][y0]
    )
    expected = (x1 - x0 + 1) * (y1 - y0 + 1)
    return actual == expected


def _point_in_column_ranges(
    point: GridPoint,
    column_ranges: Sequence[Optional[tuple[int, int]]],
) -> bool:
    x_coord, y_coord = point
    if x_coord < 0 or x_coord >= len(column_ranges):
        return False
    value = column_ranges[x_coord]
    if value is None:
        return False
    lower_y, upper_y = value
    return lower_y <= y_coord <= upper_y


def _grade_point_masks(grades: Sequence[GridPoint]) -> dict[GridPoint, int]:
    point_masks: dict[GridPoint, int] = {}
    for index, grade in enumerate(grades):
        point_masks[grade] = point_masks.get(grade, 0) | (1 << index)
    return point_masks


def _grade_bit_masks_by_x(
    grades: Sequence[GridPoint],
    bounds: GridBounds,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    by_x: list[list[tuple[int, int]]] = [[] for _ in range(bounds.max_grade[0] + 1)]
    for index, grade in enumerate(grades):
        x_coord, y_coord = grade
        if 0 <= x_coord <= bounds.max_grade[0]:
            by_x[x_coord].append((y_coord, 1 << index))
    return tuple(tuple(sorted(entries)) for entries in by_x)


def _active_grade_masks_by_x(
    grades: Sequence[GridPoint],
    bounds: GridBounds,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    exact_by_x = _grade_bit_masks_by_x(grades, bounds)
    active_by_x: list[tuple[tuple[int, int], ...]] = []
    for x_coord in range(bounds.max_grade[0] + 1):
        y_masks: dict[int, int] = {}
        for exact_x in range(x_coord, bounds.max_grade[0] + 1):
            for y_coord, point_mask in exact_by_x[exact_x]:
                y_masks[y_coord] = y_masks.get(y_coord, 0) | point_mask
        active_by_x.append(tuple(sorted(y_masks.items())))
    return tuple(active_by_x)


def _active_grade_mask(
    point: GridPoint,
    active_masks_by_x: Sequence[Sequence[tuple[int, int]]],
) -> int:
    x_coord, y_coord = point
    if x_coord < 0 or x_coord >= len(active_masks_by_x):
        return 0
    mask = 0
    for grade_y, point_mask in active_masks_by_x[x_coord]:
        if grade_y >= y_coord:
            mask |= point_mask
    return mask


def _grade_mask_in_column_ranges(
    column_ranges: Sequence[Optional[tuple[int, int]]],
    masks_by_x: Sequence[Sequence[tuple[int, int]]],
) -> int:
    mask = 0
    for x_coord, value in enumerate(column_ranges):
        if value is None or x_coord >= len(masks_by_x):
            continue
        lower_y, upper_y = value
        for y_coord, point_mask in masks_by_x[x_coord]:
            if y_coord < lower_y:
                continue
            if y_coord > upper_y:
                break
            mask |= point_mask
    return mask


def _column_ranges_size(column_ranges: Sequence[Optional[tuple[int, int]]]) -> int:
    size = 0
    for value in column_ranges:
        if value is None:
            continue
        lower_y, upper_y = value
        size += upper_y - lower_y + 1
    return size


def _column_ranges_bounds(
    column_ranges: Sequence[Optional[tuple[int, int]]],
) -> Optional[_ColumnRangeBounds]:
    min_x = None
    max_x = None
    min_y = None
    max_y = None
    for x_coord, value in enumerate(column_ranges):
        if value is None:
            continue
        lower_y, upper_y = value
        min_x = x_coord if min_x is None else min(min_x, x_coord)
        max_x = x_coord if max_x is None else max(max_x, x_coord)
        min_y = lower_y if min_y is None else min(min_y, lower_y)
        max_y = upper_y if max_y is None else max(max_y, upper_y)
    if min_x is None or max_x is None or min_y is None or max_y is None:
        return None
    return _ColumnRangeBounds(min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)


def _column_ranges_max_y(column_ranges: Sequence[Optional[tuple[int, int]]]) -> Optional[int]:
    max_y = None
    for value in column_ranges:
        if value is None:
            continue
        _lower_y, upper_y = value
        max_y = upper_y if max_y is None else max(max_y, upper_y)
    return max_y


def _is_antichain(points: Sequence[GridPoint]) -> bool:
    for left, right in combinations(points, 2):
        if _leq(left, right) or _leq(right, left):
            return False
    return True


def _passes_criterion(
    candidate: IntervalCandidate,
    support_data: ResolutionSupportData,
    criterion: str,
) -> bool:
    if criterion == "crt0":
        return passes_crt0(candidate, support_data)
    if criterion in {"crt1", "crt1_2d"}:
        return passes_crt1_2d(candidate, support_data)
    raise ValueError(f"Unknown criterion: {criterion!r}.")


def _ensure_candidate(
    interval: Interval | IntervalCandidate | tuple[Iterable[GridPoint], Iterable[GridPoint]],
    support_data: ResolutionSupportData,
) -> IntervalCandidate:
    if isinstance(interval, IntervalCandidate):
        return interval
    if isinstance(interval, Interval):
        return build_interval_candidate(interval, support_data.bounds)
    if isinstance(interval, tuple) and len(interval) == 2:
        return build_interval_candidate(Interval(interval[0], interval[1]), support_data.bounds)
    raise TypeError(f"Unsupported interval value: {interval!r}.")


def _parse_resolution_txt(path: PathLike) -> _ParsedResolution:
    path = Path(path)
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    meaningful_lines = []
    for raw_line in raw_lines:
        content = raw_line.partition("#")[0].strip()
        if content:
            meaningful_lines.append(content)

    if len(meaningful_lines) < 3 or meaningful_lines[0] != "scc2020":
        raise CandidateParseError(f"{path} is not an scc2020-style finite-grid txt file.")
    try:
        num_parameters = int(meaningful_lines[1])
    except ValueError as exc:
        raise CandidateParseError(f"Invalid parameter count in {path}.") from exc
    if num_parameters != 2:
        raise CandidateParseError(f"Only 2-parameter outputs are supported, got {num_parameters}.")

    field = "GF(2)"
    index = 2
    while index < len(meaningful_lines) and meaningful_lines[index].startswith("--"):
        field = _parse_option_line(meaningful_lines[index], field)
        index += 1
    if index >= len(meaningful_lines):
        raise CandidateParseError(f"Missing generator sizes in {path}.")
    generator_sizes = tuple(int(token) for token in meaningful_lines[index].split())
    index += 1
    while index < len(meaningful_lines) and meaningful_lines[index].startswith("--"):
        field = _parse_option_line(meaningful_lines[index], field)
        index += 1

    data_lines = meaningful_lines[index:]
    expected_data_lines = sum(generator_sizes)
    if len(data_lines) != expected_data_lines:
        raise CandidateParseError(
            f"{path} has {len(data_lines)} data lines, expected {expected_data_lines}."
        )

    offset = 0
    generator_grades: list[tuple[GridPoint, ...]] = []
    matrix_columns: list[tuple[tuple[int, ...], ...]] = []
    for block_index, size in enumerate(generator_sizes[:-1]):
        grades = []
        columns = []
        target_size = generator_sizes[block_index + 1]
        for _ in range(size):
            line = data_lines[offset]
            grades.append(_parse_column_grade(line, num_parameters))
            columns.append(_parse_column_rows(line, num_parameters, target_size))
            offset += 1
        generator_grades.append(tuple(grades))
        matrix_columns.append(tuple(columns))

    final_grades = []
    if generator_sizes:
        for _ in range(generator_sizes[-1]):
            final_grades.append(_parse_grade_only_line(data_lines[offset], num_parameters))
            offset += 1
    generator_grades.append(tuple(final_grades))

    grid_min = _parse_grid_comment(raw_lines, "grid min")
    grid_max = _parse_grid_comment(raw_lines, "grid max")
    if grid_min is None:
        raise CandidateParseError(f"Missing '# grid min:' comment in {path}.")
    if grid_max is None:
        raise CandidateParseError(f"Missing '# grid max:' comment in {path}.")

    return _ParsedResolution(
        path=path,
        num_parameters=num_parameters,
        generator_sizes=generator_sizes,
        generator_grades=tuple(generator_grades),
        matrix_columns=tuple(matrix_columns),
        grid_min=grid_min,
        grid_max=grid_max,
        field=field,
    )


def _parse_column_grade(line: str, num_parameters: int) -> GridPoint:
    tokens = line.replace(";", " ; ").split()
    if len(tokens) < num_parameters:
        raise CandidateParseError(f"Missing grade in line: {line!r}.")
    return _parse_grade_tokens(tokens[:num_parameters])


def _parse_column_rows(line: str, num_parameters: int, target_size: int) -> tuple[int, ...]:
    tokens = line.replace(";", " ; ").split()
    entries = tokens[num_parameters:]
    if entries and entries[0] == ";":
        entries = entries[1:]
    rows: set[int] = set()
    for entry in entries:
        if entry == ";":
            continue
        row_token, separator, coeff_token = entry.partition(":")
        try:
            row = int(row_token) - 1
            coeff = int(coeff_token) if separator else 1
        except ValueError as exc:
            raise CandidateParseError(f"Invalid matrix entry {entry!r} in line {line!r}.") from exc
        if row < 0 or row >= target_size:
            raise CandidateParseError(f"Row index {row + 1} is outside 1..{target_size} in line {line!r}.")
        if coeff % 2:
            if row in rows:
                rows.remove(row)
            else:
                rows.add(row)
    return tuple(sorted(rows))


def _parse_option_line(line: str, field: str) -> str:
    tokens = line.split()
    if not tokens:
        return field
    if tokens[0] == "--field":
        return " ".join(tokens[1:])
    return field


def _parse_grade_only_line(line: str, num_parameters: int) -> GridPoint:
    tokens = line.split()
    if len(tokens) != num_parameters:
        raise CandidateParseError(f"Expected a grade-only line, got: {line!r}.")
    return _parse_grade_tokens(tokens)


def _parse_grade_tokens(tokens: Sequence[str]) -> GridPoint:
    if len(tokens) != 2:
        raise CandidateParseError(f"Expected a 2D grade, got {tokens!r}.")
    try:
        return (int(tokens[0]), int(tokens[1]))
    except ValueError as exc:
        raise CandidateParseError(f"Finite-grid grades must be integers, got {tokens!r}.") from exc


def _parse_points_field(value: str) -> tuple[GridPoint, ...]:
    value = value.strip()
    if not value or value == "-":
        return ()
    points = tuple((int(match.group(1)), int(match.group(2))) for match in POINT_PATTERN.finditer(value))
    if not points:
        raise CandidateParseError(f"Could not parse point field: {value!r}.")
    return _normalize_points(points)


def _parse_grid_comment(raw_lines: Sequence[str], key: str) -> Optional[GridPoint]:
    prefix = f"{key}:"
    for raw_line in raw_lines:
        stripped = raw_line.strip()
        if not stripped.startswith("#"):
            continue
        comment = stripped[1:].strip()
        if comment.startswith(prefix):
            return _parse_grade_tokens(comment[len(prefix) :].split())
    return None


def _require_compressed_grid(resolution: _ParsedResolution) -> None:
    if resolution.grid_min != (0, 0):
        raise CandidateParseError(
            f"{resolution.path} has grid min {resolution.grid_min}; expected compressed grid min (0, 0)."
        )
    GridBounds(resolution.grid_max)


def _as_grid_point(point: Sequence[int]) -> GridPoint:
    if len(point) != 2:
        raise ValueError(f"Expected a 2D grid point, got {point!r}.")
    return (int(point[0]), int(point[1]))


def _normalize_points_with_multiplicity(points: Iterable[Sequence[int]]) -> tuple[GridPoint, ...]:
    return tuple(_as_grid_point(point) for point in points)


def _normalize_points(points: Iterable[Sequence[int]]) -> tuple[GridPoint, ...]:
    return tuple(sorted({_as_grid_point(point) for point in points}))


def _gridsize(bounds: GridBounds) -> tuple[int, int]:
    return bounds.dimensions


def _leq(left: GridPoint, right: GridPoint) -> bool:
    return left[0] <= right[0] and left[1] <= right[1]


def _join_grid(left: GridPoint, right: GridPoint) -> GridPoint:
    return (max(left[0], right[0]), max(left[1], right[1]))


def _meet_grid(left: GridPoint, right: GridPoint) -> GridPoint:
    return (min(left[0], right[0]), min(left[1], right[1]))


def _is_connected_points(points: Sequence[GridPoint]) -> bool:
    if not points:
        return False
    point_tuple = tuple(points)
    visited = {point_tuple[0]}
    stack = [point_tuple[0]]
    while stack:
        point = stack.pop()
        for other in point_tuple:
            if other not in visited and (_leq(point, other) or _leq(other, point)):
                visited.add(other)
                stack.append(other)
    return len(visited) == len(point_tuple)


def _l1_distance(left: GridPoint, right: GridPoint) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _column_row_masks(columns: Sequence[Sequence[int]]) -> tuple[int, ...]:
    masks = []
    for rows in columns:
        mask = 0
        for row in rows:
            mask |= 1 << row
        masks.append(mask)
    return tuple(masks)


def _row_column_masks(row_count: int, columns: Sequence[Sequence[int]]) -> tuple[int, ...]:
    masks = [0] * row_count
    for column_index, rows in enumerate(columns):
        for row in rows:
            masks[row] |= 1 << column_index
    return tuple(masks)


def _active_row_mask(point: GridPoint, row_grades: Sequence[GridPoint]) -> int:
    mask = 0
    for row_index, grade in enumerate(row_grades):
        if _leq(point, grade):
            mask |= 1 << row_index
    return mask


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


def _default_output_path(projective_txt: PathLike) -> Path:
    path = Path(projective_txt)
    suffix = ".finite_grid.projective_resolution.txt"
    name = path.name
    prefix = name[: -len(suffix)] if name.endswith(suffix) else path.stem
    return Path("outputs") / "interval_candidates" / f"{prefix}.interval_candidates.txt"


def _format_point_plain(point: GridPoint) -> str:
    return f"{point[0]} {point[1]}"


def _format_point(point: GridPoint) -> str:
    return f"({point[0]}, {point[1]})"


def _format_points(points: Sequence[GridPoint]) -> str:
    if not points:
        return "-"
    return " ".join(_format_point(point) for point in points)


def _format_point_column_ranges(points: Sequence[GridPoint]) -> str:
    if not points:
        return "-"
    by_x: dict[int, list[int]] = {}
    for x_coord, y_coord in _normalize_points(points):
        by_x.setdefault(x_coord, []).append(y_coord)

    parts = []
    for x_coord in sorted(by_x):
        y_values = sorted(set(by_x[x_coord]))
        ranges = []
        start = previous = y_values[0]
        for y_coord in y_values[1:]:
            if y_coord == previous + 1:
                previous = y_coord
                continue
            ranges.append((start, previous))
            start = previous = y_coord
        ranges.append((start, previous))
        formatted_ranges = ",".join(
            str(left) if left == right else f"{left}-{right}"
            for left, right in ranges
        )
        parts.append(f"{x_coord}:{formatted_ranges}")
    return " ".join(parts)
