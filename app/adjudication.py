"""Pure adjudication rules for radio applications.

An application grants:

* a coverage **circle** in planar coordinates ``(cx, cy, r)``;
* a **closed** frequency interval ``[freq_low, freq_high]``;
* a **half-open** UTC time interval ``[start_time, end_time)``.

Two *effective* (approved and not later revoked) applications interfere iff
ALL of the following hold:

1. the time intervals intersect;
2. the coverage circles intersect;
3. the closed frequency bands intersect, OR the minimum gap between them is
   strictly smaller than the non-negative guard band.

The module is free of framework / database code so the rules can be unit
tested directly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

# Conflict reason codes (also documented in the README).
REASON_FREQUENCY_INTERSECTION = "frequency_intersection"
REASON_WITHIN_GUARD_BAND = "within_guard_band"


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    radius: float


@dataclass(frozen=True)
class License:
    """An effective approved application, as seen by the adjudicator."""

    application_id: int
    circle: Circle
    freq_low: float
    freq_high: float
    start_time: datetime
    end_time: datetime


@dataclass(frozen=True)
class Conflict:
    other: License
    time_overlap: bool
    area_overlap: bool
    frequency_intersects: bool
    minimum_frequency_gap: float
    reason: str  # REASON_*; only set when all three trigger conditions hold


@dataclass(frozen=True)
class PairConflict:
    """An interfering pair inside a set of licenses (symmetric view).

    Used when a whole set is checked against itself, e.g. previewing a
    proposed guard band against the current effective set.
    """

    first: License
    second: License
    time_overlap: bool
    area_overlap: bool
    frequency_intersects: bool
    minimum_frequency_gap: float
    reason: str  # REASON_*


@dataclass(frozen=True)
class AvailableRange:
    """One maximal run of interference-free candidate lower bounds.

    A retuning keeps the application's original band width, coverage circle
    and UTC window fixed and only moves its lower bound ``x``.  Every such
    ``x`` inside this range keeps the retuned application clear of the
    current effective set.  The inclusivity flags say whether each endpoint
    itself is usable: closed-band touching blocks (endpoint included in the
    blocked side), while a gap exactly equal to the guard band is allowed
    (guarded boundaries therefore come back exclusive).
    """

    low: float
    high: float
    low_inclusive: bool
    high_inclusive: bool


@dataclass(frozen=True)
class _BlockedInterval:
    """Lower bounds blocked by one effective license, clipped to a window."""

    low: float
    high: float
    low_inclusive: bool
    high_inclusive: bool


def circles_intersect(a: Circle, b: Circle) -> bool:
    """Planar disks intersect iff centre distance <= sum of radii.

    Touching at one point (distance exactly equal to the sum) counts as an
    intersection.
    """
    distance = math.hypot(a.cx - b.cx, a.cy - b.cy)
    return distance <= a.radius + b.radius


def closed_intervals_intersect(
    low_a: float, high_a: float, low_b: float, high_b: float
) -> bool:
    """Closed intervals: touching endpoints intersect (a1 <= b0 and b1 <= a0)."""
    return low_a <= high_b and low_b <= high_a


def half_open_intervals_intersect(
    start_a: datetime, end_a: datetime, start_b: datetime, end_b: datetime
) -> bool:
    """[start_a, end_a) vs [start_b, end_b): end == start does NOT overlap."""
    return start_a < end_b and start_b < end_a


def minimum_frequency_gap(
    low_a: float, high_a: float, low_b: float, high_b: float
) -> float:
    """0 when the closed bands intersect, otherwise the gap between them."""
    if closed_intervals_intersect(low_a, high_a, low_b, high_b):
        return 0.0
    return low_b - high_a if high_a < low_b else low_a - high_b


def frequency_conflicts(
    low_a: float,
    high_a: float,
    low_b: float,
    high_b: float,
    guard_band: float,
) -> tuple[bool, bool, float, str | None]:
    """Return (conflicts, intersects, minimum_gap, reason).

    ``guard_band`` must be non-negative.  A gap exactly equal to the guard
    band is allowed ("strictly smaller than" triggers the conflict).
    """
    if closed_intervals_intersect(low_a, high_a, low_b, high_b):
        return True, True, 0.0, REASON_FREQUENCY_INTERSECTION
    gap = minimum_frequency_gap(low_a, high_a, low_b, high_b)
    if gap < guard_band:
        return True, False, gap, REASON_WITHIN_GUARD_BAND
    return False, False, gap, None


def _pair_trigger(
    a: License, b: License, guard_band: float
) -> tuple[bool, float, str] | None:
    """Shared interference gate for one pair of licenses.

    Returns ``(frequency_intersects, minimum_gap, reason)`` when the pair
    interferes under ``guard_band``, else ``None``.  Time and area
    intersection are necessary conditions; frequency overlap (or a
    guard-band violation) is the third.
    """
    t_overlap = half_open_intervals_intersect(
        a.start_time, a.end_time, b.start_time, b.end_time
    )
    a_overlap = circles_intersect(a.circle, b.circle)
    if not (t_overlap and a_overlap):
        return None
    freq_conflicts, freq_intersects, gap, reason = frequency_conflicts(
        a.freq_low, a.freq_high, b.freq_low, b.freq_high, guard_band
    )
    if not freq_conflicts:
        return None
    return freq_intersects, gap, (reason or REASON_FREQUENCY_INTERSECTION)


def evaluate(
    candidate: License, effective: list[License], guard_band: float
) -> list[Conflict]:
    """All interference conflicts between ``candidate`` and effective grants.

    Every checked grant is returned with its trigger flags so callers can
    preview exactly which conditions fired.
    """
    conflicts: list[Conflict] = []
    for other in effective:
        trigger = _pair_trigger(candidate, other, guard_band)
        if trigger is None:
            continue
        freq_intersects, gap, reason = trigger
        conflicts.append(
            Conflict(
                other=other,
                time_overlap=True,
                area_overlap=True,
                frequency_intersects=freq_intersects,
                minimum_frequency_gap=gap,
                reason=reason,
            )
        )
    return conflicts


def find_interfering_pairs(
    licenses: list[License], guard_band: float
) -> list[PairConflict]:
    """Every interfering pair *within* ``licenses`` under ``guard_band``.

    Used to (pre-)check a proposed guard band against the current effective
    set: the change may only be confirmed when this returns an empty list.
    Each interfering combination is reported once, in input order.
    """
    pairs: list[PairConflict] = []
    for i in range(len(licenses)):
        for j in range(i + 1, len(licenses)):
            trigger = _pair_trigger(licenses[i], licenses[j], guard_band)
            if trigger is None:
                continue
            freq_intersects, gap, reason = trigger
            pairs.append(
                PairConflict(
                    first=licenses[i],
                    second=licenses[j],
                    time_overlap=True,
                    area_overlap=True,
                    frequency_intersects=freq_intersects,
                    minimum_frequency_gap=gap,
                    reason=reason,
                )
            )
    return pairs


def evaluate_group(
    candidates: list[License], effective: list[License], guard_band: float
) -> tuple[dict[int, list[Conflict]], dict[int, list[Conflict]]]:
    """Adjudicate an ordered group as one atomic decision.

    Returns ``(against_effective, within_group)``, each keyed by the
    candidate's ``application_id`` (every candidate gets a key, possibly
    mapped to an empty list):

    * ``against_effective[id]`` - clashes with the currently effective grants,
      in effective-set order;
    * ``within_group[id]``      - clashes with other members of this group.

    Interference is symmetric, so an intra-group clash between two members is
    reported on *both* members (each listing the other as the conflict
    object).  Input order determines only the order in which a successful
    group consumes its consecutive revisions; it never hides a clash.
    """
    against_effective: dict[int, list[Conflict]] = {
        candidate.application_id: [] for candidate in candidates
    }
    within_group: dict[int, list[Conflict]] = {
        candidate.application_id: [] for candidate in candidates
    }

    for candidate in candidates:
        against_effective[candidate.application_id].extend(
            evaluate(candidate, effective, guard_band)
        )

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            # evaluate() with a singleton other returns 0 or 1 conflict;
            # run both directions so each side gets itself as the subject.
            forward = evaluate(candidates[i], [candidates[j]], guard_band)
            if not forward:
                continue
            backward = evaluate(candidates[j], [candidates[i]], guard_band)
            within_group[candidates[i].application_id].append(forward[0])
            within_group[candidates[j].application_id].append(backward[0])

    return against_effective, within_group


# --------------------------------------------------------------------------- #
# Retuning search: maximal interference-free ranges for the lower bound
# --------------------------------------------------------------------------- #
def lower_bound_blocks(
    lower_bound: float,
    width: float,
    lic: License,
    guard_band: float,
) -> bool:
    """Whether a candidate retuned to ``[lower_bound, lower_bound + width]``
    conflicts with one license ``lic``.

    Coverage circle and time window are those of the retuned application, so
    callers only need to pass licenses whose circles and windows already
    intersect them; this predicate is the frequency gate alone.  It goes
    through :func:`frequency_conflicts`, the exact rule a single-application
    preview uses at any concrete lower bound.
    """
    conflicts, _, _, _ = frequency_conflicts(
        lower_bound, lower_bound + width, lic.freq_low, lic.freq_high, guard_band
    )
    return conflicts


def _raw_blocked_interval(
    lic: License, width: float, guard_band: float
) -> tuple[float, float, bool, bool]:
    """The set of lower bounds blocked by ``lic`` (before window clipping).

    With the retuned band ``[x, x + width]`` against license band
    ``[lic.freq_low, lic.freq_high]``:

    * it intersects the closed band iff
      ``lic.freq_low - width <= x <= lic.freq_high`` (touching included);
    * otherwise a strict guard-band violation happens exactly on
      ``lic.freq_low - width - guard_band < x < lic.freq_high + guard_band``.

    Together the blocked set is the open interval
    ``(lic.freq_low - width - guard_band, lic.freq_high + guard_band)``; the
    endpoint flags are asked from :func:`lower_bound_blocks` so the result
    agrees exactly with a single-application preview at the boundary values
    (in particular both ends close when the guard band is zero).
    """
    raw_low = lic.freq_low - width - guard_band
    raw_high = lic.freq_high + guard_band
    low_inclusive = lower_bound_blocks(raw_low, width, lic, guard_band)
    high_inclusive = lower_bound_blocks(raw_high, width, lic, guard_band)
    return raw_low, raw_high, low_inclusive, high_inclusive


def _clip_blocked(
    raw_low: float,
    raw_high: float,
    raw_low_inclusive: bool,
    raw_high_inclusive: bool,
    window_low: float,
    window_high: float,
) -> _BlockedInterval | None:
    """Clip one blocked interval into the candidate window.

    Endpoints strictly inside the window after clipping are blocked; an
    endpoint that lands exactly on a window boundary inherits the original
    inclusivity (boundary points themselves are re-checked later).  A
    degenerate point that is not blocked is dropped.
    """
    if raw_high < window_low or raw_low > window_high:
        return None
    if raw_low < window_low:
        low = window_low
        low_inclusive = raw_low_inclusive if raw_low == window_low else True
    else:
        low = raw_low
        low_inclusive = raw_low_inclusive
    if raw_high > window_high:
        high = window_high
        high_inclusive = raw_high_inclusive if raw_high == window_high else True
    else:
        high = raw_high
        high_inclusive = raw_high_inclusive
    if low == high and not (low_inclusive and high_inclusive):
        return None
    return _BlockedInterval(low, high, low_inclusive, high_inclusive)


def _merge_blocked(
    intervals: list[_BlockedInterval],
) -> list[_BlockedInterval]:
    """Merge overlapping blocked intervals, keeping blocked touching points.

    Two intervals merge when they overlap in the real line.  Intervals that
    only meet at one point merge as well whenever that point is blocked on
    either side; two open ends merely touching stay separate so the free
    singleton between them survives.
    """
    if not intervals:
        return []
    ordered = sorted(
        intervals,
        key=lambda iv: (iv.low, iv.high, not iv.low_inclusive),
    )
    merged: list[_BlockedInterval] = [ordered[0]]
    for iv in ordered[1:]:
        last = merged[-1]
        if iv.low > last.high:
            merged.append(iv)
            continue
        if iv.low == last.high and not (iv.low_inclusive or last.high_inclusive):
            # Two open ends touch at a free point: the union is not connected
            # at that point, so keep the intervals separate.
            merged.append(iv)
            continue
        high = last.high
        high_inclusive = last.high_inclusive
        if iv.high > high:
            high = iv.high
            high_inclusive = iv.high_inclusive
        elif iv.high == high:
            high_inclusive = last.high_inclusive or iv.high_inclusive
        merged[-1] = _BlockedInterval(
            low=last.low,
            high=high,
            low_inclusive=last.low_inclusive,
            high_inclusive=high_inclusive,
        )
    return merged


def available_lower_bands(
    candidate: License,
    width: float,
    window_low: float,
    window_high: float,
    effective: list[License],
    guard_band: float,
) -> list[AvailableRange]:
    """Maximal interference-free ranges for the candidate's lower bound.

    The candidate keeps its coverage circle, UTC window and band *width*
    fixed; its closed band becomes ``[x, x + width]`` (``width == 0`` is the
    single-frequency case).  Search only over ``window_low <= x <=
    window_high``; the window is a closed interval.

    Only effective licenses whose circles AND half-open windows intersect
    the candidate can block it - the same gates as :func:`evaluate`.  For
    every such license the blocked lower bounds form one interval; the
    answer is the merged complement inside the search window.  Endpoint
    inclusivity is decided with :func:`lower_bound_blocks`, i.e. by the same
    frequency rule a single-application preview applies at that exact lower
    bound, so both conclusions always agree.
    """
    if window_low > window_high:
        raise ValueError("window_low must not exceed window_high")
    if width < 0 or guard_band < 0:
        raise ValueError("width and guard_band must be non-negative")

    relevant = [
        lic
        for lic in effective
        if half_open_intervals_intersect(
            candidate.start_time,
            candidate.end_time,
            lic.start_time,
            lic.end_time,
        )
        and circles_intersect(candidate.circle, lic.circle)
    ]

    clipped: list[_BlockedInterval] = []
    for lic in relevant:
        raw_low, raw_high, low_inc, high_inc = _raw_blocked_interval(
            lic, width, guard_band
        )
        iv = _clip_blocked(
            raw_low,
            raw_high,
            low_inc,
            high_inc,
            window_low,
            window_high,
        )
        if iv is not None:
            clipped.append(iv)

    blocked = _merge_blocked(clipped)

    def free_at(value: float) -> bool:
        return not any(
            lower_bound_blocks(value, width, lic, guard_band) for lic in relevant
        )

    # Walk the (window-clipped, sorted, merged) blocked intervals and emit
    # the raw complementary gaps between them.
    raw_gaps: list[list[float]] = []
    cursor = window_low
    for iv in blocked:
        if cursor <= iv.low:
            raw_gaps.append([cursor, iv.low])
        cursor = iv.high
    if cursor <= window_high:
        raw_gaps.append([cursor, window_high])

    # Neighbouring raw gaps can share one endpoint where two blocked
    # intervals merely touch.  They are parts of one continuous free run
    # exactly when the shared point itself is free (e.g. two guard-band
    # boundaries one band width apart leave a gap equal to the guard band on
    # both sides); an inclusive, blocked touching point must keep the gaps
    # separate.  The decision uses the same frequency rule a single preview
    # applies at that point, so the merged ranges stay maximal and agree with
    # the preview at every boundary.
    merged_gaps: list[list[float]] = []
    for gap in raw_gaps:
        low, high = gap
        if merged_gaps:
            prev = merged_gaps[-1]
            if prev[1] == low and free_at(low):
                prev[1] = high
                continue
        merged_gaps.append([low, high])

    ranges: list[AvailableRange] = []
    for low, high in merged_gaps:
        if low == high:
            if free_at(low):
                ranges.append(AvailableRange(low, high, True, True))
            continue
        ranges.append(
            AvailableRange(low, high, free_at(low), free_at(high))
        )
    return ranges
