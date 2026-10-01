from datetime import datetime, timedelta, timezone

from app.adjudication import (
    AvailableRange,
    REASON_FREQUENCY_INTERSECTION,
    REASON_WITHIN_GUARD_BAND,
    Circle,
    License,
    circles_intersect,
    closed_intervals_intersect,
    evaluate,
    evaluate_group,
    available_lower_bands,
    find_interfering_pairs,
    frequency_conflicts,
    half_open_intervals_intersect,
    minimum_frequency_gap,
)

T0 = datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc)
T1 = T0 + timedelta(hours=2)
T2 = T1
T3 = T0 + timedelta(hours=4)


def lic(
    app_id,
    cx=0.0,
    cy=0.0,
    r=10.0,
    flo=100.0,
    fhi=110.0,
    start=T0,
    end=T1,
):
    return License(
        application_id=app_id,
        circle=Circle(cx, cy, r),
        freq_low=flo,
        freq_high=fhi,
        start_time=start,
        end_time=end,
    )


class TestCircles:
    def test_nested_circles(self):
        assert circles_intersect(Circle(0, 0, 10), Circle(1, 1, 2))

    def test_touching_at_one_point_counts(self):
        # centre distance exactly 10 = radii sum 5+5
        assert circles_intersect(Circle(0, 0, 5), Circle(10, 0, 5))

    def test_disjoint(self):
        assert not circles_intersect(Circle(0, 0, 5), Circle(20, 0, 5))

    def test_two_dimensional_distance(self):
        # 3-4-5 triangle, radii 2+2=4 < 5 -> disjoint
        assert not circles_intersect(Circle(0, 0, 2), Circle(3, 4, 2))


class TestIntervals:
    def test_closed_endpoints_touching_intersect(self):
        assert closed_intervals_intersect(100, 110, 110, 120)
        assert closed_intervals_intersect(110, 120, 100, 110)

    def test_closed_disjoint(self):
        assert not closed_intervals_intersect(100, 109, 110, 120)

    def test_half_open_touching_does_not_overlap(self):
        assert not half_open_intervals_intersect(T0, T1, T2, T3)
        assert half_open_intervals_intersect(T0, T3, T1, T3 + timedelta(hours=1))

    def test_half_open_partial(self):
        assert half_open_intervals_intersect(
            T0, T3, T0 + timedelta(hours=1), T3 + timedelta(hours=1)
        )


class TestFrequency:
    def test_overlapping_bands(self):
        conflicts, intersects, gap, reason = frequency_conflicts(
            100, 110, 105, 115, guard_band=2.0
        )
        assert conflicts and intersects and gap == 0
        assert reason == REASON_FREQUENCY_INTERSECTION

    def test_endpoint_touch_intersects_closed_band(self):
        conflicts, intersects, gap, _ = frequency_conflicts(
            100, 110, 110, 120, guard_band=2.0
        )
        assert conflicts and intersects and gap == 0

    def test_gap_smaller_than_guard(self):
        # gap 1.5 < 2.0
        conflicts, intersects, gap, reason = frequency_conflicts(
            100, 110, 111.5, 120, guard_band=2.0
        )
        assert conflicts and not intersects and gap == 1.5
        assert reason == REASON_WITHIN_GUARD_BAND

    def test_gap_equal_to_guard_is_allowed(self):
        # Strictly smaller: gap == guard band must NOT conflict.
        conflicts, intersects, gap, reason = frequency_conflicts(
            100, 110, 112, 120, guard_band=2.0
        )
        assert not conflicts and not intersects and gap == 2.0 and reason is None

    def test_gap_larger_than_guard(self):
        conflicts, *_ = frequency_conflicts(100, 110, 120, 130, guard_band=2.0)
        assert not conflicts

    def test_zero_guard_band_disjoint_is_ok(self):
        assert minimum_frequency_gap(100, 110, 112, 120) == 2.0
        conflicts, *_ = frequency_conflicts(100, 110, 110.0001, 120, 0.0)
        assert not conflicts


class TestEvaluate:
    def test_requires_time_and_area_and_frequency(self):
        # same place/freq but adjacent (non-overlapping) times -> no conflict
        later = lic(2, start=T1, end=T3)
        assert evaluate(lic(1, end=T1), [later], guard_band=5.0) == []

        # same time/freq but disjoint areas -> no conflict
        far = lic(2, cx=100.0, cy=0.0, r=1.0)
        assert evaluate(lic(1), [far], guard_band=5.0) == []

        # same time/place, separated frequencies beyond guard -> no conflict
        separated = lic(2, flo=200, fhi=210)
        assert evaluate(lic(1), [separated], guard_band=2.0) == []

    def test_full_conflict(self):
        conflicts = evaluate(lic(1), [lic(2, flo=105, fhi=115)], 2.0)
        assert len(conflicts) == 1
        assert conflicts[0].other.application_id == 2
        assert conflicts[0].reason == REASON_FREQUENCY_INTERSECTION

    def test_guard_band_conflict_through_all_gates(self):
        other = lic(2, flo=111, fhi=112)  # gap 1 < guard 2
        (conflict,) = evaluate(lic(1), [other], 2.0)
        assert conflict.reason == REASON_WITHIN_GUARD_BAND
        assert conflict.minimum_frequency_gap == 1.0
        assert conflict.time_overlap and conflict.area_overlap
        assert not conflict.frequency_intersects


class TestEvaluateGroup:
    def test_clean_group_has_no_conflicts_anywhere(self):
        # Disjoint frequencies, well beyond the 2 MHz guard band.
        members = [lic(1, flo=100, fhi=101), lic(2, flo=200, fhi=201)]
        against, within = evaluate_group(members, [lic(9, flo=300, fhi=301)], 2.0)
        assert against == {1: [], 2: []}
        assert within == {1: [], 2: []}

    def test_each_member_checked_against_effective_set(self):
        members = [lic(1, flo=100, fhi=110), lic(2, flo=200, fhi=210)]
        against, within = evaluate_group(members, [lic(9, flo=105, fhi=106)], 2.0)
        assert len(against[1]) == 1
        assert against[1][0].other.application_id == 9
        assert within == {1: [], 2: []}

    def test_intra_group_clash_reported_on_both_sides(self):
        # Two members overlap each other in band but not the effective set.
        members = [lic(1, flo=100, fhi=110), lic(2, flo=105, fhi=115)]
        against, within = evaluate_group(members, [lic(9, flo=300, fhi=301)], 2.0)
        assert against == {1: [], 2: []}
        assert within[1][0].other.application_id == 2
        assert within[2][0].other.application_id == 1
        assert within[1][0].reason == REASON_FREQUENCY_INTERSECTION

    def test_ordering_does_not_hide_intra_group_clash(self):
        # The clash is between members at positions 1 and 3; every candidate
        # still gets a key, and the middle member is clean.
        members = [
            lic(1, flo=100, fhi=110),
            lic(2, flo=200, fhi=210),
            lic(3, flo=105, fhi=106),
        ]
        against, within = evaluate_group(members, [], 2.0)
        assert set(against) == {1, 2, 3}
        assert within[1][0].other.application_id == 3
        assert within[3][0].other.application_id == 1
        assert within[2] == []

    def test_guard_band_applies_within_group(self):
        # gap 1 < guard 2 between two otherwise disjoint members
        members = [lic(1, flo=100, fhi=110), lic(2, flo=111, fhi=112)]
        _, within = evaluate_group(members, [], 2.0)
        assert within[1][0].reason == REASON_WITHIN_GUARD_BAND
        assert within[1][0].minimum_frequency_gap == 1.0


class TestFindInterferingPairs:
    def test_no_pairs_when_clean(self):
        members = [lic(1, flo=100, fhi=110), lic(2, flo=200, fhi=210)]
        assert find_interfering_pairs(members, 2.0) == []

    def test_pair_detected_with_trigger_details(self):
        a, b = lic(1, flo=100, fhi=110), lic(2, flo=111, fhi=112)
        (pair,) = find_interfering_pairs([a, b], 2.0)
        assert pair.first is a and pair.second is b
        assert pair.reason == REASON_WITHIN_GUARD_BAND
        assert pair.minimum_frequency_gap == 1.0
        assert pair.time_overlap and pair.area_overlap
        assert not pair.frequency_intersects

    def test_frequency_intersection_pair(self):
        pairs = find_interfering_pairs(
            [lic(1, flo=100, fhi=110), lic(2, flo=105, fhi=115)], 2.0
        )
        assert len(pairs) == 1
        assert pairs[0].reason == REASON_FREQUENCY_INTERSECTION
        assert pairs[0].frequency_intersects

    def test_pair_requires_time_and_area(self):
        # frequencies overlap but areas are disjoint -> no pair
        assert find_interfering_pairs(
            [lic(1), lic(2, cx=100.0, r=1.0)], 5.0
        ) == []
        # half-open time windows merely touching -> no pair
        assert find_interfering_pairs(
            [lic(1, end=T1), lic(2, start=T1, end=T3)], 5.0
        ) == []

    def test_every_interfering_combination_reported_once_in_order(self):
        members = [
            lic(1, flo=100, fhi=110),
            lic(2, flo=105, fhi=115),
            lic(3, flo=300, fhi=301),
            lic(4, flo=108, fhi=109),
        ]
        pairs = find_interfering_pairs(members, 2.0)
        assert [(p.first.application_id, p.second.application_id) for p in pairs] == [
            (1, 2),
            (1, 4),
            (2, 4),
        ]

    def test_gap_equal_to_proposed_band_is_allowed(self):
        # Strictly-smaller rule: gap 2 == proposed band 2 -> no pair.
        members = [lic(1, flo=100, fhi=110), lic(2, flo=112, fhi=120)]
        assert find_interfering_pairs(members, 2.0) == []
        (pair,) = find_interfering_pairs(members, 2.5)
        assert pair.minimum_frequency_gap == 2.0


def _cand(width, app_id=0, cx=0.0, cy=0.0, r=10.0, start=T0, end=T1):
    return lic(
        app_id,
        cx=cx,
        cy=cy,
        r=r,
        flo=0.0,
        fhi=width,
        start=start,
        end=end,
    )


class TestAvailableLowerBands:
    def test_single_license_guarded_blocked_interval(self):
        # W=10, G=2, license [100,110] -> lower bounds blocked on (88, 112)
        ranges = available_lower_bands(
            _cand(10.0), 10.0, 0.0, 200.0, [lic(1, flo=100.0, fhi=110.0)], 2.0
        )
        assert ranges == [
            AvailableRange(0.0, 88.0, True, True),
            AvailableRange(112.0, 200.0, True, True),
        ]

    def test_zero_guard_band_closes_the_blocked_interval(self):
        # G=0: [90, 110] blocked; touching endpoints 90 and 110 conflict
        ranges = available_lower_bands(
            _cand(10.0), 10.0, 0.0, 200.0, [lic(1, flo=100.0, fhi=110.0)], 0.0
        )
        assert ranges == [
            AvailableRange(0.0, 90.0, True, False),
            AvailableRange(110.0, 200.0, False, True),
        ]

    def test_frequency_touching_blocks_endpoint_even_with_guard(self):
        # x == license high touches the closed band -> blocked (inclusive end
        # of the intersection region); only the guarded outer side is open.
        ranges = available_lower_bands(
            _cand(0.0), 0.0, 100.0, 120.0, [lic(1, flo=100.0, fhi=110.0)], 2.0
        )
        assert ranges == [AvailableRange(112.0, 120.0, True, True)]

    def test_single_frequency_application_point_blocked_with_zero_guard(self):
        # W=0, G=0, point license 100: only x == 100 is blocked
        ranges = available_lower_bands(
            _cand(0.0), 0.0, 0.0, 200.0, [lic(1, flo=100.0, fhi=100.0)], 0.0
        )
        assert ranges == [
            AvailableRange(0.0, 100.0, True, False),
            AvailableRange(100.0, 200.0, False, True),
        ]

    def test_gap_exactly_equal_to_guard_leaves_a_free_singleton(self):
        # Two licenses [100,110] and [114,120], W=0, G=2:
        # blocked (98,112) and (112,122); x=112 is a free singleton because
        # the gap to either side equals the guard band exactly.
        ranges = available_lower_bands(
            _cand(0.0),
            0.0,
            0.0,
            200.0,
            [lic(1, flo=100.0, fhi=110.0), lic(2, flo=114.0, fhi=120.0)],
            2.0,
        )
        assert ranges == [
            AvailableRange(0.0, 98.0, True, True),
            AvailableRange(112.0, 112.0, True, True),
            AvailableRange(122.0, 200.0, True, True),
        ]

    def test_open_blockers_meeting_at_a_free_point_do_not_shrink_ranges(self):
        # W=10, G=2: license [100,110] blocks (88,112) and license
        # [124,130] blocks (112,132); the open blocker ends meet at the free
        # point x=112.  The free set is [0,88] U {112} U [132,250], so the
        # singleton must survive without merging the neighbouring ranges.
        ranges = available_lower_bands(
            _cand(10.0),
            10.0,
            0.0,
            250.0,
            [lic(1, flo=100.0, fhi=110.0), lic(2, flo=124.0, fhi=130.0)],
            2.0,
        )
        assert ranges == [
            AvailableRange(0.0, 88.0, True, True),
            AvailableRange(112.0, 112.0, True, True),
            AvailableRange(132.0, 250.0, True, True),
        ]

    def test_blocked_touching_point_keeps_gaps_separate(self):
        # Under guard band 0: license [100,110] blocks [90,110] and license
        # [120,200] blocks [110,200]; both include 110 (closed-band touching),
        # so the blockers merge over [90,200] and 110 never joins the gaps.
        ranges = available_lower_bands(
            _cand(10.0),
            10.0,
            0.0,
            250.0,
            [lic(1, flo=100.0, fhi=110.0), lic(2, flo=120.0, fhi=200.0)],
            0.0,
        )
        assert ranges == [
            AvailableRange(0.0, 90.0, True, False),
            AvailableRange(200.0, 250.0, False, True),
        ]

    def test_overlapping_blocked_intervals_merge(self):
        # (88,112) and (108,132) overlap -> one blocked interval (88,132)
        ranges = available_lower_bands(
            _cand(10.0),
            10.0,
            0.0,
            200.0,
            [lic(1, flo=100.0, fhi=110.0), lic(2, flo=120.0, fhi=130.0)],
            2.0,
        )
        assert ranges == [
            AvailableRange(0.0, 88.0, True, True),
            AvailableRange(132.0, 200.0, True, True),
        ]

    def test_window_fully_blocked_returns_empty(self):
        ranges = available_lower_bands(
            _cand(10.0), 10.0, 90.0, 100.0, [lic(1, flo=100.0, fhi=110.0)], 2.0
        )
        assert ranges == []

    def test_window_at_open_blocker_boundary_is_free(self):
        # raw blocked (88,112): window starting exactly at 112 is entirely free
        ranges = available_lower_bands(
            _cand(10.0), 10.0, 112.0, 120.0, [lic(1, flo=100.0, fhi=110.0)], 2.0
        )
        assert ranges == [AvailableRange(112.0, 120.0, True, True)]

    def test_singleton_window_blocked_and_free(self):
        assert (
            available_lower_bands(
                _cand(10.0), 10.0, 100.0, 100.0, [lic(1, flo=100.0, fhi=110.0)], 2.0
            )
            == []
        )
        assert available_lower_bands(
            _cand(10.0), 10.0, 88.0, 88.0, [lic(1, flo=100.0, fhi=110.0)], 2.0
        ) == [AvailableRange(88.0, 88.0, True, True)]

    def test_time_and_area_gates_apply(self):
        later = lic(1, flo=100.0, fhi=110.0, start=T1, end=T3)
        far = lic(2, flo=100.0, fhi=110.0, cx=1000.0, r=1.0)
        assert available_lower_bands(
            _cand(10.0, end=T1), 10.0, 0.0, 200.0, [later], 2.0
        ) == [AvailableRange(0.0, 200.0, True, True)]
        assert available_lower_bands(
            _cand(10.0), 10.0, 0.0, 200.0, [far], 2.0
        ) == [AvailableRange(0.0, 200.0, True, True)]

    def test_invalid_arguments_rejected(self):
        import pytest

        with pytest.raises(ValueError):
            available_lower_bands(
                _cand(10.0), 10.0, 200.0, 0.0, [], 2.0
            )
        with pytest.raises(ValueError):
            available_lower_bands(_cand(10.0), -1.0, 0.0, 1.0, [], 2.0)

    def test_endpoint_inclusivity_matches_evaluate_everywhere(self):
        # For every reported endpoint the inclusive flag must equal "no
        # conflict" from the single-application rule, and interiors are free.
        licenses = [lic(1, flo=100.0, fhi=110.0), lic(2, flo=114.0, fhi=120.0)]
        width, guard = 3.0, 2.0
        ranges = available_lower_bands(
            _cand(width), width, 0.0, 200.0, licenses, guard
        )

        def blocked(x):
            moved = lic(50, flo=x, fhi=x + width)
            return bool(evaluate(moved, licenses, guard))

        for rng in ranges:
            assert blocked(rng.low) == (not rng.low_inclusive)
            assert blocked(rng.high) == (not rng.high_inclusive)
            if rng.low < rng.high:
                assert not blocked((rng.low + rng.high) / 2.0)
