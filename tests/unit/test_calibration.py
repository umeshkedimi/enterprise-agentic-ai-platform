"""The floor recommendation, tested as the judgement call it is.

`recommend_floor` is pure and the interesting cases are all about restraint —
when it declines to recommend anything, and what it says instead. A calibration
report that produces a confident threshold from thin data would be worse than
the deferred decision it replaced, so most of what follows checks that it
doesn't.
"""

from app.services.calibration_service import (
    BUCKET_EDGES,
    CalibrationBucket,
    _bounds,
    recommend_floor,
)


def bucket(index: int, *, evaluations: int, mean: float | None, graded: int | None = None):
    lower, upper = _bounds(index)
    graded = evaluations if graded is None else graded
    return CalibrationBucket(
        lower=lower,
        upper=upper,
        evaluations=evaluations,
        graded=graded,
        mean_score=mean,
        abstentions=evaluations - graded,
    )


def full_range(mean_by_index: dict[int, float], *, per_bucket: int = 40):
    """A complete bucket list, defaulting to well-grounded outside the named ones."""
    return [
        bucket(i, evaluations=per_bucket, mean=mean_by_index.get(i, 0.95))
        for i in range(len(BUCKET_EDGES) + 1)
    ]


def test_no_floor_is_proposed_from_thin_data():
    """The whole reason the decision was deferred in the first place."""
    result = recommend_floor([bucket(0, evaluations=5, mean=0.1)])

    assert result.floor is None
    assert "at least" in result.rationale
    assert result.turns_that_would_abstain == 0


def test_a_floor_is_the_top_of_the_losing_run():
    # Buckets 0 and 1 (below 0.3) produce badly grounded answers; everything
    # above clears the target.
    buckets = full_range({0: 0.2, 1: 0.4})
    result = recommend_floor(buckets)

    assert result.floor == BUCKET_EDGES[1]
    assert result.floor == 0.3


def test_the_recommendation_reports_what_the_floor_would_cost():
    """The number a threshold picked off a chart never comes with.

    Turns below the floor do not become better answers, they become abstentions
    — the floor trades one failure for another, and an operator can only weigh
    that with the count in front of them.
    """
    buckets = full_range({0: 0.2, 1: 0.4}, per_bucket=40)
    result = recommend_floor(buckets)

    assert result.turns_that_would_abstain == 80


def test_no_floor_when_even_weak_retrievals_answer_well():
    result = recommend_floor(full_range({}))

    assert result.floor is None
    assert "doing its job" in result.rationale


def test_a_thinly_sampled_bucket_stops_the_walk_rather_than_counting_as_bad():
    """Silence is not evidence of failure.

    Bucket 1 holds three graded turns. Treating that as a losing band would push
    the floor up on the strength of three rows; stopping keeps the floor where
    the data actually supports it.
    """
    buckets = full_range({0: 0.1, 1: 0.1})
    buckets[1] = bucket(1, evaluations=3, mean=0.1)

    result = recommend_floor(buckets)

    assert result.floor == BUCKET_EDGES[0]
    assert result.floor == 0.2


def test_a_bucket_of_pure_abstentions_stops_the_walk():
    """No graded turns means no mean score, which means nothing to conclude."""
    buckets = full_range({0: 0.1})
    buckets[1] = bucket(1, evaluations=50, mean=None, graded=0)
    buckets[2] = bucket(2, evaluations=50, mean=0.1)

    result = recommend_floor(buckets)

    # Stops at the abstention band rather than reaching over it to bucket 2.
    assert result.floor == BUCKET_EDGES[0]


def test_the_target_is_a_parameter_not_a_constant():
    """A stricter target moves the floor up; the rule does not change."""
    buckets = full_range({0: 0.2, 1: 0.5, 2: 0.85})

    assert recommend_floor(buckets, target=0.8).floor == 0.3
    assert recommend_floor(buckets, target=0.9).floor == 0.4


def test_bucket_bounds_are_contiguous_and_open_at_the_top():
    lowest = _bounds(0)
    highest = _bounds(len(BUCKET_EDGES))

    assert lowest == (0.0, BUCKET_EDGES[0])
    assert highest == (BUCKET_EDGES[-1], None)
    for i in range(1, len(BUCKET_EDGES) + 1):
        assert _bounds(i)[0] == _bounds(i - 1)[1]
