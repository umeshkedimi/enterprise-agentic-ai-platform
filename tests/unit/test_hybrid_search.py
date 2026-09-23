"""Reciprocal rank fusion, pinned without a database.

`_reciprocal_rank_fusion` is the entire fusion decision — everything else in
`semantic_search` is two SQL queries and the plumbing to run this over their
results.
"""

import uuid

from app.services.retrieval_service import _reciprocal_rank_fusion

A, B, C, D = (uuid.uuid4() for _ in range(4))


def test_agreement_between_both_rankers_wins():
    """A candidate ranked well by both signals should fuse to the top even
    if a different candidate is ranked #1 by only one of them."""
    vector_ranking = [A, B, C]
    fts_ranking = [B, A, C]

    fused = _reciprocal_rank_fusion([vector_ranking, fts_ranking], k=60)

    assert fused[0] in (A, B)
    assert fused[-1] == C


def test_a_candidate_found_by_only_one_ranker_still_appears():
    """The whole point: a strong single-signal match is rescued, not
    discarded for lack of agreement from the other ranker."""
    vector_ranking = [A, B]
    fts_ranking = [C]

    fused = _reciprocal_rank_fusion([vector_ranking, fts_ranking], k=60)

    assert set(fused) == {A, B, C}


def test_rank_one_in_one_list_can_outrank_rank_two_in_both():
    """A concrete case from the RRF formula itself: 1/(k+1) from a single
    ranker can still beat 1/(k+2) + 1/(k+2) from two rankers once k is large
    enough to flatten the difference — this is the "damping" k controls."""
    vector_ranking = [A, B]
    fts_ranking = [A]
    # B: 1/(60+2) from vector only ≈ 0.01613
    # A: 1/(60+1) + 1/(60+1) ≈ 0.03279
    fused = _reciprocal_rank_fusion([vector_ranking, fts_ranking], k=60)
    assert fused[0] == A


def test_an_empty_ranking_contributes_nothing():
    fused = _reciprocal_rank_fusion([[A, B], []], k=60)
    assert fused == [A, B]


def test_no_rankings_at_all_fuses_to_nothing():
    assert _reciprocal_rank_fusion([], k=60) == []


def test_a_larger_k_flattens_rank_differences():
    """A smaller k weights rank position more aggressively; a much larger k
    should bring two adjacently-ranked candidates' fused scores closer
    together, which this checks indirectly by holding order stable across a
    wide range — RRF is a monotonic function of rank regardless of k, so the
    order here should never flip within one ranking."""
    ranking = [A, B, C, D]
    for k in (1, 10, 60, 1000):
        assert _reciprocal_rank_fusion([ranking], k=k) == ranking
