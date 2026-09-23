"""Number extraction and source matching, pinned in isolation.

`evaluation_service._apply_numeric_verification` (tested in
`test_evaluation.py`, alongside the rest of the claim pipeline it's part of)
covers the integration into a claim's `supported` flag. These pin the actual
arithmetic underneath it — what counts as a number, and what counts as
finding one — without needing a claim or a judge in the picture at all.
"""

from app.services.numeric_verification import unverified_numbers


def test_a_number_present_verbatim_in_a_source_is_verified():
    assert unverified_numbers("Staff get 25 days.", ["Full-time staff get 25 days."]) == []


def test_a_number_absent_from_every_source_is_unverified():
    assert unverified_numbers("Staff get 25 days.", ["Some unrelated text."]) == ["25"]


def test_a_number_only_the_answer_computed_is_the_incident_this_module_fixes():
    """The exact shape of the Chunk 7 bug: a source stating a rule, not a
    result, and an answer that did the arithmetic itself."""
    assert unverified_numbers(
        "Part-time staff get 15 days, pro-rated.",
        ["Part-time staff receive a pro-rata share of full-time leave."],
    ) == ["15"]


def test_a_spelled_out_number_in_the_source_still_verifies_a_digit_claim():
    assert (
        unverified_numbers("Staff get 25 days.", ["Full-time staff accrue twenty-five days."])
        == []
    )


def test_a_spelled_out_number_with_a_space_instead_of_a_hyphen_still_verifies():
    assert (
        unverified_numbers("Staff get 25 days.", ["Full-time staff accrue twenty five days."])
        == []
    )


def test_round_tens_spell_out_without_a_compound():
    assert unverified_numbers("30 days.", ["Employees accrue thirty days."]) == []


def test_teens_spell_out_without_a_compound():
    assert unverified_numbers("15 days.", ["Employees accrue fifteen days."]) == []


def test_a_claim_with_no_numbers_has_nothing_to_verify():
    assert unverified_numbers("Employees may request leave in advance.", ["anything"]) == []


def test_citation_markers_are_not_treated_as_numbers():
    assert unverified_numbers("Staff get leave [1].", ["no numbers here"]) == []
    assert unverified_numbers("Staff get leave [1, 2].", ["no numbers here"]) == []


def test_thousands_separators_are_normalised_on_both_sides():
    assert unverified_numbers("Revenue was $1,234.", ["Reported revenue: 1234 dollars."]) == []


def test_a_percentage_matches_the_bare_number_in_the_source():
    assert unverified_numbers("Uptime was 99%.", ["Measured uptime: 99 percent."]) == []


def test_a_number_verified_by_any_one_of_several_sources_counts():
    result = unverified_numbers(
        "Staff get 25 days.",
        ["Unrelated text.", "Full-time staff accrue 25 days."],
    )
    assert result == []


def test_a_number_above_ninety_nine_is_not_spelled_out_and_fails_closed():
    """Deliberately outside what `_spelled_out` covers — a number this large
    with no digit match anywhere is reported unverified rather than guessed
    at, which is the correct, conservative outcome either way."""
    assert unverified_numbers("Revenue was 150.", ["No numbers in this source."]) == ["150"]


def test_multiple_unverified_numbers_are_all_reported_sorted():
    assert unverified_numbers("15 days and 40 dollars.", ["no numbers here"]) == ["15", "40"]
