"""Deterministic numeric-claim verification.

A live run in the evaluation chunk showed a judge crediting an answer's own
arithmetic to a source that never stated the result: the source said
"pro-rata", the answer computed "25 × 3/5 = 15 days", and the v1 judge
returned the claim as supported. The fix that actually held was the *other*
half of that incident — enforcing "supported means named a source" in code,
because a model checking its own arithmetic is the same unreliable
self-report the platform already refuses to trust for a groundedness score.

This module is a second, independent instance of that same principle. A
model deciding whether a sentence is a genuine claim, versus a meta-statement
about what the sources lack, is a judgement call with no ground truth to
check it against — still an open problem, see the "known limitation" note
this module's caller carries forward. Whether a specific *number* in a claim
appears in the text it cites is not a judgement call. It is exactly checkable,
which is what makes it worth a deterministic pass instead of one more
sentence added to a rubric a model may or may not follow.

Only ever narrows what counts as supported, never widens it: a claim with no
numbers at all is untouched, and a number this module can't account for
(nothing below handles thousands separators combined with decimals, or
anything above ninety-nine spelled out) fails closed — the claim is left
exactly as the judge scored it rather than guessed at either way.
"""

import re
from collections.abc import Sequence

# Citation markers ("[1]", "[1, 2]") are source references, not facts about
# the world — stripped before scanning a claim for numbers so a claim's own
# citation is never mistaken for a number that needs grounding.
_CITATION_MARKERS = re.compile(r"\[[\d,\s]+\]")
# The decimal group requires a digit *after* the dot, deliberately — "\.?\d*"
# would also match the bare "." that ends a sentence ("...150.") as part of
# the number, capturing "150." instead of "150" and failing every comparison
# against a source that (correctly) has no trailing period in its own digits.
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?%?")

_ONES_BY_VALUE = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}
_TENS_BY_VALUE = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}


def _numbers_in(text: str) -> set[str]:
    """Every number-looking token in `text`, normalised — thousands commas
    and a trailing `%` stripped — with citation markers removed first."""
    stripped = _CITATION_MARKERS.sub(" ", text)
    return {match.replace(",", "").rstrip("%") for match in _NUMBER.findall(stripped)}


def _spelled_out(number: str) -> str | None:
    """The English words for a small whole number (`"25"` -> `"twenty-five"`),
    or `None` for anything this doesn't cover — a decimal, anything at or
    above one hundred. Deliberately not a general number-words parser: the
    only job here is stopping a source that spells a number out ("twenty-five
    days") from looking, to a purely digit-based check, like it never
    mentioned the number at all.
    """
    if not number.isdigit():
        return None
    value = int(number)
    if value in _ONES_BY_VALUE:
        return _ONES_BY_VALUE[value]
    if value in _TENS_BY_VALUE:
        return _TENS_BY_VALUE[value]
    if value < 100:
        tens_word = _TENS_BY_VALUE.get(value - value % 10)
        ones_word = _ONES_BY_VALUE.get(value % 10)
        if tens_word and ones_word:
            return f"{tens_word}-{ones_word}"
    return None


def _appears_in(number: str, source_text: str) -> bool:
    if number in _numbers_in(source_text):
        return True
    words = _spelled_out(number)
    if words is None:
        return False
    # Sources write compounds both hyphenated and spaced ("twenty-five" and
    # "twenty five" both occur in real text) — normalised to spaces on both
    # sides so either form matches.
    normalized_source = source_text.lower().replace("-", " ")
    return words.replace("-", " ") in normalized_source


def unverified_numbers(claim_text: str, source_texts: Sequence[str]) -> list[str]:
    """Numbers in `claim_text` that appear in none of `source_texts` — as a
    digit string or, failing that, spelled out in English.

    Empty when the claim has no numbers to check at all, which is the common
    case: most claims are not quantities, and this has nothing to say about
    those.
    """
    candidates = _numbers_in(claim_text)
    if not candidates:
        return []
    return sorted(
        number
        for number in candidates
        if not any(_appears_in(number, text) for text in source_texts)
    )
