from app.core.security import (
    API_KEY_PREFIX,
    KEY_PREFIX_STORED_LEN,
    generate_api_key,
    hash_api_key,
    key_prefix,
)


def test_generated_key_has_prefix_and_is_unique():
    a = generate_api_key()
    b = generate_api_key()
    assert a.startswith(API_KEY_PREFIX)
    # Two mints must never collide — the whole auth model rests on this.
    assert a != b


def test_hash_is_deterministic_and_hex_64():
    key = generate_api_key()
    digest = hash_api_key(key)
    # Deterministic so a presented key hashes to the same value we stored.
    assert digest == hash_api_key(key)
    # SHA-256 hex fits the String(64) column exactly.
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_hash_differs_for_different_keys():
    assert hash_api_key(generate_api_key()) != hash_api_key(generate_api_key())


def test_hash_does_not_contain_plaintext():
    # A DB dump must not leak the usable credential.
    key = generate_api_key()
    assert key not in hash_api_key(key)


def test_key_prefix_length_matches_stored_column():
    key = generate_api_key()
    prefix = key_prefix(key)
    assert prefix == key[:KEY_PREFIX_STORED_LEN]
    assert len(prefix) == KEY_PREFIX_STORED_LEN
