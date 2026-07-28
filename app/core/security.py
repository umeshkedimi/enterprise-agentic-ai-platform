import hashlib
import secrets

# A recognizable prefix so a leaked key is greppable in logs and revocable on
# sight, and so callers can tell our credentials apart from other bearer tokens.
API_KEY_PREFIX = "eaap_sk_"

# How many leading characters we persist for display/identification. Long enough
# to disambiguate keys in a UI, short enough to reveal nothing useful.
KEY_PREFIX_STORED_LEN = 12


def generate_api_key() -> str:
    """Mint a new API key: our prefix plus 256 bits of URL-safe randomness."""
    return f"{API_KEY_PREFIX}{secrets.token_urlsafe(32)}"


def hash_api_key(key: str) -> str:
    """Return the SHA-256 hex digest of a key (64 chars).

    A plain fast hash — not bcrypt/argon2 — is the correct choice here: those
    exist to slow brute force against low-entropy, human-chosen passwords. An
    API key is 256 bits of CSPRNG output and is not brute-forceable, so key
    stretching buys nothing and would only prevent the indexed hash lookup that
    makes authentication a single, fast query.
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def key_prefix(key: str) -> str:
    """The stored, non-secret prefix of a key, for identifying it in a UI."""
    return key[:KEY_PREFIX_STORED_LEN]
