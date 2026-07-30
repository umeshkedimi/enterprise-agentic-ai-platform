"""Symmetric encryption for credentials the platform has to *use*, not verify.

This is deliberately separate from `app/core/security.py`, and the distinction is
the whole point of the module. An API key the platform issues is only ever
checked, so it is stored as a one-way hash and a database dump yields nothing
usable. A credential for somebody else's MCP server has to be replayed on every
call, so it cannot be hashed — the platform needs the plaintext back.

Reversible storage is strictly weaker, and the honest response is to make the
weakness explicit rather than to pretend a hash would do: the ciphertext is
useless without a key held outside the database, and if that key is unset the
platform refuses to accept a credential at all rather than writing one in the
clear.
"""

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import Settings


class EncryptionUnavailableError(RuntimeError):
    """No usable encryption key is configured (→ the operator's wiring)."""


class DecryptionError(RuntimeError):
    """Stored ciphertext could not be read with the configured key.

    Almost always a rotated or replaced key. Distinct from "no key configured"
    because the fix is different: one is a missing setting, the other is a
    setting that no longer matches what is in the database.
    """


def _cipher(settings: Settings) -> Fernet:
    key = settings.credential_encryption_key
    if not key:
        # Fails closed, like the admin token. An unset secret must never be read
        # as "encryption not required".
        raise EncryptionUnavailableError(
            "CREDENTIAL_ENCRYPTION_KEY is not set; the platform will not store "
            "third-party credentials unencrypted."
        )
    try:
        return Fernet(key.encode("utf-8"))
    except (ValueError, TypeError) as exc:
        raise EncryptionUnavailableError(
            "CREDENTIAL_ENCRYPTION_KEY is not a valid Fernet key. Generate one with: "
            "python -c 'from cryptography.fernet import Fernet; "
            "print(Fernet.generate_key().decode())'"
        ) from exc


def encrypt(plaintext: str, settings: Settings) -> str:
    return _cipher(settings).encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt(ciphertext: str, settings: Settings) -> str:
    try:
        return _cipher(settings).decrypt(ciphertext.encode("ascii")).decode("utf-8")
    except InvalidToken as exc:
        raise DecryptionError("stored credential could not be decrypted") from exc
