import bcrypt
import hashlib

# Bcrypt has a maximum password length of 72 bytes
MAX_PASSWORD_BYTES = 72
BCRYPT_ROUNDS = 12


def hash_password(password: str) -> str:
    """Hash a plaintext password for storage using bcrypt.

    Bcrypt has a 72-byte limit. We handle this by:
    1. Check if password encoding is ≤72 bytes
    2. If longer, SHA256 hash it first (standard approach)
    3. Then bcrypt the result

    Returns the hash as a utf-8 string for storage in database.
    """
    password_bytes = password.encode("utf-8")

    # If password is >72 bytes, pre-hash with SHA256
    if len(password_bytes) > MAX_PASSWORD_BYTES:
        # Use SHA256 digest (binary) to avoid expanding the size too much
        password_bytes = hashlib.sha256(password_bytes).digest()

    # Hash with bcrypt
    salt = bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a bcrypt hash.

    Must use the same logic as hash_password to handle >72 byte passwords.
    """
    password_bytes = plain_password.encode("utf-8")

    # If password is >72 bytes, pre-hash with SHA256 (same as hash_password)
    if len(password_bytes) > MAX_PASSWORD_BYTES:
        password_bytes = hashlib.sha256(password_bytes).digest()

    # Verify with bcrypt
    try:
        return bcrypt.checkpw(password_bytes, hashed_password.encode("utf-8"))
    except (ValueError, TypeError):
        # Invalid hash format
        return False
