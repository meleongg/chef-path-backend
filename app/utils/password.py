from passlib.context import CryptContext
import hashlib

# Use bcrypt which is industry standard
# We'll manually enforce the 72-byte limit by hashing long passwords first
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Bcrypt has a maximum password length of 72 bytes
MAX_PASSWORD_BYTES = 72


def hash_password(password: str) -> str:
    """Hash a plaintext password for storage.

    Bcrypt has a 72-byte limit. We handle this by:
    1. Check if password encoding is ≤72 bytes
    2. If longer, SHA256 hash it first (standard approach)
    3. Then bcrypt the result

    This ensures we never pass >72 bytes to bcrypt.
    """
    password_bytes = password.encode("utf-8")

    # If password is ≤72 bytes, hash directly
    if len(password_bytes) <= MAX_PASSWORD_BYTES:
        return pwd_context.hash(password)

    # If >72 bytes, hash with SHA256 first, then bcrypt
    # This is a standard workaround for bcrypt's limitation
    sha256_hash = hashlib.sha256(password_bytes).hexdigest()
    return pwd_context.hash(sha256_hash)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a stored hash.

    Must use the same logic as hash_password to handle >72 byte passwords.
    """
    password_bytes = plain_password.encode("utf-8")

    # If password is ≤72 bytes, verify directly
    if len(password_bytes) <= MAX_PASSWORD_BYTES:
        return pwd_context.verify(plain_password, hashed_password)

    # If >72 bytes, hash with SHA256 first, then verify
    sha256_hash = hashlib.sha256(password_bytes).hexdigest()
    return pwd_context.verify(sha256_hash, hashed_password)
