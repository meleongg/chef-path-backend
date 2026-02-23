from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Bcrypt has a maximum password length of 72 bytes
MAX_PASSWORD_LENGTH = 72


def hash_password(password: str) -> str:
    """Hash a plaintext password for storage.

    Bcrypt has a 72-byte limit, so we truncate longer passwords.
    This is safe because users with >72 byte passwords still get
    a unique hash based on the first 72 bytes.
    """
    # Convert to bytes and truncate to 72 bytes, then back to string
    password_bytes = password.encode("utf-8")[:MAX_PASSWORD_LENGTH]
    truncated = password_bytes.decode("utf-8", errors="ignore")
    return pwd_context.hash(truncated)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a stored hash.

    Must truncate the same way as hash_password
    to ensure consistent comparison.
    """
    # Truncate the same way
    password_bytes = plain_password.encode("utf-8")[:MAX_PASSWORD_LENGTH]
    truncated = password_bytes.decode("utf-8", errors="ignore")
    return pwd_context.verify(truncated, hashed_password)
