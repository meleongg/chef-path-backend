from passlib.context import CryptContext

# Use bcrypt_sha256 which doesn't have the 72-byte limit
# SHA256 hashes long passwords first, then bcrypt hashes the result
pwd_context = CryptContext(
    schemes=["bcrypt_sha256"],
    deprecated="auto",
    bcrypt_sha256__rounds=12,  # Good balance of security and speed
)


def hash_password(password: str) -> str:
    """Hash a plaintext password for storage.

    Uses bcrypt_sha256 which can handle passwords of any length.
    The password is first hashed with SHA256, then bcrypt hashes the result.
    This avoids bcrypt's 72-byte limitation while maintaining security.
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a stored hash.

    Works with bcrypt_sha256 hashes which can verify any password length.
    """
    return pwd_context.verify(plain_password, hashed_password)
