"""
Utility functions for UUID/string conversions.
Used for LLM compatibility since LLMs output strings, but database operations need UUIDs.
"""

import uuid
from typing import List, Union


def uuid_to_str(uid: Union[uuid.UUID, str]) -> str:
    """Convert a UUID to string. If already a string, return as-is."""
    return str(uid) if isinstance(uid, uuid.UUID) else uid


def str_to_uuid(uid: Union[str, uuid.UUID]) -> uuid.UUID:
    """Convert a string to UUID. If already a UUID, return as-is."""
    return uuid.UUID(uid) if isinstance(uid, str) else uid


def uuids_to_strs(uids: List[Union[uuid.UUID, str]]) -> List[str]:
    """Convert a list of UUIDs to strings for LLM compatibility."""
    return [uuid_to_str(uid) for uid in uids]


def strs_to_uuids(uids: List[Union[str, uuid.UUID]]) -> List[uuid.UUID]:
    """Convert a list of strings to UUIDs for database operations."""
    return [str_to_uuid(uid) for uid in uids]
