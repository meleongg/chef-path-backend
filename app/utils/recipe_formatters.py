"""
Utility functions for recipe instruction conversion.
Handles conversion between JSON and text formats.
"""

from typing import List, Dict, Union


def instructions_json_to_text(
    instructions_json: Union[List[Dict[str, any]], str]
) -> str:
    """
    Convert structured instruction JSON back to plain text format.
    Useful for embedding generation or display in non-JSON contexts.

    Args:
        instructions_json: List of instruction dictionaries with step/text, or JSON string

    Returns:
        Newline-separated instruction text
    """
    import json

    # Handle both JSON string and list inputs
    if isinstance(instructions_json, str):
        try:
            instructions_json = json.loads(instructions_json)
        except json.JSONDecodeError:
            return instructions_json  # Return as-is if not valid JSON

    if not instructions_json:
        return ""

    # Sort by step number to ensure correct order
    sorted_steps = sorted(instructions_json, key=lambda x: x.get("step", 0))

    # Join with newlines
    return "\n".join(
        [step.get("text", "") for step in sorted_steps if step.get("text")]
    )
