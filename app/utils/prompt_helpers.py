"""
Helper functions for building LLM prompts with expanded descriptions.
"""

from app.constants import COOKING_GOAL_DESCRIPTIONS, SKILL_LEVEL_DESCRIPTIONS


def get_goal_description(goal_key: str) -> str:
    """
    Returns expanded goal description for LLM prompts.

    Args:
        goal_key: The short goal key (e.g., "techniques", "health")

    Returns:
        Expanded description (e.g., "Learn New Techniques (e.g., knife skills, sauces, dough)")
        Falls back to the key itself if not found.
    """
    return COOKING_GOAL_DESCRIPTIONS.get(goal_key, goal_key)


def get_skill_description(skill_key: str) -> str:
    """
    Returns expanded skill level description for LLM prompts.

    Args:
        skill_key: The short skill key (e.g., "beginner", "intermediate")

    Returns:
        Expanded description (e.g., "Beginner - Just starting out, needs simple recipes")
        Falls back to the key itself if not found.
    """
    return SKILL_LEVEL_DESCRIPTIONS.get(skill_key, skill_key)
