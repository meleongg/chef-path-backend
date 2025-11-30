"""
Intent Classification Service for Chat Routing

This service determines whether a user's message requires:
1. General Q&A (cheap, stateless) - e.g., "What is a roux?"
2. Plan Modification (expensive, stateful) - e.g., "Replace the chicken recipe"
3. Analytics/Progress (medium cost) - e.g., "How many recipes have I completed?"
"""

from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.constants import GENERATIVE_MODEL

IntentType = Literal["general_knowledge", "plan_modification", "analytics"]


class IntentClassifier:
    """Fast, lightweight intent classification for chat routing."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=GENERATIVE_MODEL,
            temperature=0,  # Deterministic for classification
            max_tokens=10,  # Just need a single word response
        )

    def classify_intent(self, user_message: str) -> IntentType:
        """
        Classify user intent into one of three categories.

        Args:
            user_message: The user's chat message

        Returns:
            IntentType: One of "general_knowledge", "plan_modification", or "analytics"
        """

        system_prompt = """You are an intent classifier for a cooking app chat interface.

Classify the user's message into EXACTLY ONE of these categories:

1. "general_knowledge" - Questions about cooking techniques, ingredients, definitions, recipes, how-to guides
   Examples:
   - "What is a roux?"
   - "How do I dice an onion?"
   - "What's the difference between baking and roasting?"
   - "Can you explain how to make pasta?"
   - "What temperature should I cook chicken at?"

2. "plan_modification" - Requests to change, swap, or modify their meal plan or recipes
   Examples:
   - "Replace the chicken recipe"
   - "Swap this with something vegetarian"
   - "I don't like spicy food, change this recipe"
   - "Make it easier"
   - "Remove the taco recipe from my plan"
   - "Add more protein-rich meals"

3. "analytics" - Questions about their progress, stats, or history
   Examples:
   - "How many recipes have I completed?"
   - "What's my progress this week?"
   - "Show me my cooking history"
   - "What difficulty level am I at?"

Respond with ONLY ONE WORD: either "general_knowledge", "plan_modification", or "analytics"
No explanation, no punctuation, just the classification."""

        user_prompt = f"User message: {user_message}"

        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )

            # Extract and normalize the response
            intent = response.content.strip().lower()

            # Validate response
            if intent in ["general_knowledge", "plan_modification", "analytics"]:
                return intent  # type: ignore
            else:
                # Default to general_knowledge if uncertain
                print(
                    f"[IntentClassifier] Unexpected response: {intent}, defaulting to general_knowledge"
                )
                return "general_knowledge"

        except Exception as e:
            print(f"[IntentClassifier] Error during classification: {e}")
            # Safe default: treat as general knowledge (cheapest option)
            return "general_knowledge"


# Singleton instance for reuse across requests
_intent_classifier = None


def get_intent_classifier() -> IntentClassifier:
    """Get or create the singleton intent classifier instance."""
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = IntentClassifier()
    return _intent_classifier


def classify_message_intent(user_message: str) -> IntentType:
    """
    Convenience function to classify a user message.

    Args:
        user_message: The user's chat message

    Returns:
        IntentType: The classified intent
    """
    classifier = get_intent_classifier()
    return classifier.classify_intent(user_message)
