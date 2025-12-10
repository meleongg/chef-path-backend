"""
LangSmith Evaluation Script for ChefPath Agent

This script programmatically creates datasets and runs evaluations
for intent classification and agent behavior testing.

Usage:
    python scripts/evaluate_agent.py setup     # Create datasets
    python scripts/evaluate_agent.py intent    # Run intent classification eval
    python scripts/evaluate_agent.py agent     # Run agent behavior eval
    python scripts/evaluate_agent.py all       # Run all evaluations
"""

import os
import sys
from typing import Any
from dotenv import load_dotenv
from langsmith import Client, traceable

# Add parent directory to path for imports (allows running from any directory)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.intent_classifier import classify_message_intent
from app.agents.planner_agent import get_agent_with_checkpointer, PlanState
from app.agents.checkpoint_setup import initialize_postgres_saver
from langchain_core.messages import HumanMessage

load_dotenv()

# Initialize LangSmith client
ls_client = Client()

# ============================================================================
# DATASET DEFINITIONS
# ============================================================================

INTENT_CLASSIFICATION_EXAMPLES = [
    {
        "inputs": {"message": "Change the chicken to vegetarian"},
        "outputs": {"intent": "plan_modification"},
    },
    {
        "inputs": {"message": "What temperature should I bake chicken at?"},
        "outputs": {"intent": "general_knowledge"},
    },
    {
        "inputs": {"message": "How many recipes have I completed?"},
        "outputs": {"intent": "analytics"},
    },
    {
        "inputs": {"message": "Replace all recipes with vegan options"},
        "outputs": {"intent": "plan_modification"},
    },
    {
        "inputs": {"message": "Can I substitute butter with oil?"},
        "outputs": {"intent": "general_knowledge"},
    },
    {
        "inputs": {"message": "Swap the pasta dish for a salad"},
        "outputs": {"intent": "plan_modification"},
    },
    {
        "inputs": {
            "message": "What's the difference between baking powder and baking soda?"
        },
        "outputs": {"intent": "general_knowledge"},
    },
    {
        "inputs": {"message": "Show me my cooking statistics"},
        "outputs": {"intent": "analytics"},
    },
    {
        "inputs": {"message": "I want something spicier instead of the mild recipe"},
        "outputs": {"intent": "plan_modification"},
    },
    {
        "inputs": {"message": "How do I properly dice an onion?"},
        "outputs": {"intent": "general_knowledge"},
    },
]

AGENT_BEHAVIOR_EXAMPLES = [
    {
        "inputs": {
            "user_id": "test-user-1",
            "user_goal": "techniques",
            "frequency": 3,
            "exclude_ids": [],
            "candidate_recipes": [],
            "messages": [
                {
                    "content": "I want 3 Italian recipes for beginners focusing on knife skills"
                }
            ],
        },
        "outputs": {"expected_recipe_count": 3},
    },
    {
        "inputs": {
            "user_id": "test-user-2",
            "user_goal": "health",
            "frequency": 2,
            "exclude_ids": ["abc-123-def"],
            "candidate_recipes": [],
            "messages": [{"content": "Give me 2 healthy low-sodium recipes"}],
        },
        "outputs": {"expected_recipe_count": 2},
    },
    {
        "inputs": {
            "user_id": "test-user-3",
            "user_goal": "efficiency",
            "frequency": 4,
            "exclude_ids": [],
            "candidate_recipes": [],
            "messages": [{"content": "4 quick weeknight meals under 30 minutes"}],
        },
        "outputs": {"expected_recipe_count": 4},
    },
    {
        "inputs": {
            "user_id": "test-user-4",
            "user_goal": "cuisine",
            "frequency": 3,
            "exclude_ids": ["xyz-789", "uvw-456"],
            "candidate_recipes": [],
            "messages": [{"content": "3 authentic Mexican recipes"}],
        },
        "outputs": {"expected_recipe_count": 3},
    },
    {
        "inputs": {
            "user_id": "test-user-5",
            "user_goal": "confidence",
            "frequency": 2,
            "exclude_ids": [],
            "candidate_recipes": ["recipe-1", "recipe-2", "recipe-3"],
            "messages": [
                {"content": "Change the first recipe to something vegetarian"}
            ],
        },
        "outputs": {"expected_recipe_count": 3},  # Still 3 total, but one modified
    },
]

# ============================================================================
# EVALUATORS (Property-Based Tests)
# ============================================================================


def correct_recipe_count(run: Any, example: Any) -> dict:
    """Verify agent returns exactly the requested number of recipes."""
    outputs = run.outputs
    inputs = example.inputs

    expected_count = inputs["frequency"]
    actual_count = len(outputs.get("candidate_recipes", []))

    return {
        "key": "correct_recipe_count",
        "score": 1.0 if actual_count == expected_count else 0.0,
        "comment": f"Expected {expected_count}, got {actual_count}",
    }


def no_duplicate_recipes(run: Any, example: Any) -> dict:
    """Verify no duplicate recipe IDs in the output."""
    outputs = run.outputs
    recipes = outputs.get("candidate_recipes", [])

    has_duplicates = len(recipes) != len(set(recipes))

    return {
        "key": "no_duplicates",
        "score": 0.0 if has_duplicates else 1.0,
        "comment": "Duplicates found" if has_duplicates else "No duplicates",
    }


def exclusion_respected(run: Any, example: Any) -> dict:
    """Verify excluded recipes don't appear in output."""
    outputs = run.outputs
    inputs = example.inputs

    excluded = set(inputs.get("exclude_ids", []))
    selected = set(outputs.get("candidate_recipes", []))

    violations = excluded & selected  # Intersection

    return {
        "key": "exclusion_respected",
        "score": 0.0 if violations else 1.0,
        "comment": (
            f"Violations: {list(violations)}"
            if violations
            else "No exclusions violated"
        ),
    }


def finalize_tool_called(run: Any, example: Any) -> dict:
    """Verify agent called finalize_recipe_selection at the end."""
    outputs = run.outputs
    tool_calls = outputs.get("tool_calls", [])

    if not tool_calls:
        return {"key": "finalize_called", "score": 0.0, "comment": "No tool calls"}

    last_tool = tool_calls[-1].get("name")
    finalize_called = last_tool == "finalize_recipe_selection"

    return {
        "key": "finalize_called",
        "score": 1.0 if finalize_called else 0.0,
        "comment": f"Last tool: {last_tool}",
    }


def tool_sequence_valid(run: Any, example: Any) -> dict:
    """Verify tool calls follow expected patterns."""
    outputs = run.outputs
    tool_calls = outputs.get("tool_calls", [])
    tool_names = [tc.get("name") for tc in tool_calls]

    # Rules:
    # 1. Must start with get_recipe_candidates or generate_and_save_new_recipe
    # 2. Must end with finalize_recipe_selection
    # 3. No finalize in the middle

    if not tool_names:
        return {"key": "tool_sequence", "score": 0.0, "comment": "No tools called"}

    valid_start = tool_names[0] in [
        "get_recipe_candidates",
        "generate_and_save_new_recipe",
    ]
    valid_end = tool_names[-1] == "finalize_recipe_selection"
    no_early_finalize = "finalize_recipe_selection" not in tool_names[:-1]

    all_valid = valid_start and valid_end and no_early_finalize

    return {
        "key": "tool_sequence",
        "score": 1.0 if all_valid else 0.0,
        "comment": f"Sequence: {' -> '.join(tool_names[:5])}{'...' if len(tool_names) > 5 else ''}",
    }


# ============================================================================
# TEST TARGETS (What We're Evaluating)
# ============================================================================


@traceable(name="intent_classifier_eval")
def evaluate_intent_classification(inputs: dict) -> dict:
    """Wrapper for intent classification evaluation."""
    message = inputs["message"]
    intent = classify_message_intent(message)
    return {"intent": intent}


@traceable(name="agent_plan_generation_eval")
def evaluate_agent_plan_generation(inputs: dict) -> dict:
    """Wrapper for full agent evaluation."""
    # Extract inputs
    user_message = inputs["messages"][0]["content"]
    user_id = inputs["user_id"]
    user_goal = inputs["user_goal"]
    frequency = inputs["frequency"]
    exclude_ids = inputs.get("exclude_ids", [])
    candidate_recipes = inputs.get("candidate_recipes", [])

    # Build initial state
    initial_state: PlanState = {
        "messages": [HumanMessage(content=user_message)],
        "user_id": user_id,
        "user_goal": user_goal,
        "candidate_recipes": candidate_recipes,
        "frequency": frequency,
        "exclude_ids": exclude_ids,
    }

    # Run agent
    checkpointer = initialize_postgres_saver()
    agent = get_agent_with_checkpointer(checkpointer)

    final_state = agent.invoke(
        initial_state,
        config={
            "configurable": {"thread_id": f"eval-{user_id}"},
            "recursion_limit": 25,
        },
    )

    # Extract tool calls from messages
    tool_calls = []
    for msg in final_state.get("messages", []):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    {
                        "name": tc.get("name"),
                        "args": tc.get("args"),
                    }
                )

    return {
        "candidate_recipes": final_state.get("candidate_recipes", []),
        "tool_calls": tool_calls,
        "message_count": len(final_state.get("messages", [])),
    }


# ============================================================================
# DATASET SETUP
# ============================================================================


def setup_datasets():
    """Create or update datasets in LangSmith."""
    print("\n" + "=" * 80)
    print("DATASET SETUP")
    print("=" * 80 + "\n")

    # Intent Classification Dataset
    intent_dataset_name = "Intent Classification - ChefPath"
    print(f"Setting up dataset: {intent_dataset_name}")

    try:
        # Try to read existing dataset
        intent_dataset = ls_client.read_dataset(dataset_name=intent_dataset_name)
        print(f"  ✓ Found existing dataset (ID: {intent_dataset.id})")

        # Delete existing examples to avoid duplicates
        print(f"  → Clearing old examples...")
        examples = list(ls_client.list_examples(dataset_id=intent_dataset.id))
        for example in examples:
            ls_client.delete_example(example.id)
        print(f"  ✓ Cleared {len(examples)} old examples")

    except Exception:
        # Create new dataset
        print(f"  → Creating new dataset...")
        intent_dataset = ls_client.create_dataset(
            dataset_name=intent_dataset_name,
            description="Test cases for message intent classification (plan_modification, general_knowledge, analytics)",
        )
        print(f"  ✓ Created dataset (ID: {intent_dataset.id})")

    # Add examples
    print(f"  → Adding {len(INTENT_CLASSIFICATION_EXAMPLES)} examples...")
    ls_client.create_examples(
        dataset_id=intent_dataset.id,
        examples=INTENT_CLASSIFICATION_EXAMPLES,
    )
    print(f"  ✓ Added {len(INTENT_CLASSIFICATION_EXAMPLES)} examples\n")

    # Agent Behavior Dataset
    agent_dataset_name = "Agent Behavior - ChefPath"
    print(f"Setting up dataset: {agent_dataset_name}")

    try:
        # Try to read existing dataset
        agent_dataset = ls_client.read_dataset(dataset_name=agent_dataset_name)
        print(f"  ✓ Found existing dataset (ID: {agent_dataset.id})")

        # Delete existing examples
        print(f"  → Clearing old examples...")
        examples = list(ls_client.list_examples(dataset_id=agent_dataset.id))
        for example in examples:
            ls_client.delete_example(example.id)
        print(f"  ✓ Cleared {len(examples)} old examples")

    except Exception:
        # Create new dataset
        print(f"  → Creating new dataset...")
        agent_dataset = ls_client.create_dataset(
            dataset_name=agent_dataset_name,
            description="Test cases for agent tool usage, recipe selection, and output validation",
        )
        print(f"  ✓ Created dataset (ID: {agent_dataset.id})")

    # Add examples
    print(f"  → Adding {len(AGENT_BEHAVIOR_EXAMPLES)} examples...")
    ls_client.create_examples(
        dataset_id=agent_dataset.id,
        examples=AGENT_BEHAVIOR_EXAMPLES,
    )
    print(f"  ✓ Added {len(AGENT_BEHAVIOR_EXAMPLES)} examples\n")

    print("=" * 80)
    print("✓ DATASET SETUP COMPLETE")
    print("=" * 80)
    print(f"\nView datasets at: https://smith.langchain.com/datasets\n")


# ============================================================================
# RUN EVALUATIONS
# ============================================================================


def run_intent_classification_eval():
    """Run intent classification evaluation."""
    print("\n" + "=" * 80)
    print("INTENT CLASSIFICATION EVALUATION")
    print("=" * 80 + "\n")

    dataset_name = "Intent Classification - ChefPath"

    # Verify dataset exists
    try:
        dataset = ls_client.read_dataset(dataset_name=dataset_name)
        print(f"✓ Using dataset: {dataset_name} (ID: {dataset.id})")
    except Exception as e:
        print(f"❌ Dataset '{dataset_name}' not found. Run 'setup' first.")
        print(f"   Error: {e}")
        return

    # Define evaluator
    def intent_accuracy(run: Any, example: Any) -> dict:
        actual_intent = run.outputs.get("intent")
        expected_intent = example.outputs.get("intent")

        return {
            "key": "intent_accuracy",
            "score": 1.0 if actual_intent == expected_intent else 0.0,
            "comment": f"Expected: {expected_intent}, Got: {actual_intent}",
        }

    # Run evaluation
    print(f"\n→ Running evaluation...\n")
    results = ls_client.evaluate(
        evaluate_intent_classification,
        data=dataset_name,
        evaluators=[intent_accuracy],
        experiment_prefix="intent-classification",
        description="Testing intent classification accuracy",
        max_concurrency=4,
    )

    print("\n" + "=" * 80)
    print("✓ EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nView results at: https://smith.langchain.com/experiments\n")
    return results


def run_agent_behavior_eval():
    """Run full agent behavior evaluation."""
    print("\n" + "=" * 80)
    print("AGENT BEHAVIOR EVALUATION")
    print("=" * 80 + "\n")

    dataset_name = "Agent Behavior - ChefPath"

    # Verify dataset exists
    try:
        dataset = ls_client.read_dataset(dataset_name=dataset_name)
        print(f"✓ Using dataset: {dataset_name} (ID: {dataset.id})")
    except Exception as e:
        print(f"❌ Dataset '{dataset_name}' not found. Run 'setup' first.")
        print(f"   Error: {e}")
        return

    # Run evaluation with multiple property-based evaluators
    print(f"\n→ Running evaluation with 5 evaluators...\n")
    results = ls_client.evaluate(
        evaluate_agent_plan_generation,
        data=dataset_name,
        evaluators=[
            correct_recipe_count,
            no_duplicate_recipes,
            exclusion_respected,
            finalize_tool_called,
            tool_sequence_valid,
        ],
        experiment_prefix="agent-behavior",
        description="Testing agent tool usage and output properties",
        max_concurrency=2,  # Lower for DB-heavy operations
    )

    print("\n" + "=" * 80)
    print("✓ EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nView results at: https://smith.langchain.com/experiments\n")
    return results


# ============================================================================
# MAIN
# ============================================================================


def main():
    if len(sys.argv) < 2:
        print("\nUsage: python scripts/evaluate_agent.py [setup|intent|agent|all]\n")
        print("Commands:")
        print("  setup  - Create/update datasets in LangSmith")
        print("  intent - Run intent classification evaluation")
        print("  agent  - Run agent behavior evaluation")
        print("  all    - Run all evaluations\n")
        sys.exit(1)

    command = sys.argv[1]

    if command == "setup":
        setup_datasets()
    elif command == "intent":
        run_intent_classification_eval()
    elif command == "agent":
        run_agent_behavior_eval()
    elif command == "all":
        run_intent_classification_eval()
        print("\n")  # Spacing between evals
        run_agent_behavior_eval()
    else:
        print(f"❌ Unknown command: {command}")
        print("Valid commands: setup, intent, agent, all")
        sys.exit(1)


if __name__ == "__main__":
    main()
