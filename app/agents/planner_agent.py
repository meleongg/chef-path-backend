import operator
import uuid
from typing import TypedDict, Annotated, List, Literal
from langchain_core.messages import AnyMessage, ToolMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from app.services.adaptive_planner import (
    get_recipe_candidates,
    generate_and_save_new_recipe,
    finalize_recipe_selection,
)
from app.constants import GENERATIVE_MODEL
from app.errors.planner_agent import NoRecipesSelectedError
from app.agents.global_state import CHECKPOINT_SAVER_INSTANCE


# --- Define the Graph State Schema ---
class PlanState(TypedDict):
    """Represents the state of the adaptive meal planning workflow."""

    # Conversation History (MUST use operator.add to append messages)
    messages: Annotated[List[AnyMessage], operator.add]

    # Structured context from the DB
    user_id: uuid.UUID
    user_goal: str

    # Result of the Hybrid Retrieval Tool
    candidate_recipes: List[str]  # List of recipe IDs/summaries found by Vector Search

    # Agent decision marker
    next_action: Literal["tool", "generate", "end"]

    # User's preferred number of meals per week
    frequency: int


# System prompt for the AI agent
SYSTEM_PROMPT = """You are ChefPath, an expert adaptive meal planning assistant.

Your goal is to find recipes for the user by calling tools. Follow this workflow:

CONTEXT PROVIDED TO YOU:
- user_id: The UUID of the current user (available in the conversation context)
- user_goal: The user's cooking goal (e.g., "techniques", "health", etc.)
- frequency: Number of meals requested per week

WORKFLOW:
Step 1: Call get_recipe_candidates tool
- CRITICAL: Pass the user_id EXACTLY as provided in the context as a UUID object, not as a string placeholder
- Do NOT pass similarity_threshold parameter (let it use the default)
- Only pass: intent_query (your search string), user_id (the actual UUID), exclude_ids (if any), and limit (number requested)
- Create a search query based on user preferences (cuisine, difficulty, goals)
- This will return recipes with IDs or "No suitable recipes found"

Step 2: Handle the search result
- If recipes are found: Call finalize_recipe_selection with those recipe IDs
  - Extract IDs from format "ID: <uuid>"
  - Pass exactly the number of IDs requested
  - IMPORTANT: Even if the difficulty doesn't perfectly match the user's level, SELECT THE RECIPES ANYWAY
  - The system will handle difficulty appropriately - your job is to select recipes that match cuisine/goal preferences
- If NO recipes found (empty result) after 2 attempts: Respond with a helpful message
  - Explain that no matching recipes exist in our database
  - Suggest the user adjust their criteria (different cuisine, difficulty, etc.)
  - Or offer to generate a custom recipe (use generate_and_save_new_recipe sparingly)

RULES:
- NEVER use placeholder strings like "user_id" or "user_id_placeholder" - use the ACTUAL UUID from context
- Do NOT override similarity_threshold - omit it from your tool call
- ALWAYS call finalize_recipe_selection if get_recipe_candidates returns ANY recipes (even if difficulty doesn't match perfectly)
- Only give up and respond with text if get_recipe_candidates returns "No suitable recipes found" after 2 attempts
- Always prefer existing recipes over generating new ones
- Keep responses concise and actionable

The system will present the final plan after finalize_recipe_selection succeeds."""

# init AI agent with tools bound
tools = [get_recipe_candidates, generate_and_save_new_recipe, finalize_recipe_selection]
llm = ChatOpenAI(model=GENERATIVE_MODEL, temperature=0)
llm_with_tools = llm.bind_tools(tools)

# Create tool node for executing tools
tool_node = ToolNode(tools)


# main LLM agent reasoning node
def call_agent_reasoner(state: PlanState) -> PlanState:
    """
    The main reasoning node. Prompts the LLM to decide the next action (Tool Call or Response).
    Includes error handling for network/API failures.
    """
    print("\n[call_agent_reasoner] === ENTERED ===")
    print(
        f"[call_agent_reasoner] Current candidate_recipes: {state.get('candidate_recipes', [])}"
    )
    print(f"[call_agent_reasoner] Message count: {len(state.get('messages', []))}")
    print(f"[call_agent_reasoner] User goal: {state['user_goal']}")
    print(f"[call_agent_reasoner] User ID: {state['user_id']}")
    print(f"[call_agent_reasoner] Frequency: {state['frequency']}")

    try:
        print(f"[call_agent_reasoner] Invoking agent with state messages...")
        # Invoke LLM with system prompt and user messages
        # Add context about user_id, goal, and frequency to help LLM use correct values
        context_message = SystemMessage(
            content=f"{SYSTEM_PROMPT}\n\nCURRENT CONTEXT:\n- user_id: {state['user_id']}\n- user_goal: {state['user_goal']}\n- frequency: {state['frequency']} meals"
        )
        messages = [context_message] + state["messages"]
        response = llm_with_tools.invoke(messages)

        print(f"[call_agent_reasoner] Agent response type: {type(response)}")
        print(
            f"[call_agent_reasoner] Agent response content: {response.content[:200] if response.content else 'No content'}"
        )

        # response is an AIMessage directly, not a dict
        print(
            f"[call_agent_reasoner] Has tool_calls: {bool(getattr(response, 'tool_calls', False))}"
        )

        if getattr(response, "tool_calls", False):
            print(f"[call_agent_reasoner] Tool calls: {response.tool_calls}")
            return {"messages": [response], "next_action": "tool"}
        else:
            print(f"[call_agent_reasoner] No tool calls, ending cycle")
            return {"messages": [response], "next_action": "end"}

    except Exception as e:
        print(f"❌ EXCEPTION in call_agent_reasoner: {type(e).__name__}: {e}")
        error_message = f"AI Error: Unable to communicate with the planning engine due to a network issue. Please try again. ({type(e).__name__})"
        return {"messages": [AIMessage(content=error_message)], "next_action": "end"}


# tool execution node
def execute_tool(state: PlanState) -> PlanState:
    """Execute the tool called by the LLM and return the result as a ToolMessage.
    Includes error handling for tool execution (e.g., database connection failure).
    """
    print("\n[execute_tool] === ENTERED ===")
    print(
        f"[execute_tool] Current candidate_recipes BEFORE tool: {state.get('candidate_recipes', [])}"
    )

    # The last message contains the tool call request from the LLM
    tool_call = state["messages"][-1].tool_calls[0]
    print(f"[execute_tool] Tool name: {tool_call.get('name', 'unknown')}")
    print(f"[execute_tool] Tool args: {tool_call.get('args', {})}")

    try:
        # Use ToolNode to execute the tool
        result = tool_node.invoke(state)
        print(f"[execute_tool] Tool result type: {type(result)}")
        print(f"[execute_tool] Tool result: {result}")

        # Check if finalize_recipe_selection was called
        tool_name = tool_call.get("name")
        if tool_name == "finalize_recipe_selection":
            recipe_ids = tool_call.get("args", {}).get("recipe_ids", [])
            print(f"[execute_tool] Finalizing selection with recipe_ids: {recipe_ids}")
            return {
                "messages": result["messages"],
                "candidate_recipes": recipe_ids,
                "next_action": "end",
            }

        print(
            f"[execute_tool] Current candidate_recipes AFTER tool: {state.get('candidate_recipes', [])}"
        )

        return result

    except Exception as e:
        error_content = f"Tool Execution Failed: The database retrieval encountered an error (e.g., connection timeout or bad query syntax). Error Type: {type(e).__name__}"
        return {
            "messages": [
                ToolMessage(content=error_content, tool_call_id=tool_call["id"])
            ]
        }


# output node
def finalize_plan_output(state: PlanState) -> PlanState:
    """
    Processes the final tool output (which should be the submitted plan)
    and updates the PlanState with the final list of UUIDs.
    """
    print("\n[finalize_plan_output] === ENTERED ===")
    print(f"[finalize_plan_output] Full state keys: {state.keys()}")
    print(f"[finalize_plan_output] State: {state}")

    # assume recipes are provided (as a result of previous graph nodes)
    final_selections = state.get("candidate_recipes", [])
    print(f"[finalize_plan_output] Final selections: {final_selections}")
    print(f"[finalize_plan_output] Final selections type: {type(final_selections)}")
    print(f"[finalize_plan_output] Final selections length: {len(final_selections)}")

    if not final_selections:
        print("[finalize_plan_output] ❌ ERROR: No candidate recipes found!")
        # Check if agent provided an explanation in the last message
        last_message = state["messages"][-1]
        if hasattr(last_message, "content") and last_message.content:
            # Agent explained why no recipes were found - this is acceptable
            agent_explanation = last_message.content
            print(
                f"[finalize_plan_output] Agent explanation: {agent_explanation[:200]}"
            )
            raise NoRecipesSelectedError(
                f"No recipes found matching the criteria. Agent response: {agent_explanation}"
            )
        else:
            # No explanation - something went wrong
            raise NoRecipesSelectedError(
                "No candidate recipes were selected by the agent. Cannot finalize plan."
            )

    print(f"[finalize_plan_output] ✅ Returning {len(final_selections)} recipes")
    return {"candidate_recipes": final_selections}


# conditional router
def route_agent_action(state: PlanState) -> str:
    """Determines the next step based on the Agent's last message."""
    print("\n[route_agent_action] === ROUTING ===")
    last_message = state["messages"][-1]
    has_tool_calls = bool(
        last_message.tool_calls if hasattr(last_message, "tool_calls") else False
    )
    print(f"[route_agent_action] Last message has tool_calls: {has_tool_calls}")
    print(
        f"[route_agent_action] Current candidate_recipes: {state.get('candidate_recipes', [])}"
    )

    if has_tool_calls:
        # If the LLM requested a tool, go execute it
        print("[route_agent_action] → Routing to 'tool'")
        return "tool"
    else:
        # If the LLM responded with text (no tool calls), check if we have recipes
        candidate_recipes = state.get("candidate_recipes", [])
        if candidate_recipes:
            # Agent found recipes and finalized them, go to finalizer
            print("[route_agent_action] → Routing to 'end' (finalizer)")
            return "end"
        else:
            # Agent couldn't find recipes and gave up gracefully
            # This is actually a valid end state - the agent explained the situation
            print(
                "[route_agent_action] → Routing to 'end' (no recipes, agent gave explanation)"
            )
            # We'll raise a clear error in the finalizer
            return "end"


# --- Build the Graph ---
planner_builder = StateGraph(PlanState)

planner_builder.add_node("agent", call_agent_reasoner)
planner_builder.add_node("tool", execute_tool)
planner_builder.add_node("finalizer", finalize_plan_output)

planner_builder.set_entry_point("agent")

# Define Conditional Edge: After the agent reasons, does it need a tool or is it done?
planner_builder.add_conditional_edges(
    "agent", route_agent_action, {"tool": "tool", "end": "finalizer"}
)

# Define Loop Edge: After the tool runs, go back to the agent to reason about the tool output
planner_builder.add_edge("tool", "agent")

planner_builder.add_edge("finalizer", END)

# Compile the final graph (The runnable agent)
AdaptivePlannerAgent = planner_builder.compile(checkpointer=CHECKPOINT_SAVER_INSTANCE)
