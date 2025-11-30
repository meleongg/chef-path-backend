import operator
import uuid
import re
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

Your goal is to find or generate recipes for the user by calling tools. Follow this workflow:

CONTEXT PROVIDED TO YOU:
- user_id: The UUID of the current user (available in the conversation context)
- user_goal: The user's cooking goal (e.g., "techniques", "health", etc.)
- frequency: Number of meals requested per week

WORKFLOW:
Step 1: Call get_recipe_candidates tool
- CRITICAL: Pass the user_id EXACTLY as provided in the context as a UUID object, not as a string placeholder
- Do NOT pass similarity_threshold parameter (let it use the default)
- Only pass: intent_query (your search string), user_id (the actual UUID), exclude_ids (if any), and limit (set to frequency)
- Create a search query based on user preferences (cuisine, difficulty, goals)
- IMPORTANT: This tool automatically populates candidate_recipes state with the found recipe IDs
- The tool will also return a message indicating if there's a shortfall (e.g., "Found 2/3 recipes. Need 1 more")

Step 2: Check the candidate_recipes state after search
- After get_recipe_candidates returns, check the candidate_recipes list in the state
- Compare len(candidate_recipes) with frequency (requested number of meals)

- If len(candidate_recipes) >= frequency:
  ✅ You have enough recipes!
  - Call finalize_recipe_selection with the recipe IDs from candidate_recipes
  - CRITICAL: Do NOT generate additional recipes
  - Do NOT generate extra recipes due to difficulty mismatch, skill level, or any other reason
  - Accept the recipes found, even if difficulty doesn't perfectly match user skill level

- If len(candidate_recipes) < frequency:
  ⚠️ Not enough recipes - need to generate more
  a) Calculate how many more needed: shortfall = frequency - len(candidate_recipes)
  b) Call generate_and_save_new_recipe ONCE for ONE missing recipe
     - Base it on: user preferences, cuisine, difficulty level, and goal
     - Example: "A beginner-friendly Mexican taco recipe focusing on building confidence"
     - The tool will automatically add the new recipe ID to candidate_recipes
  c) After generation, check if len(candidate_recipes) >= frequency
  d) If still short, generate one more recipe (repeat until len(candidate_recipes) == frequency)
  e) Once you have enough, call finalize_recipe_selection with ALL IDs from candidate_recipes

- If NO recipes found (len(candidate_recipes) == 0):
  a) Try one more search with a broader/different query
  b) If still no results, generate ALL needed recipes one at a time
  c) Call finalize_recipe_selection with all generated IDs

Step 3: Recipe Generation Guidelines (when using generate_and_save_new_recipe)
- Always include: cuisine, difficulty, user_goal in your recipe_description
- Make descriptions specific and actionable
- Generate recipes ONE AT A TIME (don't call multiple times in parallel)
- After each generation, the new recipe ID is automatically added to candidate_recipes
- Check candidate_recipes count after each generation before generating more

RULES:
- NEVER use placeholder strings like "user_id" or "user_id_placeholder" - use the ACTUAL UUID from context
- Do NOT override similarity_threshold - omit it from your tool call
- ALWAYS prefer existing recipes over generating new ones
- candidate_recipes state is automatically populated by tools - you don't need to manually extract IDs
- Check len(candidate_recipes) vs frequency BEFORE deciding to generate recipes
- Only generate recipes when len(candidate_recipes) < frequency (genuine shortfall)
- Generate recipes ONE AT A TIME, checking the count after each generation
- ALWAYS call finalize_recipe_selection as the final step with ALL IDs from candidate_recipes
- Keep responses concise and actionable
- Do NOT generate extra recipes due to difficulty mismatch if you already have frequency recipes
- If a tool call fails, check len(candidate_recipes) >= frequency before retrying
- Do NOT retry failed recipe generation more than once - if it fails, proceed with what you have

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
    Handles both recipe search and recipe generation tools.
    """
    print("\n[execute_tool] === ENTERED ===")
    print(
        f"[execute_tool] Current candidate_recipes BEFORE tool: {state.get('candidate_recipes', [])}"
    )

    # The last message contains the tool call request from the LLM
    tool_call = state["messages"][-1].tool_calls[0]
    tool_name = tool_call.get("name")
    print(f"[execute_tool] Tool name: {tool_name}")
    print(f"[execute_tool] Tool args: {tool_call.get('args', {})}")

    try:
        # Use ToolNode to execute the tool
        result = tool_node.invoke(state)
        print(f"[execute_tool] Tool result type: {type(result)}")
        print(f"[execute_tool] Tool result: {result}")

        # Handle finalize_recipe_selection - marks the end of the workflow
        if tool_name == "finalize_recipe_selection":
            recipe_ids = tool_call.get("args", {}).get("recipe_ids", [])
            print(f"[execute_tool] Finalizing selection with recipe_ids: {recipe_ids}")
            return {
                "messages": result["messages"],
                "candidate_recipes": recipe_ids,
                "next_action": "end",
            }

        # Handle get_recipe_candidates - extract IDs and populate state
        elif tool_name == "get_recipe_candidates":
            # Extract recipe IDs from the tool response
            tool_message = result["messages"][0]
            content = tool_message.content

            # Parse all recipe IDs from format "ID: <uuid>"
            found_ids = re.findall(r"ID: ([a-f0-9\-]+)", content)

            if found_ids:
                print(f"[execute_tool] Found {len(found_ids)} recipe IDs from search")
                print(f"[execute_tool] Recipe IDs: {found_ids}")

                # Populate candidate_recipes state so agent knows what we have
                return {
                    "messages": result["messages"],
                    "candidate_recipes": found_ids,
                }
            else:
                print(f"[execute_tool] ⚠️ No recipe IDs found in search response")
                return result

        # Handle generate_and_save_new_recipe - adds to candidates
        elif tool_name == "generate_and_save_new_recipe":
            # Extract the newly generated recipe ID from the tool response
            tool_message = result["messages"][0]
            content = tool_message.content

            # Parse the recipe ID from response format "Successfully generated recipe: <uuid>"
            match = re.search(r"Successfully generated recipe: ([a-f0-9\-]+)", content)

            if match:
                new_recipe_id = match.group(1)
                print(f"[execute_tool] Generated new recipe ID: {new_recipe_id}")

                # Add to existing candidates (don't replace them)
                current_candidates = state.get("candidate_recipes", [])
                updated_candidates = current_candidates + [new_recipe_id]

                print(f"[execute_tool] Updated candidate_recipes: {updated_candidates}")
                return {
                    "messages": result["messages"],
                    "candidate_recipes": updated_candidates,
                }
            else:
                print(
                    f"[execute_tool] ⚠️ Could not extract recipe ID from generation response"
                )
                return result

        # For other tools, just return the messages
        print(
            f"[execute_tool] Current candidate_recipes AFTER tool: {state.get('candidate_recipes', [])}"
        )

        return result

    except Exception as e:
        error_content = f"Tool Execution Failed: The database retrieval encountered an error (e.g., connection timeout or bad query syntax). Error Type: {type(e).__name__}: {str(e)}"
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

# Compile the graph without a checkpointer
# The checkpointer will be passed at runtime via config["checkpointer"]
AdaptivePlannerAgent = planner_builder.compile()
