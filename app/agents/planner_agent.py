import operator
import uuid
from typing import TypedDict, Annotated, List, Literal
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    ToolMessage,
    AIMessage,
)
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from app.services.adaptive_planner import (
    get_recipe_candidates,
    generate_and_save_new_recipe,
)
from scripts.constants import GENERATIVE_MODEL
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
    next_action: Literal["tool", "generate", "critique", "end"]

    # User's preferred number of meals per week
    frequency: int


# init AI agent
llm = ChatOpenAI(model=GENERATIVE_MODEL, temperature=0)
tools = [get_recipe_candidates, generate_and_save_new_recipe]
agent = create_agent(llm, tools=tools)


# main LLM agent reasoning node
def call_agent_reasoner(state: PlanState) -> PlanState:
    """
    The main reasoning node. Prompts the LLM to decide the next action (Tool Call or Response).
    Includes error handling for network/API failures.
    """
    frequency = getattr(state, 'frequency')
    user_prompt = (
        f"User Goal: {state['user_goal']}. "
        f"User ID: {state['user_id']}. "
        f"Your task is to assemble a weekly meal plan (" + str(frequency) + " recipes) for the user. "
        "Use the available tools to retrieve or generate recipes as needed. "
        "Do you need to use a tool to proceed?"
    )

    try:
        response = agent.invoke(
            [HumanMessage(content=user_prompt)] + state["messages"]
        )

        if response.tool_calls:
            return {"messages": [response], "next_action": "tool"}
        else:
            return {"messages": [response], "next_action": "end"}

    except Exception as e:
        error_message = f"AI Error: Unable to communicate with the planning engine due to a network issue. Please try again. ({type(e).__name__})"
        print(f"Agent Reasoning Failed: {e}")
        return {"messages": [AIMessage(content=error_message)], "next_action": "end"}


# tool execution node
def execute_tool(state: PlanState) -> PlanState:
    """Execute the tool called by the LLM and return the result as a ToolMessage.
    Includes error handling for tool execution (e.g., database connection failure).
    """

    # The last message contains the tool call request from the LLM
    tool_call = state["messages"][-1].tool_calls[0]

    try:
        tool_output = agent.invoke(tool_call)

        return {
            "messages": [ToolMessage(content=tool_output, tool_call_id=tool_call["id"])]
        }

    except Exception as e:
        error_content = f"Tool Execution Failed: The database retrieval encountered an error (e.g., connection timeout or bad query syntax). Error Type: {type(e).__name__}"
        print(f"Tool Execution Failed: {e}")
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
    # assume recipes are provided (as a result of previous graph nodes)
    final_selections = state.get("candidate_recipes", [])

    if not final_selections:
        # throw custom error if no recipes returned
        raise NoRecipesSelectedError("No candidate recipes were selected by the agent. Cannot finalize plan.")

    print(f"✅ Finalizer found {len(final_selections)} recipes to commit.")

    return {"candidate_recipes": final_selections}


# Conditional Router (The Edges Logic)
def route_agent_action(state: PlanState) -> str:
    """Determines the next step based on the Agent's last message."""

    if state["messages"][-1].tool_calls:
        # If the LLM requested a tool, go execute it
        return "tool"
    else:
        # If the LLM responded directly, end the cycle
        return "end"


# --- Build the Graph ---
planner_builder = StateGraph(PlanState)

# Add Nodes
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
AdaptivePlannerAgent = planner_builder.compile()
