import operator
from typing import TypedDict, Annotated, List, Literal, Optional
from langchain_core.messages import BaseMessage, AnyMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolExecutor
from langchain_openai import ChatOpenAI
from app.services.adaptive_planner import get_recipe_candidates
import uuid


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


# --- Initialize Model and ToolExecutor ---
# We use a powerful model for the Agent's reasoning (e.g., GPT-4o-mini is cost-effective)
LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define the list of tools the agent can call
tools = [get_recipe_candidates]

# Bind the tools to the model (enabling function calling)
LLM_WITH_TOOLS = LLM.bind_tools(tools)

# ToolExecutor handles calling the Python functions when the LLM requests it
tool_executor = ToolExecutor(tools)


# A. Node for LLM reasoning and Tool Invocation
def call_agent_reasoner(state: PlanState) -> PlanState:
    """
    The main reasoning node. Prompts the LLM to decide the next action (Tool Call or Response).
    """
    # Augment prompt with user context and available tools
    user_prompt = f"User Goal: {state['user_goal']}. User ID: {state['user_id']}. Please generate 7 recipes. Do you need to use a tool?"

    # Invoke the LLM with the available tools
    response = LLM_WITH_TOOLS.invoke(
        [HumanMessage(content=user_prompt)] + state["messages"]
    )

    # Check if the LLM chose a tool
    if response.tool_calls:
        # If tool is called, transition to the tool execution node
        return {"messages": [response], "next_action": "tool"}
    else:
        # If no tool is called, transition to the end to respond to the user
        return {"messages": [response], "next_action": "end"}


# B. Node for Tool Execution
def execute_tool(state: PlanState) -> PlanState:
    """Execute the tool called by the LLM and return the result as a ToolMessage."""

    # The last message contains the tool call request from the LLM
    tool_call = state["messages"][-1].tool_calls[0]

    # Execute the Python function (get_recipe_candidates)
    tool_output = tool_executor.invoke(tool_call)

    # Add the tool's result to the messages list
    return {
        "messages": [ToolMessage(content=tool_output, tool_call_id=tool_call["id"])]
    }


# C. Conditional Router (The Edges Logic)
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

# Define Entry Point
planner_builder.set_entry_point("agent")

# Define Conditional Edge: After the agent reasons, does it need a tool or is it done?
planner_builder.add_conditional_edges(
    "agent",
    route_agent_action,
    {"tool": "tool", "end": END}
)

# Define Loop Edge: After the tool runs, go back to the agent to reason about the tool output
planner_builder.add_edge("tool", "agent")

# Compile the final graph (The runnable agent)
AdaptivePlannerAgent = planner_builder.compile()
