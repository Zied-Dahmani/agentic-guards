"""
Agentic Guards - Order Retrieval Agent with Fallback using LangChain + LangGraph.

This script demonstrates:
- Creating AI agents with LangChain and LangGraph
- Using tools to simulate order retrieval
- Handling fallback logic if the primary API call fails
- Managing state transitions in a state graph
"""

from typing import Any, Dict, List, NotRequired, TypedDict
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from openai import AuthenticationError
import os


# -----------------------------
# Typed State Definition
# -----------------------------
class State(TypedDict, total=False):
    """
    Defines the shape of the state used in the LangGraph.
    
    Attributes:
        messages: Optional list of chat messages exchanged so far.
        response: Optional dictionary containing the agent's raw response.
        final_answer: Optional string representing the final extracted answer.
    """
    messages: NotRequired[List[BaseMessage]]
    response: NotRequired[Dict[str, Any]]
    final_answer: NotRequired[str]


# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY environment variable.")


# -----------------------------
# Initialize Language Model
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0,
)


# -----------------------------
# Define Tools
# -----------------------------
@tool
def get_orders() -> List[Dict[str, Any]]:
    """
    Simulated tool to retrieve recent orders from the system.

    Raises:
        Exception: Simulated API failure for demonstration purposes.

    Returns:
        List of dictionaries representing orders (if no exception raised).
    """
    raise Exception("Simulated API failure for demonstration purposes.")


@tool
def get_orders_fallback() -> Any:
    """
    Fallback tool that returns yesterday's orders
    if the main order retrieval fails.

    Returns:
        List of fallback orders.
    """
    return [
        {"order_id": 991, "item": "Laptop", "quantity": 2, "status": "Shipped"},
    ]


# -----------------------------
# Define Agents
# -----------------------------
primary_handler = create_agent(
    model=llm,
    tools=[get_orders],
    system_prompt=(
        "You are a helpful assistant that retrieves recent orders. "
        "Always use get_orders when appropriate."
    ),
)

fallback_handler = create_agent(
    model=llm,
    tools=[get_orders_fallback],
    system_prompt=(
        "You are the fallback assistant. "
        "Tell the user that the main system failed and return yesterday's orders."
    ),
)


# -----------------------------
# Define Graph Nodes (State Transitions)
# -----------------------------
def run_agent_node(state: State) -> State:
    """
    Invokes the primary agent to get orders.
    If authentication fails, handles error.
    On any other failure, calls the fallback agent.

    Args:
        state: Current state containing messages.

    Returns:
        Updated state with agent response.
    """
    messages = state.get("messages", [])
    try:
        response = primary_handler.invoke({"messages": messages})
        return {"response": response}
    except AuthenticationError:
        # Handle API auth issues explicitly
        return {"response": {"messages": [{"content": "API authentication failed."}]}}
    except Exception:
        # On any other error, invoke fallback agent
        fallback_response = fallback_handler.invoke({"messages": messages})
        return {"response": fallback_response}


def extract_answer_node(state: State) -> State:
    """
    Extracts the final answer string from the agent's response messages.

    Args:
        state: Current state containing the response dictionary.

    Returns:
        Updated state containing the final answer string.
    """
    messages = state.get("response", {}).get("messages", [])
    if not messages:
        return {"final_answer": "No response."}

    last = messages[-1]

    # Extract content based on possible types
    if hasattr(last, "content"):
        return {"final_answer": last.content}

    if isinstance(last, dict) and "content" in last:
        return {"final_answer": last["content"]}

    return {"final_answer": str(last)}


# -----------------------------
# Build the LangGraph
# -----------------------------
graph = StateGraph(State)

# Register nodes with graph
graph.add_node("run_agent", run_agent_node)
graph.add_node("extract_answer", extract_answer_node)

# Define edges between nodes
graph.add_edge("run_agent", "extract_answer")
graph.add_edge("extract_answer", END)

# Set entry point of the graph
graph.set_entry_point("run_agent")

# Compile the graph into a callable app
app = graph.compile()


# -----------------------------
# Main execution entrypoint
# -----------------------------
if __name__ == "__main__":
    # Initial user message
    input_messages = [HumanMessage(content="Show me the recent orders.")]
    
    # Invoke the app with initial messages
    result = app.invoke({"messages": input_messages})

    print("🤖 LangGraph Agent Result:", result["final_answer"])
