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
# State Definition
# -----------------------------
class State(TypedDict, total=False):
    messages: NotRequired[List[BaseMessage]]
    response: NotRequired[Dict[str, Any]]
    final_answer: NotRequired[str]


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY")


# -----------------------------
# Model
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0,
)


# -----------------------------
# Tools
# -----------------------------
@tool
def get_orders() -> List[Dict[str, Any]]:
    """Get recent orders from the system."""
    return [
        {"order_id": 1, "item": "Laptop", "quantity": 2, "status": "Shipped"},
        {"order_id": 2, "item": "Smartphone", "quantity": 1, "status": "Processing"},
        {"order_id": 3, "item": "Headphones", "quantity": 5, "status": "Delivered"},
    ]


@tool
def get_orders_fallback() -> Any:
    """Fallback tool used when regular order retrieval fails."""
    return [
        {"order_id": 991, "item": "Laptop", "quantity": 2, "status": "Shipped"},
    ]


# -----------------------------
# Agents
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
# Graph Nodes
# -----------------------------
def run_agent_node(state: State) -> State:
    messages = state.get("messages", [])
    try:
        response = primary_handler.invoke({"messages": messages})
        return {"response": response}
    except AuthenticationError:
        return {"response": {"messages": [{"content": "API authentication failed."}]}}
    except Exception:
        fallback_response = fallback_handler.invoke({"messages": messages})
        return {"response": fallback_response}


def extract_answer_node(state: State) -> State:
    messages = state.get("response", {}).get("messages", [])
    if not messages:
        return {"final_answer": "No response."}

    last = messages[-1]

    if hasattr(last, "content"):
        return {"final_answer": last.content}

    if isinstance(last, dict) and "content" in last:
        return {"final_answer": last["content"]}

    return {"final_answer": str(last)}


# -----------------------------
# LangGraph
# -----------------------------
graph = StateGraph(State)

graph.add_node("run_agent", run_agent_node)
graph.add_node("extract_answer", extract_answer_node)

graph.add_edge("run_agent", "extract_answer")
graph.add_edge("extract_answer", END)
graph.set_entry_point("run_agent")

app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({
        "messages": [HumanMessage(content="Show me the recent orders.")]
    })
    print("🤖 LangGraph Agent Result:", result["final_answer"])
