from typing import Any, Dict, List, Optional, TypedDict
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
import os
from dotenv import load_dotenv
from fallback_agent import get_orders


# Load env
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in environment variables")


# Initialize model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0,
)


# Define State with optional fields
class Node:
    def __init__(
        self,
        messages: List[BaseMessage],
        parent: Optional["Node"] = None,
    ):
        self.messages = messages
        self.parent = parent
        self.children = []
        self.value = 0
        self.visits = 0
        self.height = 1 if not parent else parent.height + 1
        self.is_solved = False

class TreeState(TypedDict):
    root: Node
    input: str


# -----------------------------------------------------------
# Create agents with tools
# -----------------------------------------------------------

generator = create_agent(
    model=llm,
    tools=[get_orders],
    system_prompt=(
        "You are the generator. Answer the user's request and "
        "call the get_orders tool when needed."
    ),
)

reflector = create_agent(
    model=llm,
    system_prompt=(
        "You are a critic agent reviewing the generator's answer. "
        "If the answer includes any 'order_id', respond with: "
        "'The answer must NOT include order IDs. Please remove them and provide the corrected answer.' "
        "If the answer is too verbose, respond with: "
        "'The answer should be more concise and focused.' "
        "If the answer is polite but not confident, respond with: "
        "'Please provide a confident and clear answer.' "
        "If the answer follows all policies, respond with: 'No further improvement needed.'"
    ),
)


def generator_agent(state: TreeState) -> TreeState:
    # Call generator with current input messages
    messages = state["root"].messages
    response = generator.invoke({"messages": messages})
    new_messages = messages + response["messages"]
    child_node = Node(messages=new_messages, parent=state["root"])
    state["root"].children.append(child_node)
    state["root"] = child_node
    return state


def reflection_agent(state: TreeState) -> TreeState:
    messages = state["root"].messages

    # Filter out empty messages
    filtered = [m for m in messages if hasattr(m, "content") and m.content.strip() != ""]

    # Find last user message (HumanMessage)
    user_msgs = [m for m in filtered if isinstance(m, HumanMessage)]
    last_user = user_msgs[-1] if user_msgs else None

    # Find last generator message - assume last non-user message
    last_gen = None
    for m in reversed(filtered):
        if not isinstance(m, HumanMessage):
            last_gen = m
            break

    if not last_user or not last_gen:
        print("Warning: Could not find user or generator messages properly.")
        return state

    reflector_messages = [last_user, last_gen]

    print("Messages to reflector cleaned:", [m.content for m in reflector_messages])

    response = reflector.invoke({"messages": reflector_messages})

    # Rest as before...
    new_messages = messages + response["messages"]
    if any(
        "no further improvement" in msg.content.lower()
        for msg in response["messages"]
        if hasattr(msg, "content")
    ):
        state["root"].is_solved = True

    child_node = Node(messages=new_messages, parent=state["root"])
    state["root"].children.append(child_node)
    state["root"] = child_node
    return state


def should_loop(state: TreeState):
    root = state["root"]
    if root.is_solved:
        return END
    if root.height > 2:
        return END
    return "expand"


def expand(state: TreeState) -> TreeState:
    # Run generator then reflection in sequence
    state = generator_agent(state)
    state = reflection_agent(state)
    return state


def start(state: TreeState) -> TreeState:
    # Initialize root node with the user's initial input message
    initial_message = HumanMessage(content=state["input"])
    root = Node(messages=[initial_message])
    state["root"] = root
    return state


# Build the graph
builder = StateGraph(TreeState)
builder.add_node("start", start)
builder.add_node("expand", expand)
builder.set_entry_point("start")

builder.add_conditional_edges("start", should_loop)
builder.add_conditional_edges("expand", should_loop)

graph = builder.compile()


# To run:

if __name__ == "__main__":
    initial_input = "Show me the recent orders."
    result_state = graph.invoke({"input": initial_input})

    # Extract final messages from the root node
    final_messages = result_state["root"].messages
    final_contents = [msg.content for msg in final_messages if hasattr(msg, "content")]

    print("=== Conversation Log ===")
    for content in final_contents:
        print(content)
