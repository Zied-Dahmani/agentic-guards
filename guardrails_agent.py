from typing import Any
import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import (
    PIIMiddleware,
    HumanInTheLoopMiddleware,
    AgentMiddleware,
    AgentState,
    before_model,
    after_model,
    hook_config,
)
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import AIMessage
from langgraph.types import Command

# ------------------------------------------------------
# Load API Key
# ------------------------------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0,
)


# ------------------------------------------------------
# 1. Deterministic Content Filter Middleware
# ------------------------------------------------------
class ContentFilterMiddleware(AgentMiddleware):
    """Blocks inputs with banned keywords before model runs."""

    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime):
        if not state["messages"]:
            return None

        first_message = state["messages"][0]
        if first_message.type != "human":
            return None

        content = first_message.content.lower()

        for keyword in self.banned_keywords:
            if keyword in content:
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": "I cannot process requests containing inappropriate content. Please rephrase."
                    }],
                    "jump_to": "end"
                }

        return None


@before_model(can_jump_to=["end"])
def check_message_limit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    if len(state["messages"]) >= 10:
        return {
            "messages": [AIMessage("Conversation limit reached.")],
            "jump_to": "end"
        }
    return None

@after_model
def log_response(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"Model returned: {state['messages'][-1].content}")
    return None


# ------------------------------------------------------
# Combined Agent
# ------------------------------------------------------
agent = create_agent(
    model=llm,
    middleware=[
        # Layer 1 — Hard Filters BEFORE agent
        ContentFilterMiddleware(banned_keywords=["hack", "exploit", "malware"]),

        # Layer 2 — PII protection (input + output)
        PIIMiddleware("email", strategy="redact", apply_to_input=True, apply_to_output=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),

        # Layer 3 — Human-in-the-loop (approval for specific tools)
        HumanInTheLoopMiddleware(interrupt_on={"send_email": True}),

        # Layer 4 — Hard Filters AFTER model
        check_message_limit,
        log_response,
    ],
    # Needed for Human-in-the-loop
    checkpointer=InMemorySaver(),
)


# ------------------------------------------------------
# Example Run
# ------------------------------------------------------
# Human-in-the-loop requires a thread ID for persistence
config = {"configurable": {"thread_id": "some_id"}}

# Agent will pause and wait for approval before executing sensitive tools
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Send an email to zied.dahmani2@gmail.com with my car details"}]},
    config=config
)

result = agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config  # Same thread ID to resume the paused conversation
)

print("Final Agent Response:", result["messages"][-1].content)