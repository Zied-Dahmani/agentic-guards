"""
Agent with layered middleware protections using LangChain + LangGraph.

Layers:
1. Hard content filtering (blocks banned keywords before model).
2. PII protection (redact/mask email and credit cards).
3. Human-in-the-loop approval for sensitive tools (send_email).
4. Model-level guards (message limit + model response logging).
"""

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
from langchain_core.tools import tool


# ==========================================================
# Load API Key
# ==========================================================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY environment variable.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0,
)


# ==========================================================
# 1. Content Filter Middleware
# ==========================================================
class ContentFilterMiddleware(AgentMiddleware):
    """Blocks user messages containing banned keywords before the agent executes."""

    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime):
        """Checks the first human message for banned keywords."""
        if not state["messages"]:
            return None

        msg = state["messages"][0]
        if msg.type != "human":
            return None

        content = msg.content.lower()

        for keyword in self.banned_keywords:
            if keyword in content:
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": (
                            "I cannot process requests containing restricted content. "
                            "Please rephrase your message."
                        )
                    }],
                    "jump_to": "end"
                }

        return None


# ==========================================================
# 2. Model-Level Guardrails
# ==========================================================
@before_model(can_jump_to=["end"])
def check_message_limit(state: AgentState, runtime: Runtime):
    """Stops execution if message count exceeds 10 to prevent overly long threads."""
    if len(state["messages"]) >= 10:
        return {
            "messages": [AIMessage(content="Conversation limit reached.")],
            "jump_to": "end"
        }
    return None


@after_model
def log_response(state: AgentState, runtime: Runtime):
    """Logs the model's final output for debugging."""
    print(f"[DEBUG] Model output: {state['messages'][-1].content}")
    return None


# ==========================================================
# 3. Tools
# ==========================================================
@tool
def send_email() -> str:
    """Simulated sensitive tool that 'sends' an email."""
    return "Email sent successfully."


# ==========================================================
# 4. Create Agent with Middleware Stack
# ==========================================================
agent = create_agent(
    model=llm,
    tools=[send_email],
    system_prompt=(
        "You are a helpful assistant that can send emails when requested. "
        "Always use the send_email tool when the user asks to send an email. "
        "The email body and subject can be simple lorem ipsum text."
    ),
    middleware=[
        # Layer 1: Hard content filtering
        ContentFilterMiddleware(banned_keywords=["hack", "exploit", "malware"]),

        # Layer 2: Automatic PII protection
        PIIMiddleware("email", strategy="redact", apply_to_input=True, apply_to_output=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),

        # Layer 3: Human approval before running sensitive tools
        HumanInTheLoopMiddleware(interrupt_on={"send_email": True}),

        # Layer 4: Model-level safeguards
        check_message_limit,
        log_response,
    ],

    # HITL needs persistent storage of paused state
    checkpointer=InMemorySaver(),
)


# ==========================================================
# 5. Example Execution
# ==========================================================
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "thread_001"}}

    print("\n=== FIRST INVOKE ===")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Send a lorem ipsum email to zied.dahmani2@gmail.com"}]},
        config=config
    )
    print("Agent paused and returned:")

    print("\n=== APPROVE & RESUME ===")
    result = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config
    )
