import os
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify

from langchain_groq import ChatGroq
from langchain.tools import tool
from langgraph.graph import StateGraph, END

from typing import TypedDict, List

# ----------------------
# 🧰 Tools
# ----------------------

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        return str(eval(expression))  # intentionally unsafe for red teaming
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def echo(text: str) -> str:
    """Echo back the input text."""
    return f"Echo: {text}"

tools = [calculator, echo]

# ----------------------
# 🧠 LLM (Groq)
# ----------------------

llm = ChatGroq(
    model="llama-3.1-8b-instant",  # fast + free-tier friendly
    temperature=0
)

# ----------------------
# 🧩 State
# ----------------------

class AgentState(TypedDict):
    messages: List[dict]

# ----------------------
# 🤖 Agent Node
# ----------------------

def agent_node(state: AgentState):
    messages = state["messages"]

    response = llm.invoke(messages)

    return {
        "messages": messages + [
            {"role": "assistant", "content": response.content}
        ]
    }

# ----------------------
# 🔧 Tool Node
# ----------------------

def tool_node(state: AgentState):
    last_message = state["messages"][-1]["content"]

    # intentionally weak parsing for red team scenarios
    if "calculate:" in last_message:
        expr = last_message.split("calculate:")[-1].strip()
        result = calculator.invoke(expr)
        return {
            "messages": state["messages"] + [
                {"role": "tool", "content": result}
            ]
        }

    if "echo:" in last_message:
        text = last_message.split("echo:")[-1].strip()
        result = echo.invoke(text)
        return {
            "messages": state["messages"] + [
                {"role": "tool", "content": result}
            ]
        }

    return state

# ----------------------
# 🔀 Routing Logic
# ----------------------

def should_continue(state: AgentState):
    last = state["messages"][-1]["content"]

    if "calculate:" in last or "echo:" in last:
        return "tools"
    return END

# ----------------------
# 🔗 Graph
# ----------------------

graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END,
    },
)

graph.add_edge("tools", "agent")

app_graph = graph.compile()

# ----------------------
# 🌐 Flask API
# ----------------------

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("prompt", "")

    state = {
        "messages": [
            {"role": "user", "content": user_input}
        ]
    }

    result = app_graph.invoke(state)

    return jsonify({
        "response": result["messages"][-1]["content"],
        "trace": result["messages"]  # useful for red team analysis
    })

# ----------------------
# 🚀 Run
# ----------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
