"""The LangChain agent: an LLM that plans the analysis and calls profiling tools."""

from __future__ import annotations

import logging
import os

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from .tools import Workspace, make_tools

log = logging.getLogger(__name__)

DEFAULT_MODEL = "anthropic:claude-sonnet-5-5"

SYSTEM_PROMPT = """You are a careful data analyst doing exploratory data analysis.

You have tools that profile the loaded datasets. Every number you state must come
from a tool result in this conversation; never estimate or invent statistics.

Default plan when asked for a general EDA:
1. list_datasets, then describe_schema.
2. null_audit and data_quality_check to judge whether the data can be trusted.
3. distribution_overview, then profile_column for anything unusual (skew, outliers, odd categories).
4. correlation_analysis; plot a scatter for the strongest relationship worth explaining.
5. run_history to see whether this run differs from earlier ones.
6. write_report with a short summary: what the data is, the biggest quality risks,
   the notable distributions and relationships, and questions worth investigating next.

If the user asks a specific question, answer it with the fewest tools needed.
Be explicit when a statistic comes from a sample rather than all rows.
Correlation is not causation; say so when it matters."""


def build_agent(ws: Workspace, model: str | BaseChatModel | None = None):
    model = model or os.environ.get("EDA_AGENT_MODEL", DEFAULT_MODEL)
    log.info("building agent with model=%s", model if isinstance(model, str) else type(model).__name__)
    return create_agent(model, tools=make_tools(ws), system_prompt=SYSTEM_PROMPT)


def ask(ws: Workspace, question: str, model: str | BaseChatModel | None = None, max_steps: int = 40) -> str:
    agent = build_agent(ws, model)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"recursion_limit": max_steps},
    )
    final = next((m for m in reversed(result["messages"]) if isinstance(m, AIMessage)), None)
    if final is None:
        return ""
    content = final.content
    if isinstance(content, list):
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content
