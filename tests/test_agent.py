"""Runs the real LangChain agent loop offline with a scripted chat model."""

import json

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from eda_agent.agent import ask
from eda_agent.data import load_dataset
from eda_agent.run_log import RunLogger
from eda_agent.tools import Workspace, make_tools


class ScriptedToolModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def call(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name}"}])


def make_ws(path, tmp_path):
    ds = load_dataset(path)
    return Workspace(datasets={ds.name: ds}, out_dir=tmp_path / "out", run_logger=RunLogger(tmp_path / "out" / "runs.jsonl"))


def test_agent_calls_tools_and_writes_report(messy_csv, tmp_path):
    ws = make_ws(messy_csv, tmp_path)
    script = iter([
        call("describe_schema"),
        call("null_audit"),
        call("correlation_analysis", method="spearman"),
        call("plot", kind="scatter", column="x", y="y"),
        call("write_report", summary="x and y are strongly related."),
        AIMessage(content="Done: report written."),
    ])
    answer = ask(ws, "Run an EDA", model=ScriptedToolModel(messages=script))

    assert answer == "Done: report written."
    called = [c["tool"] for c in ws.run_logger.tool_calls]
    assert called == ["describe_schema", "null_audit", "correlation_analysis", "plot", "write_report"]
    assert all(c["ok"] for c in ws.run_logger.tool_calls)
    report = (tmp_path / "out" / "messy_report.md").read_text()
    assert "x and y are strongly related." in report and "scatter_x__y.png" in report


def test_tool_errors_go_back_to_the_model(messy_csv, tmp_path):
    ws = make_ws(messy_csv, tmp_path)
    tools = {t.name: t for t in make_tools(ws)}
    out = json.loads(tools["profile_column"].invoke({"column": "nope"}))
    assert "unknown column" in out["error"]
    assert ws.run_logger.tool_calls[-1]["ok"] is False
