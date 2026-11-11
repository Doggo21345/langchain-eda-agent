"""Automated exploratory data analysis with a LangChain agent."""

from .data import Dataset, load_dataset
from .run_log import RunLogger
from .tools import Workspace, make_tools

__all__ = ["Dataset", "load_dataset", "RunLogger", "Workspace", "make_tools"]
__version__ = "0.1.0"
