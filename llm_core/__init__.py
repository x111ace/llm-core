"""
llm-core

A high-performance, stateful, and structured conversation engine powered by Rust.
"""

# Expose the core Rust-backed classes and functions at the top level of the package.
from _llm_core import (
    Chat,
    Message,
    run_sorter,
    SchemaItems,
    SchemaProperty,
    SimpleSchema,
    SortingInstructions,
    Tool,
    ToolDefinition,
)

# Define what gets imported with a `from llm_core import *`
__all__ = [
    "Chat",
    "Message",
    "run_sorter",
    "SchemaItems",
    "SchemaProperty",
    "SimpleSchema",
    "SortingInstructions",
    "Tool",
    "ToolDefinition",
]
