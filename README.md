# Rust LLM Core

Python is slow, and that won't scale well for intensive LLM first apps. 

To solve this, and allow for useability across platforms, I've written; a Rust core for interacting with LLMs from various providers.
This core doesn't simply make API calls, it allows a plethora of necessary functionality for building agentic application & services.

- Native tool library
- Structured Response
- Data Sorter Feature
- Major LLM Providers
- Async swarm calling
- Token Usage Tracker

## Installation

See `test.py` for examples of how to use the core in Python. Make sure you have conda installed. The system will not install otherwise. Windows only.

To install the core, run:
```bash
cd llm-core
python llm_core/__rsdl__.py -i
```

Then you can run:
```bash
conda activate llm_core_venv
python test.py
```

When building features in the core,
```bash
conda deactivate
python llm_core/__rsdl__.py -r
<<<<<<< HEAD
```

---

Full Command-Chain:
```bash
cd llm-core
python llm_core/__rsdl__.py -i
conda activate llm_core_venv
python test.py
conda deactivate
python llm_core/__rsdl__.py -r
=======
>>>>>>> 12a7a9d (Updated README.md)
```