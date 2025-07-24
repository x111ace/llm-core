"
The goal is to build the architectural skeleton first, test it end-to-end with placeholder logic, and then circle back to implement the complex RAG mechanics. 

This ensures the integration points are solid before we tackle embeddings and vector search. Here is the proposed task list for the `Retrieval` feature implementation:

*   Research the best way to implement RAG in Rust, using SQLite as the database.
    *   Use markitdown, docling and crawl4ai to collect the documents to be saved in markdown format
        *   Look into ways to use vision AI to perform OCR on images and PDFs, to convert them to markdown
    *   Design a multi-database architecture where each knowledge domain (books, docs, etc.) gets its own DB
        *   Use a fine tuned, or specially prompted embedding model for each database type
        *   Store documents with docID, title, source_type (book/webpage/etc), and metadata
            *   Implement chunking with chnkID that includes source DB identifier for traceability
        *   With the embedding model's vectorization, we can store the documents and chunks relationally to each other, respectively.
            *   This will allow for the retrieval of either the most relevant content, or the most relevant titles, or both.
    *   Follow existing GitHub repos that utilize RAG with SQLite, and use them as a reference for the implementation.
        *   https://github.com/

### Phase 1: Core Rust Implementation & Standalone Test

1.  **Create `llm-core/llm_core/core/src/retrieval.rs`**:
    *   Define the core data structures:
        *   `RetrievalInstructions`: To hold the user's query, the path to the knowledge source (e.g., a text file or directory), and other parameters.
        *   `RetrievalResult`: To hold the retrieved context chunks.
    *   Define the main `Retrieval` struct, which will manage the process.
    *   Implement a public async function `run_retrieval_task`. Initially, this will contain **placeholder logic** that reads a file and returns its content as a 'retrieved chunk' without any real embedding or search.

2.  **Register the Module in `lib.rs`**:
    *   Add `pub mod retrieval;` to `llm-core/llm_core/core/src/lib.rs` to make it a part of the crate.

3.  **Create the Standalone Test in `full_test.rs`**:
    *   Add a new test `#[tokio::test] async fn test_retrieval_mode()`.
    *   This test will create `RetrievalInstructions` pointing to a temporary file with sample text.
    *   It will call our placeholder `Retrieval::run_retrieval_task`.
    *   It will assert that the `RetrievalResult` contains the expected text from the sample file, confirming the basic file I/O and data flow work correctly in Rust.

### Phase 2: Python Bindings & Testing

4.  **Implement PyO3 Bindings in `python_b.rs`**:
    *   Create a `PyRetrievalInstructions` class to expose the instructions to Python.
    *   Create a `#[pyfunction]` named `run_retrieval`. This function will handle the call from Python, manage the `tokio` runtime, and invoke the core Rust `run_retrieval_task`.
    *   It will convert the Rust `RetrievalResult` back into a Python object (e.g., a dictionary).

5.  **Register Bindings in `lib.rs`**:
    *   Add the new `PyRetrievalInstructions` class and the `run_retrieval` function to the `_llm_core` pymodule.

6.  **Verify with a Python Script**:
    *   Update `test.py` with a new function to call `run_retrieval` to ensure the Python-to-Rust bridge is fully functional.

### Phase 3: Native Tool Integration

7.  **Create the Tool Wrapper in `config/toolkit.rs`**:
    *   Implement a new synchronous Rust function `retrieval_tool` that parses JSON arguments.
    *   This function will create its own `tokio` runtime to call the async `run_retrieval_task`.
    *   It will wrap the result in a `JsonValue` for the `Orchestra` to consume.

8.  **Register the Native Tool**:
    *   In `get_rust_tool_library()`, define the `ToolDefinition` for `retrieval_tool`, specifying its name, description, and parameters.
    *   Add the new tool to the `ToolLibrary`.

9.  **Create Chat Integration Test in `full_test.rs`**:
    *   Add a final test, `#[tokio::test] async fn test_retrieval_in_chat()`.
    *   This will initialize a `Chat` session with native tools.
    *   It will send a prompt designed to trigger the `retrieval_tool`.
    *   It will assert that the model's final response correctly synthesizes the information returned by our placeholder tool.
"