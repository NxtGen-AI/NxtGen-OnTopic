# Query Answering System
A conversational AI system that takes user queries, refines them, retrieves relevant documents, and generates answers using NLP techniques and LangChain framework.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [System Overview](#system-overview)

## Installation
To install the required dependencies, follow these steps:

1. **Install Ollama**: First, ensure that you have [Ollama](https://ollama.com/) installed on your system.
2. **Pull LLaMA Model**: Run `ollama pull llama3.1:8b` to download the required LLaMA model.
3. **Create a Python Environment**: Create a new Python environment by running `python -m venv qa-env` (replace `qa-env` with your desired environment name).
4. **Activate the Environment**: Activate the newly created environment:
   * On Windows, run `qa-env\Scripts\activate` (replace `qa-env`)
   * On macOS/Linux, run `source qa-env/bin/activate` (replace `qa-env`)
5. **Install Python Dependencies**: Install the required Python dependencies by running `pip install -r requirements.txt`

## Usage
To use the question answering system, simply run the main script and follow the prompts.

## System Overview

## 1. Initialize Vector Store (vector_store.py)

### Purpose
Set up a Chroma vector database with document embeddings to enable semantic search.

### Process
1. Use embeddings from either Ollama or OpenAI based on the `LLM_TYPE` environment variable.
2. Populate the database with initial documents from `constants.docs`.
3. Initialize a retriever for fetching relevant documents during query processing.

## 2. Define NLP Tasks (query_workflows.py)

### Purpose
Implement functions to handle each NLP task in the pipeline, including query rewriting, document retrieval, topic classification, reranking, and answer generation.

### Process
1. **rewrite_query()**: Enhance query clarity for better search results.
2. **retrieve_documents()**: Fetch relevant documents from the vector store.
3. **classify_topic()**: Determine if the query is within the expected domain.
4. **rerank_documents()**: Refine document relevance based on context or additional criteria.
5. **generate_answer()**: Produce a response using retrieved and ranked documents.

## System Workflow

### 1. Initialize Vector Store (vector_store.py)

#### Purpose
Set up a Chroma vector database with document embeddings to enable semantic search.

#### Process
1. Use embeddings from either Ollama or OpenAI based on the `LLM_TYPE` environment variable.
2. Populate the database with initial documents from `constants.docs`.
3. Initialize a retriever for fetching relevant documents during query processing.

### 2. Define NLP Tasks (query_workflows.py)

#### Purpose
Implement functions to handle each NLP task in the pipeline, including query rewriting, document retrieval, topic classification, reranking, and answer generation.

#### Process
1. **rewrite_query()**: Enhance query clarity for better search results.
2. **retrieve_documents()**: Fetch relevant documents from the vector store.
3. **classify_topic()**: Determine if the query is within the expected domain.
4. **rerank_documents()**: Refine document relevance based on context or additional criteria.
5. **generate_answer()**: Produce a response using retrieved and ranked documents.

### 3. Set Up Workflow (workflow_setup.py)

#### Purpose
Configure the processing flow as a state machine, defining the sequence of steps from query input to answer generation.

#### Process
1. Define nodes for each task: rewriting, retrieval, classification, reranking, and generation.
2. Link nodes in logical order to create a coherent workflow.

### 4. Initialize and Run Pipeline (main.py)

#### Purpose
Compile the defined workflow into an executable application and process incoming queries.

#### Process
1. Use `setup_workflow()` from `workflow_setup` to initialize the pipeline structure.
2. Compile the workflow into an app for execution.
3. Accept a query as input, execute it through the pipeline, and return the generated answer.

### 5. Log and Monitor (utils.py)

#### Purpose
Utilize logging functions to track processing steps and ensure readability of output for debugging and monitoring.

#### Process
1. Use appropriate logging functions to record each stage's progress.
2. Ensure that outputs are formatted clearly for better understanding.

### Summary

- **Initialization**: Load documents into the vector store, enabling efficient semantic search.
- **Processing Flow**: Define a workflow that sequentially processes queries through rewriting, retrieval, classification, reranking, and answer generation.
- **Execution**: Run each query through the pipeline using the compiled app, with logging to monitor progress.