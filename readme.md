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
The system consists of several stages:

1. **Query Rewriting**: Refines the user's query to improve its clarity and relevance.
2. **Document Retrieval**: Retrieves relevant documents from a corpus based on the rewritten query.
3. **Topic Classification**: Determines whether the retrieved documents are on-topic or off-topic with respect to the original query.
4. **Reranking Documents**: Ranks the retrieved documents in order of preference based on their relevance to the query.
5. **Answer Generation**: Generates an answer to the user's query based on the top-ranked documents.

