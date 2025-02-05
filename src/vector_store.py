"""
Contains initialization of the vector store using Chroma.

This module sets up a vector store with Chroma, utilizing either OllamaEmbeddings
or OpenAIEmbeddings based on the LLM_TYPE environment variable. 
If LLM_TYPE is set to 'ollama', it uses OllamaEmbeddings with the model specified
in constants.MODEL_OLLAMA; otherwise, it defaults to OpenAIEmbeddings.

Key components:
- get_embeddings(): Function to determine and return the appropriate embeddings instance.
- embedding_function: The initialized embeddings function based on LLM_TYPE.
- db: The Chroma vector store initialized with documents from constants.docs.
- retriever: A retriever object created from the Chroma database.
"""

import os

# langchain imports
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

import constants

def get_embeddings():
    """Function to get embedding instance based on environment variable."""

    embedding_type = os.getenv("LLM_TYPE", constants.LLMType.OLLAMA.value)

    if embedding_type == constants.LLMType.OLLAMA.value:
        return OllamaEmbeddings(model = constants.MODEL_OLLAMA)

    return OpenAIEmbeddings()

# Embedding function and documents
embedding_function = get_embeddings()

# Initialize Chroma vector store
db = Chroma.from_documents(constants.docs, embedding_function)
retriever = db.as_retriever()
