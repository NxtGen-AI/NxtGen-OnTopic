"""
This module provides functionality for interacting with different Large Language Models (LLMs) 
using LangChain.

Imports:
- os: For handling environment variables.
- langchain_ollama.ChatOllama: Ollama LLM integration.
- langchain_openai.ChatOpenAI: OpenAI GPT model integration.
- langchain_core.prompts.ChatPromptTemplate: For creating chat prompts.
- langchain_core.output_parsers.StrOutputParser: For parsing string outputs.
- constants: Contains necessary constant values used in this module.

Functions:
1. get_llm(): Retrieves an LLM instance based on the 'LLM_TYPE' environment variable, defaulting
to Ollama if not specified or unrecognized.
2. create_prompt(question): Creates a chat prompt template from the provided question.
3. get_pipeline(prompt): Sets up a processing pipeline with the specified prompt, an appropriate
LLM, and a string output parser.
4. process_with_pipeline(prompt, inputs): Executes the pipeline with the given inputs, returning
the processed result.

Purpose:
This module abstracts away the complexities of selecting and configuring different LLMs via 
LangChain, allowing users to focus on defining prompts and processing inputs without worrying
about the underlying model specifics. It provides a flexible framework for integrating various
LLMs into applications by leveraging environment variables for model selection.
"""

import os

# langchain imports
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import constants

def get_llm():
    """
    Retrieves an instance of a Large Language Model (LLM) based on the environment 
    variable 'LLM_TYPE'.

    The function checks the value of the 'LLM_TYPE' environment variable and returns 
    an instance of the corresponding LLM. If the environment variable is not set or 
    has an unexpected value, it defaults to Ollama.

    Returns:
        An instance of either ChatOllama or ChatOpenAI.
    """

    llm_type = os.getenv("LLM_TYPE", constants.LLMType.OLLAMA.value)

    if llm_type == constants.LLMType.OLLAMA.value:
        return ChatOllama(
            model = constants.MODEL_OLLAMA, 
            temperature = constants.TEMPERATURE_DEFAULT
        )

    return ChatOpenAI(model = constants.MODEL_OPENAI, temperature = constants.TEMPERATURE_DEFAULT)

def create_prompt(question: str) -> ChatPromptTemplate:
    """
    Create a chat prompt template for the LLM.

    Args:
        question (str): The initial question to be rewritten.

    Returns:
        ChatPromptTemplate: The chat prompt template.
    """
    human_message = constants.HUMAN_MESSAGE_TEMPLATE_REWRITER.format(question=question)
    return ChatPromptTemplate.from_messages(
        [
            ("system", constants.SYSTEM_MESSAGE_REWRITER),
            ("human", human_message),
        ]
    )

def get_pipeline(prompt):
    """
    Creates a pipeline consisting of the given prompt, a large language model (LLM), 
    and a string output parser.

    Args:
        prompt: The input prompt to be used in the pipeline.

    Returns:
        A pipeline object that can be used for processing inputs.
    """
    llm = get_llm()
    return prompt | llm | StrOutputParser()

def process_with_pipeline(prompt, inputs):
    """
    Get a pipeline based on the given prompt and invoke it with the specified inputs.

    Parameters:
    - prompt (str): The prompt used to determine which pipeline to get.
    - inputs (dict): A dictionary containing the input data for the pipeline invocation.

    Returns:
    - pipeline_result: The outcome of invoking the pipeline with the provided inputs.
    """
    chain = get_pipeline(prompt)
    pipeline_result = chain.invoke(inputs)
    return pipeline_result
