"""
This module represents the state of an agent through the AgentState data structure, 
capturing essential information such as the question posed, top relevant documents, 
output from a language model, and classification results.
"""

from typing import TypedDict, List

class AgentState(TypedDict):
    """
    Represents the state of an agent.

    Attributes:
        question (str): The question posed to the agent.
        top_documents (List[str]): A list of top relevant documents.
        llm_output (str): The output from the language model.
        classification_result (str): The result of the classification.
    """
    question: str
    top_documents: List[str]
    llm_output: str
    classification_result: str
