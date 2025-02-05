"""
This module provides various utilities and configurations for language model operations.

Modules and Imports:
- Uses `enum` for creating enumeration types.
- Utilizes `langchain.schema.Document` to handle document data with associated metadata.

Enumerations:
- `LLMType`: Defines different types of Large Language Models, currently supporting 'ollama'.

Constants:
- MODEL_OLLAMA: Specifies the model name for Ollama ('llama3.1:8b').
- MODEL_OPENAI: Indicates the OpenAI GPT model ('gpt-4o-mini').
- TEMPERATURE_DEFAULT: Sets the default temperature value (0) for model responses.

Message Templates and Prompts:
- SYSTEM_MESSAGE_REWRITER: System prompt for a question re-writer focused on retrieval optimization.
- HUMAN_MESSAGE_TEMPLATE_REWRITER: Template for generating improved questions from initial inputs.
- CLASSIFIER_INTRODUCTION and CLASSIFIER_PROMPT_TEMPLATE: Prompts for determining if questions and 
retrieved documents are on-topic.
- RERANKING_PROMPT_TEMPLATE: Instructions to rank documents based on relevance to a given question.
- ANSWER_TEMPLATE: Template for answering questions using provided context.

Example Documents:
- `docs`: A list of Document instances with sample content related to AI engineering activities, 
including metadata about their source and creation dates.
"""

from enum import Enum

from langchain.schema import Document

class LLMType(str, Enum):
    """
    Defines an enumeration for LLM types
    """
    OLLAMA = "ollama"

# Define constants for model parameters
MODEL_OLLAMA = "llama3.1:8b"
MODEL_OPENAI = "gpt-4o-mini"
TEMPERATURE_DEFAULT = 0

SYSTEM_MESSAGE_REWRITER = """You are a question re-writer that converts an input question to a better version that is optimized 
    for retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""


HUMAN_MESSAGE_TEMPLATE_REWRITER = "Here is the initial question: \n\n {question} \n Formulate an improved question."

# Example documents
docs = [
    Document(
        page_content="EOD REPORT THURSDAY 14 NOVEMBER\nARYAA\n\n- Tested out lab workflow using sample data provided by them and created a small model.\n\n- Created a small knowledge file using our GOA_GST data and used this to train the Granite model to test if it is working as expected -- IT IS -- also resulted in a small model fine-tuned on GOA GST data.\n\n- Now generating a large knowledge file which will contain data from an entire directory of PDFs about CA ACTS, which will be used to train our model.",
        metadata={"source": "AI-Engineering Channel", "created_at": "2024-11-14"},
    ),
    Document(
        page_content="Shilpa\n- Moved workflow to PEFT-oriented strategies\n- Post InstructLab pipeline is in place. Aryaa will cover other workflows within the NeMo framework as her pipeline on InstructLab runs.\n\nSush\n- Produced code analysis report for CSC on their code base\n- Explored Meta Self-evaluator to evaluate AMD 1B language model.",
        metadata={"source": "AI-Engineering Channel", "created_at": "2024-11-15"},
    ),
    Document(
        page_content="EOD Report Friday 15 November\nShilpa\nI have been doing model downloading and model conversion in .nemo format but it is still having an error while conversion. So, I will be solving that error.\n\nSushmender\nI am exploring Meta's Self-taught-evaluator (Llama-70B) model and requested Nvidia GPU for deploying the model.",
        metadata={"source": "AI-Engineering Channel", "created_at": "2024-11-18"},
    ),
    Document(
        page_content="Report 27th Nov\nAryaa\nThe model trained on section 1 is ready.\n\nShilpa\nI completed my pipeline for making the dataset. Now I will move to split my dataset into train, test, and valid to train my model.\n\nSushmender\nI pulled the 'AI-Engineering' channel posts using a GET request, converted them into JSON format, and am using it as a knowledge base for my RAG setup.",
        metadata={"source": "AI-Engineering Channel", "created_at": "2024-11-28"},
    ),
]

# Define constants for the classifier prompt
CLASSIFIER_INTRODUCTION = "You are a classifier that determines if a question and retrieved documents are on-topic."
CLASSIFIER_PROMPT_TEMPLATE = "Question: {question}\n\nDocuments: {documents}\n\nIs this on-topic? Respond with 'on-topic' or 'off-topic'."

RERANKING_PROMPT_TEMPLATE = """Given the question and doccuments , rank the following documents in order of preference (1st, 2nd, 3rd) based on the relevence to the question. 
        Provide your ranking as "1st preference: <complete chunk>", "2nd preference: <complete chunk>", and "3rd preference: <complete chunk>".
        If there are fewer than 3 relevant documents, skip ranking those that are not useful.
        please give the complete chunks .
        
        Question: {question}
        Documents:{documents}"""
        
ANSWER_TEMPLATE = """Answer the question based only on the following context:\n{context}\n\nQuestion: {question}. Dont mention like prefernce and context while generating the answer"""
