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

SYSTEM_MESSAGE_REWRITER = """You are a question re-writer that converts an input question to a better version that is optimized for retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""


HUMAN_MESSAGE_TEMPLATE_REWRITER = "Here is the initial question: \n\n {question} \n Formulate an improved question."

# Example documents
docs = [
    Document(
        page_content="Project Update Wednesday 22 August\\nJohn Doe\\n\\n- Researched new marketing strategies using industry trends and created a presentation.\\n\\n- Developed a social media campaign to increase brand awareness and engagement -- it's showing promising results -- also resulted in a significant increase in followers.\\n\\n- Now planning a product launch event which will feature our new line of eco-friendly products.",
        metadata={"source": "Sales Team Channel", "created_at": "2024-08-22"},
    ),
    Document(
        page_content="Jane Smith\\n- Moved sales strategy to customer-centric approach\\n- Post-launch pipeline is in place. John Doe will cover other marketing channels within the sales framework as his campaign on social media runs.\\n\\nMichael Brown\\n- Produced market analysis report for competitor company on their product line\\n- Explored new technologies to evaluate and improve our customer service.",
        metadata={"source": "Sales Team Channel", "created_at": "2024-08-23"},
    ),
    Document(
        page_content="Daily Report Thursday 23 August\\nJane Smith\\nI have been working on product positioning and rebranding but it is still having an issue with consistency. So, I will be resolving that issue.\\n\\nMichael Brown\\nI am exploring new sales tools (CRM software) and requested a demo for our team to evaluate its effectiveness.",
        metadata={"source": "Sales Team Channel", "created_at": "2024-08-24"},
    ),
    Document(
        page_content="Weekly Report 30th August\\nJohn Doe\\nThe marketing campaign for our new product is ready.\\n\\nJane Smith\\nI completed my research on customer preferences. Now I will move to create a survey to gather more data and train our sales team.\\n\\nMichael Brown\\nI pulled the 'Sales Team' channel posts using a GET request, converted them into JSON format, and am using it as a knowledge base for our sales setup.",
        metadata={"source": "Sales Team Channel", "created_at": "2024-08-30"},
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
