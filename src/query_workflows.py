"""
This module provides tools for natural language processing tasks focused on query
rewriting, document retrieval, classification, and answer generation. It leverages
large language models (LLMs) to enhance the effectiveness of these operations.

Key functionalities include:
- Query rewriting to improve clarity or relevance
- Document retrieval based on contextual queries
- Classification of content as 'on-topic' or 'off-topic'
- Generation of answers using retrieved documents

The module contains functions that handle each specific task, working together to 
create a comprehensive NLP pipeline.
"""

from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

import agent_state_class
import utils
import llm_setup
import vector_store
import constants

def rewriter(agent_state: agent_state_class.AgentState) -> agent_state_class.AgentState:
    """
    Rewrites a given question to an optimized version for retrieval.

    Args:
        agent_state (AgentState): The current state of the agent, including the initial question.

    Returns:
        AgentState: The updated agent state with the rewritten question.
    """
    utils.log_with_horizontal_line("Starting query rewriting...")
    question = agent_state["question"]

    prompt = llm_setup.create_prompt(question)
    rewritten_question = rewrite_question(prompt, question)

    agent_state["question"] = rewritten_question

    utils.log_with_horizontal_line(f"Rewritten question:\\n{rewritten_question}")
    return agent_state

def retrieve_documents(agent_state: agent_state_class.AgentState):
    """
    Retrieves relevant documents based on the question in the agent's state.

    :param agent_state: A dictionary containing the current state of the agent, 
    including the question.
    :return: The updated agent state with the top 3 retrieved documents.
    """
    utils.log_with_horizontal_line("Starting document retrieval...")
    question = agent_state["question"]
    documents = vector_store.retriever.invoke(question)

    # Retrieve top 3 docs
    agent_state["top_documents"] = [doc.page_content for doc in documents[:3]]

    utils.log_ordered_list_with_horizontal_break(
        "Retrieved documents: ", 
        agent_state['top_documents']
    )
    return agent_state

def rewrite_question(prompt: ChatPromptTemplate, question: str) -> str:
    """
    Use the LLM to rewrite the question.

    Args:
        prompt (ChatPromptTemplate): The chat prompt template.
        question (str): The initial question to be rewritten.

    Returns:
        str: The rewritten question.
    """
    return llm_setup.process_with_pipeline(prompt, {"question": question})

def on_topic_classifier(response: str) -> str:
    """Determines the classification result based on the LLM's response"""
    return "on-topic" if "on-topic" in response.lower() else "off-topic"

# Define the classifier function
def format_classifier_prompt(question: str, documents: list) -> str:
    """
    Formats the classifier prompt text with the question and documents.

    Args:
        question (str): The question to classify.
        documents (list): The retrieved documents.

    Returns:
        str: The formatted classifier prompt text.
    """
    return constants.CLASSIFIER_PROMPT_TEMPLATE.format(
        question=question,
        documents='\\n'.join(documents)
    )

def create_classifier_prompt(question: str, documents: list) -> ChatPromptTemplate:
    """
    Creates a LLM-based classifier prompt with the given question and documents.

    Args:
        question (str): The question to classify.
        documents (list): The retrieved documents.

    Returns:
        ChatPromptTemplate: A chat prompt template for classification.
    """
    classifier_prompt_text = format_classifier_prompt(question, documents)
    return ChatPromptTemplate.from_messages(
        [
            ("system", constants.CLASSIFIER_INTRODUCTION),
            (
                "human",
                classifier_prompt_text,
            ),
        ]
    )

def question_classifier(agent_state: agent_state_class.AgentState) -> agent_state_class.AgentState:
    """
    Classifies a question and its retrieved documents as on-topic or off-topic using 
    an LLM-based approach.

    Args:
        agent_state (AgentState): The current agent state containing the question and 
        top documents.

    Returns:
        AgentState: The updated agent state with the classification result.
    """

    utils.log_with_horizontal_line("Starting topic classification...")

    # Extract relevant information from the agent state
    question = agent_state["question"]
    documents = agent_state["top_documents"]
    input_data = {"question": question, "documents": "\\\\n".join(documents)}

    # Create the LLM-based classifier prompt
    classifier_prompt = create_classifier_prompt(question, documents)

    # Use a pipeline for classification
    raw_classification_result = llm_setup.process_with_pipeline(classifier_prompt, input_data)
    classification_result = on_topic_classifier(raw_classification_result)

    # Update the agent state with the classification result
    agent_state["classification_result"] = classification_result

    utils.log_with_horizontal_line(f"Classification result: {agent_state['classification_result']}")
    return agent_state

def off_topic_response(agent_state: agent_state_class.AgentState) -> agent_state_class.AgentState:
    """Handle off-topic questions by logging and updating agent state."""
    utils.log_with_horizontal_line("Question is off-topic. Ending process.")
    agent_state["llm_output"] = "The question is off-topic, ending the process."
    return agent_state

def rerank_documents(agent_state: agent_state_class.AgentState) -> agent_state_class.AgentState:
    """Rerank top documents based on preference order using LLM."""
    utils.log_with_horizontal_line("Starting document reranking with preferences...")

    # Extract necessary data from agent state
    question = agent_state["question"]
    top_documents = agent_state["top_documents"]

    # Define the prompt template for document reranking
    reranker_prompt = PromptTemplate(
        input_variables=[],
        template = constants.RERANKING_PROMPT_TEMPLATE.format(
            question=question, 
            documents=top_documents
        )
    )

    # Process the reranking prompt with the LLM
    ranking_result = llm_setup.process_with_pipeline(reranker_prompt, {
        "question": question,
        "documents": top_documents
    })

    utils.log_with_horizontal_line(f"Ranking result: {ranking_result}")

    # Update the agent state with the raw ranking result
    agent_state["top_documents"] = [ranking_result.strip()]

    return agent_state

def generate_answer(agent_state: agent_state_class.AgentState) -> agent_state_class.AgentState:
    """Generates an answer to the agent's question based on provided context.
    
    Args:
        agent_state: An instance of AgentState containing the question and relevant documents.
        
    Returns:
        The updated AgentState with the generated answer.
        
    Side Effects:
        Logs the generation process and the resulting answer.
    """
    utils.log_with_horizontal_line("Generating answer...")

    question = agent_state["question"]
    context = "\\n".join(agent_state["top_documents"])

    prompt = ChatPromptTemplate.from_template(constants.ANSWER_TEMPLATE)

    answer = llm_setup.process_with_pipeline(
        prompt,
        {"question": question, "context": context}
    )

    agent_state["llm_output"] = answer
    utils.log_with_horizontal_line(f"Generated answer: {answer}")

    return agent_state
