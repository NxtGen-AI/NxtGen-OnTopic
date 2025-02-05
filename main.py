import os

from enum import Enum
from typing import List
from typing_extensions import TypedDict

# langchain imports
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate

# langgraph imports
from langgraph.graph import StateGraph, END

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

def log_with_horizontal_line(message):
    """Function to print logs in a structured and readable format"""
    # Get the terminal width
    terminal_width = os.get_terminal_size().columns

    # Create the horizontal line
    horizontal_line = '-' * terminal_width

    # Print the formatted message
    print(horizontal_line)
    print()
    print(message)
    print()
    print(horizontal_line)

def log_ordered_list_with_horizontal_break(message: str, collection: list):
    """Function to print ordered lists in a structured and readable format"""
    # Get the terminal width
    terminal_width = os.get_terminal_size().columns

    # Create the horizontal line and indent
    horizontal_line = '-' * terminal_width

    # Print the formatted message
    print(horizontal_line)
    print()
    print(message)
    for i, item in enumerate(collection, 1):
        lines = item.split("\\n")
        print(f"{i}. {lines[0]}")  # Print the number and first line together
        for line in lines[1:]:       # Indent subsequent lines
            print(f"    {line}")
        print()
    print()
    print(horizontal_line)

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

    llm_type = os.getenv("LLM_TYPE", LLMType.OLLAMA.value)

    if llm_type == LLMType.OLLAMA.value:
        return ChatOllama(model = MODEL_OLLAMA, temperature = TEMPERATURE_DEFAULT)

    return ChatOpenAI(model = MODEL_OPENAI, temperature = TEMPERATURE_DEFAULT)

def get_embeddings():
    """Function to get embedding instance based on environment variable."""

    embedding_type = os.getenv("LLM_TYPE", LLMType.OLLAMA.value)

    if embedding_type == LLMType.OLLAMA.value:
        return OllamaEmbeddings(model = MODEL_OLLAMA)

    return OpenAIEmbeddings()

# Embedding function and documents
embedding_function = get_embeddings()

# Initialize Chroma vector store
db = Chroma.from_documents(docs, embedding_function)
retriever = db.as_retriever()

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

def create_prompt(question: str) -> ChatPromptTemplate:
    """
    Create a chat prompt template for the LLM.

    Args:
        question (str): The initial question to be rewritten.

    Returns:
        ChatPromptTemplate: The chat prompt template.
    """
    human_message = HUMAN_MESSAGE_TEMPLATE_REWRITER.format(question=question)
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_MESSAGE_REWRITER),
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

def rewrite_question(prompt: ChatPromptTemplate, question: str) -> str:
    """
    Use the LLM to rewrite the question.

    Args:
        prompt (ChatPromptTemplate): The chat prompt template.
        question (str): The initial question to be rewritten.

    Returns:
        str: The rewritten question.
    """
    return process_with_pipeline(prompt, {"question": question})

def rewriter(agent_state: AgentState) -> AgentState:
    """
    Rewrites a given question to an optimized version for retrieval.

    Args:
        agent_state (AgentState): The current state of the agent, including the initial question.

    Returns:
        AgentState: The updated agent state with the rewritten question.
    """
    log_with_horizontal_line("Starting query rewriting...")
    question = agent_state["question"]

    prompt = create_prompt(question)
    rewritten_question = rewrite_question(prompt, question)

    agent_state["question"] = rewritten_question

    log_with_horizontal_line(f"Rewritten question:\\n{rewritten_question}")
    return agent_state

# Function to retrieve documents
def retrieve_documents(agent_state: AgentState):
    """
    Retrieves relevant documents based on the question in the agent's state.

    :param agent_state: A dictionary containing the current state of the agent, 
    including the question.
    :return: The updated agent state with the top 3 retrieved documents.
    """
    log_with_horizontal_line("Starting document retrieval...")
    question = agent_state["question"]
    documents = retriever.invoke(question)

    # Retrieve top 3 docs
    agent_state["top_documents"] = [doc.page_content for doc in documents[:3]]

    log_ordered_list_with_horizontal_break("Retrieved documents: ", agent_state['top_documents'])
    return agent_state

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
    return CLASSIFIER_PROMPT_TEMPLATE.format(question=question, documents='\\n'.join(documents))

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
            ("system", CLASSIFIER_INTRODUCTION),
            (
                "human",
                classifier_prompt_text,
            ),
        ]
    )

def question_classifier(agent_state: AgentState) -> AgentState:
    """
    Classifies a question and its retrieved documents as on-topic or off-topic using 
    an LLM-based approach.

    Args:
        agent_state (AgentState): The current agent state containing the question and 
        top documents.

    Returns:
        AgentState: The updated agent state with the classification result.
    """

    log_with_horizontal_line("Starting topic classification...")

    # Extract relevant information from the agent state
    question = agent_state["question"]
    documents = agent_state["top_documents"]
    input_data = {"question": question, "documents": "\\\\n".join(documents)}

    # Create the LLM-based classifier prompt
    classifier_prompt = create_classifier_prompt(question, documents)

    # Use a pipeline for classification
    raw_classification_result = process_with_pipeline(classifier_prompt, input_data)
    classification_result = on_topic_classifier(raw_classification_result)

    # Update the agent state with the classification result
    agent_state["classification_result"] = classification_result

    log_with_horizontal_line(f"Classification result: {agent_state['classification_result']}")
    return agent_state

def off_topic_response(agent_state: AgentState) -> AgentState:
    """Handle off-topic questions by logging and updating agent state."""
    log_with_horizontal_line("Question is off-topic. Ending process.")
    agent_state["llm_output"] = "The question is off-topic, ending the process."
    return agent_state

def rerank_documents(agent_state: AgentState) -> AgentState:
    """Rerank top documents based on preference order using LLM."""
    log_with_horizontal_line("Starting document reranking with preferences...")

    # Extract necessary data from agent state
    question = agent_state["question"]
    top_documents = agent_state["top_documents"]

    # Define the prompt template for document reranking
    reranker_prompt = PromptTemplate(
        input_variables=[],
        template=RERANKING_PROMPT_TEMPLATE.format(question=question, documents=top_documents)
    )

    # Process the reranking prompt with the LLM
    ranking_result = process_with_pipeline(reranker_prompt, {
        "question": question,
        "documents": top_documents
    })

    log_with_horizontal_line(f"Ranking result: {ranking_result}")

    # Update the agent state with the raw ranking result
    agent_state["top_documents"] = [ranking_result.strip()]

    return agent_state

def generate_answer(agent_state: AgentState) -> AgentState:
    """Generates an answer to the agent's question based on provided context.
    
    Args:
        agent_state: An instance of AgentState containing the question and relevant documents.
        
    Returns:
        The updated AgentState with the generated answer.
        
    Side Effects:
        Logs the generation process and the resulting answer.
    """
    log_with_horizontal_line("Generating answer...")

    question = agent_state["question"]
    context = "\\n".join(agent_state["top_documents"])

    prompt = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)

    answer = process_with_pipeline(
        prompt,
        {"question": question, "context": context}
    )

    agent_state["llm_output"] = answer
    log_with_horizontal_line(f"Generated answer: {answer}")

    return agent_state

# Updated workflow
workflow = StateGraph(AgentState)

# Add nodes for query rewriting, classification, off-topic response, and document retrieval
workflow.add_node("rewriter", rewriter)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("topic_decision", question_classifier)
workflow.add_node("off_topic_response", off_topic_response)
workflow.add_node("rerank_documents", rerank_documents)
workflow.add_node("generate_answer", generate_answer)

# Add conditional edges for topic decision after retrieval
workflow.add_conditional_edges(
    "topic_decision",
    lambda state: "on-topic" if state["classification_result"] == "on-topic" else "off-topic",
    {
        "on-topic": "rerank_documents",
        "off-topic": "off_topic_response",
    },
)

# Add edges for reranking and answer generation
workflow.add_edge("rewriter", "retrieve_documents")
workflow.add_edge("retrieve_documents", "topic_decision")
workflow.add_edge("rerank_documents", "generate_answer")
workflow.add_edge("generate_answer", END)

# Set entry point to query rewriting
workflow.set_entry_point("rewriter")

# Compile app
app = workflow.compile()


# Example invocation
if __name__ == "__main__":
    state = {"question": "What sushmender is working on?"}
    result = app.invoke(state)
    log_with_horizontal_line(f"Final answer: {result.get('llm_output', 'Process ended.')}")
