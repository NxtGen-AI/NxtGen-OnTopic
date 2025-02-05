"""
This module sets up a state graph workflow for handling queries through
various processing stages.

The workflow consists of the following nodes:
- rewriter: Handles the initial rewriting of the query to optimize retrieval.
- retrieve_documents: Fetches relevant documents based on the rewritten query.
- topic_decision: Classifies whether the query is within the defined scope
(on-topic) or outside of it (off-topic).
- off_topic_response: Provides a standardized response when the query is deemed off-topic.
- rerank_documents: Reranks the retrieved documents to improve relevance to the query.
- generate_answer: Generates a final answer based on the most relevant documents.

The workflow follows these connections:
- The rewriter node leads to retrieving documents.
- Retrieved documents are evaluated by the topic decision classifier.
- Depending on the classification, the flow proceeds either to reranking
documents (on-topic) or providing an off-topic response.
- Reranked documents are used to generate a final answer, which then concludes the workflow.

The StateGraph is configured with these nodes and edges to manage the query 
handling process efficiently. The entry point of the workflow is set at the rewriter
node, initiating the sequence of processing steps.

This setup provides a structured approach to query processing, ensuring that each stage
builds upon the previous one to deliver accurate and relevant responses.
"""

from langgraph.graph import StateGraph, END

import agent_state_class
import query_workflows

def setup_workflow() -> StateGraph:
    """
    Sets up a state graph workflow for handling queries through various stages.

    The workflow consists of the following nodes:
    - rewriter: Handles query rewriting.
    - retrieve_documents: Retrieves relevant documents based on the query.
    - topic_decision: Classifies whether the query is on-topic or off-topic.
    - off_topic_response: Provides a response when the query is off-topic.
    - rerank_documents: Reranks the retrieved documents for relevance.
    - generate_answer: Generates an answer based on the reranked documents.

    The workflow follows these connections:
    - rewriter -> retrieve_documents
    - retrieve_documents -> topic_decision
    - topic_decision -> rerank_documents (if on-topic) or off_topic_response (if off-topic)
    - rerank_documents -> generate_answer
    - generate_answer -> END

    Returns:
        StateGraph: The configured state graph with nodes and edges set up for the query handling workflow.
    """
    workflow = StateGraph(agent_state_class.AgentState)
    
    # Add nodes for query rewriting, classification, off-topic response, and document retrieval
    workflow.add_node("rewriter", query_workflows.rewriter)
    workflow.add_node("retrieve_documents", query_workflows.retrieve_documents)
    workflow.add_node("topic_decision", query_workflows.question_classifier)
    workflow.add_node("off_topic_response", query_workflows.off_topic_response)
    workflow.add_node("rerank_documents", query_workflows.rerank_documents)
    workflow.add_node("generate_answer", query_workflows.generate_answer)


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

    return workflow