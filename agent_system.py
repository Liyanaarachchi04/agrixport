import os
from typing import TypedDict, Sequence
import streamlit as st
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langgraph.graph import StateGraph, END


# ==========================================
# 1. HELPER FUNCTIONS FOR LLMS & VECTORSTORE
# ==========================================

def get_groq_key() -> str:
    """Safely retrieves the Groq API key from environment or Streamlit secrets."""
    key = os.getenv("GROQ_API_KEY")
    if not key and "GROQ_API_KEY" in st.secrets:
        key = st.secrets["GROQ_API_KEY"]
        os.environ["GROQ_API_KEY"] = key
    return key


def get_openrouter_key() -> str:
    """Safely retrieves the OpenRouter API key from environment or Streamlit secrets."""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key and "OPENROUTER_API_KEY" in st.secrets:
        key = st.secrets["OPENROUTER_API_KEY"]
        os.environ["OPENROUTER_API_KEY"] = key
    return key


def get_router_llm():
    """Dynamically instantiates the Groq router model."""
    groq_key = get_groq_key()
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=groq_key
    )


def get_synthesis_llm():
    """Dynamically instantiates the Groq 70B reasoning model for synthesis."""
    groq_key = get_groq_key()
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        groq_api_key=groq_key
    )


def get_retriever():
    """Loads vectorstore retriever using FastEmbed embeddings."""
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})


# ==========================================
# 2. STATE DEFINITION
# ==========================================

class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    query: str
    route: str
    retrieved_docs: str
    draft_response: str
    final_response: str


# ==========================================
# 3. AGENT NODE DEFINITIONS
# ==========================================

# AGENT 1: Router Pattern
def router_agent(state: AgentState):
    query = state["query"]
    prompt = f"Categorize this user query into 'REGULATORY_QUERY' or 'GENERAL': {query}"
    
    # Initialize model dynamically
    router_llm = get_router_llm()
    res = router_llm.invoke(prompt)
    
    route = "REGULATORY_QUERY" if "REGULATORY" in res.content.upper() else "GENERAL"
    return {"route": route}


# AGENT 2: Tool-Use / ReAct Search Agent
def rag_research_agent(state: AgentState):
    query = state["query"]
    retriever = get_retriever()
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])
    return {"retrieved_docs": context}


# AGENT 3: Optimized Synthesis & Self-Critique / Reflection Pattern
def synthesis_reflection_agent(state: AgentState):
    query = state["query"]
    docs = state.get("retrieved_docs", "No regulatory context fetched.")
    
    synthesis_llm = get_synthesis_llm()

    # Single-pass prompt combining drafting, critique, and refinement
    combined_prompt = f"""
    You are an expert Sri Lanka AgriExport AI Advisor.
    User Query: {query}
    
    Retrieved Context:
    {docs}
    
    Task:
    1. Draft a clear answer based ONLY on the context above.
    2. Critique your draft internally to eliminate hallucinations or unverified claims.
    3. Output ONLY the verified, final response for the exporter.
    """
    
    res = synthesis_llm.invoke(combined_prompt)
    
    return {
        "draft_response": res.content, 
        "final_response": res.content
    }


# ==========================================
# 4. BUILD AGENT GRAPH FLOW
# ==========================================

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("router", router_agent)
workflow.add_node("researcher", rag_research_agent)
workflow.add_node("synthesizer", synthesis_reflection_agent)

# Define Entry & Conditional Edges
workflow.set_entry_point("router")
workflow.add_conditional_edges(
    "router",
    lambda state: "researcher" if state["route"] == "REGULATORY_QUERY" else "synthesizer",
    {"researcher": "researcher", "synthesizer": "synthesizer"}
)
workflow.add_edge("researcher", "synthesizer")
workflow.add_edge("synthesizer", END)

# Compile Graph
agent_app = workflow.compile()
