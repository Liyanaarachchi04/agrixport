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
    """Dynamically instantiates the OpenRouter reasoning model."""
    openrouter_key = get_openrouter_key()
    return ChatOpenAI(
        model="meta-llama/llama-3.3-70b-instruct:free",
        api_key=openrouter_key,
        base_url="https://openrouter.ai/api/v1"
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


# AGENT 3: Synthesis & Self-Critique / Reflection Pattern
def synthesis_reflection_agent(state: AgentState):
    query = state["query"]
    docs = state.get("retrieved_docs", "No regulatory context fetched.")
    
    # Initialize model dynamically
    synthesis_llm = get_synthesis_llm()

    # Step 1: Draft Response
    draft_prompt = f"Answer query: {query}\n\nUsing Context:\n{docs}"
    draft = synthesis_llm.invoke(draft_prompt).content

    # Step 2: Self-Critique / Reflection Pattern
    critique_prompt = f"Critique and verify this response against facts. Ensure no hallucination:\nDraft: {draft}"
    critique = synthesis_llm.invoke(critique_prompt).content

    # Step 3: Final Refined Synthesis
    final_prompt = f"Refine the draft based on critique.\nDraft: {draft}\nCritique: {critique}"
    final_ans = synthesis_llm.invoke(final_prompt).content

    return {"draft_response": draft, "final_response": final_ans}


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
