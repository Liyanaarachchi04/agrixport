import os
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langgraph.graph import StateGraph, END

# 1. Setup Models
# Low-cost/ultra-fast Groq model for Intent Routing
router_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# High-reasoning OpenRouter model for Synthesis & Reflection
synthesis_llm = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

# Load RAG
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Define Agent State Schema
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    query: str
    route: str
    retrieved_docs: str
    draft_response: str
    final_response: str

# AGENT 1: Router Pattern
def router_agent(state: AgentState):
    query = state["query"]
    prompt = f"Categorize this user query into 'REGULATORY_QUERY' or 'GENERAL': {query}"
    res = router_llm.invoke(prompt)
    route = "REGULATORY_QUERY" if "REGULATORY" in res.content.upper() else "GENERAL"
    return {"route": route}

# AGENT 2: Tool-Use / ReAct Search Agent
def rag_research_agent(state: AgentState):
    query = state["query"]
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])
    return {"retrieved_docs": context}

# AGENT 3: Synthesis & Self-Critique / Reflection Pattern
def synthesis_reflection_agent(state: AgentState):
    query = state["query"]
    docs = state.get("retrieved_docs", "No regulatory context fetched.")

    # Step 1: Draft Response
    draft_prompt = f"Answer query: {query}\n\nUsing Context:\n{docs}"
    draft = synthesis_llm.invoke(draft_prompt).content

    # Step 2: Self-Critique / Reflection Pattern
    critique_prompt = f"Critique and verify this response against facts. Ensure no hallucination:\nDraft: {draft}"
    critique = synthesis_llm.invoke(critique_prompt).content

    # Final Output Synthesis
    final_prompt = f"Refine the draft based on critique.\nDraft: {draft}\nCritique: {critique}"
    final_ans = synthesis_llm.invoke(final_prompt).content

    return {"draft_response": draft, "final_response": final_ans}

# Build Agent Graph Flow
workflow = StateGraph(AgentState)
workflow.add_node("router", router_agent)
workflow.add_node("researcher", rag_research_agent)
workflow.add_node("synthesizer", synthesis_reflection_agent)

workflow.set_entry_point("router")
workflow.add_conditional_edges(
    "router",
    lambda state: "researcher" if state["route"] == "REGULATORY_QUERY" else "synthesizer",
    {"researcher": "researcher", "synthesizer": "synthesizer"}
)
workflow.add_edge("researcher", "synthesizer")
workflow.add_edge("synthesizer", END)

agent_app = workflow.compile()