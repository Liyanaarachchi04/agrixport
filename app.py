import streamlit as st
import os

# Load API Keys securely from Streamlit Secrets or Environment
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
if "OPENROUTER_API_KEY" in st.secrets:
    os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]
    
from agent_system import agent_app

st.set_page_config(page_title="SL AgriExport AI Agent", page_icon="🍃", layout="wide")
st.title("🍃 Sri Lanka AgriExport Compliance Agent")

# Session Chat State
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_query = st.chat_input("Ask about SL export standards, MRL rules, or certification process...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.status("Agents collaborating...", expanded=True) as status:
            st.write("🔍 Router Agent analyzing intent...")
            inputs = {"query": user_query}
            result = agent_app.invoke(inputs)

            st.write(f"➡️ Route selected: `{result.get('route')}`")
            if result.get("retrieved_docs"):
                st.write("📚 Research Agent retrieved matching regulatory chunks.")
            st.write("✍️ Synthesizer Agent performing draft creation & self-critique...")
            status.update(label="Complete!", state="complete")

        response = result["final_response"]
        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
