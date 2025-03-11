import streamlit as st
import requests
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# ------------------------- Pydantic Response Models -------------------------

class AddressResponse(BaseModel):
    address: str = Field(..., description="The official address or location information.")
    notes: Optional[str] = Field(None, description="Any additional notes about the address.")

class FAQResponse(BaseModel):
    answer: str = Field(..., description="The answer to the frequently asked question.")
    related_topics: Optional[List[str]] = Field(None, description="List of related FAQ topics.")

# ------------------------- Sample Knowledge Base -------------------------

ADDRESS_KB = """
Our main office is located at 1234 Elm Street, Springfield, USA.
We are open from Monday to Friday, 9 AM to 5 PM.
"""

FAQ_KB = """
1. What is your return policy? You can return items within 30 days with a valid receipt.
2. Do you ship internationally? Yes, we ship worldwide with additional shipping charges.
"""

# ------------------------- LangChain Style Agent Prompts -------------------------

AGENTS = {
    "address": {
        "system_prompt": "You are a customer support agent specialized in handling address and location-related queries. Use the knowledge base to give precise answers.",
        "knowledge_base": ADDRESS_KB,
        "response_model": AddressResponse
    },
    "faq": {
        "system_prompt": "You are a customer support agent answering frequently asked questions (FAQ). Respond based on the provided knowledge base.",
        "knowledge_base": FAQ_KB,
        "response_model": FAQResponse
    }
}

# ------------------------- Intent Detection -------------------------

def detect_intent(message: str) -> str:
    """Simple intent detection based on keywords (can be improved using AI classification)."""
    address_keywords = ["address", "location", "where", "located"]
    faq_keywords = ["return", "refund", "ship", "policy", "international"]

    if any(word in message.lower() for word in address_keywords):
        return "address"
    elif any(word in message.lower() for word in faq_keywords):
        return "faq"
    else:
        return "faq"  # Default fallback

# ------------------------- Query Ollama with LangChain System -------------------------

def query_ollama_langchain(message: str, agent_key: str, ollama_url, model_name, temperature, max_tokens):
    agent = AGENTS[agent_key]
    system_prompt = agent["system_prompt"] + "\n\nKnowledge Base:\n" + agent["knowledge_base"]

    # Prepare chat messages in LangChain style
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message}
    ]

    try:
        response = requests.post(
            ollama_url,
            json={
                "model": model_name,
                "messages": messages,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                },
                "stream": False
            },
            timeout=120
        )
        if response.status_code == 200:
            return response.json()["message"]["content"]
        else:
            return f"Error: Received status code {response.status_code} from Ollama API."
    except Exception as e:
        return f"Error: {str(e)}"

# ------------------------- Streamlit App Setup -------------------------

# Set page config
st.set_page_config(page_title="Multi-Agent Customer Support", page_icon="🤖", layout="wide")
st.title("Multi-Agent Customer Support 🤖")
st.caption("Powered by Llama3.2 and Ollama - Dynamic Agent Routing")

# Session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("Model Settings")
    ollama_url = st.text_input("Ollama API URL", value="http://localhost:11434/api/chat")
    model_name = st.selectbox("Model", ["llama3.2", "llama3.2:70b"], index=0)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.slider("Max Tokens", 64, 4096, 1024, 64)
    st.divider()
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.container():
        role = "👤" if message["role"] == "user" else "🤖"
        st.markdown(f"""
        <div class="chat-message {message['role']}">
            <div class="avatar">{role}</div>
            <div class="message">{message['content']}</div>
        </div>
        """, unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Type your message here...")

# ------------------------- Main Chat Handling -------------------------

if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Detect intent
    intent = detect_intent(user_input)

    # Query the selected agent
    with st.spinner(f"Routing to {intent.capitalize()} agent..."):
        response = query_ollama_langchain(
            user_input, intent, ollama_url, model_name, temperature, max_tokens
        )

    # Append assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Rerun to refresh chat UI
    st.rerun()

# ------------------------- Footer Instructions -------------------------

st.divider()
st.markdown("""
### Instructions:
1. Make sure Ollama is running locally and the Llama3.2 model is available.
2. The chatbot dynamically routes queries to specialized agents based on intent.
3. Adjust temperature and max tokens in the sidebar for better control over responses.
4. Start by asking for an address or any FAQ like return policy.
""")
