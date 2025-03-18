import streamlit as st
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import re
import json

# Define the data models using Pydantic
class Intent(BaseModel):
    intent_type: Literal["help", "info", "location", "faq", "goodbye", "unknown"] = Field(description="The type of intent detected in the user message")
    confidence: float = Field(description="Confidence score between 0 and 1")

class UserInfo(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None

class Response(BaseModel):
    message: str = Field(description="Response message to show to the user")
    follow_up_question: Optional[str] = None
    is_farewell: bool = False

# Initialize Ollama with Llama 3.2 model
def initialize_llm():
    return Ollama(model="llama3.2")

# General purpose agent
def create_general_agent(llm):
    template = """
    You are a helpful general-purpose customer support assistant named ChatBot. Your job is to:
    1. Greet users warmly
    2. Collect basic user information (name and email)
    3. Handle general queries
    4. Route specific queries to specialized agents
    5. Say goodbye politely

    User message: {user_message}
    Current user info: {user_info}
    Conversation history: {conversation_history}

    Respond with a JSON object in the following format:
    {{"message": "Your response to the user", "follow_up_question": "Optional follow-up question", "is_farewell": false}}

    If the user is saying goodbye, set "is_farewell" to true.
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["user_message", "user_info", "conversation_history"]
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

# Intent recognition agent
def create_intent_agent(llm):
    template = """
    You are an intent classification system. Given a user message, determine the most likely intent.
    
    User message: {user_message}
    
    Return a JSON object with the following format:
    {{"intent_type": "<INTENT>", "confidence": <CONFIDENCE>}}
    
    Where <INTENT> is one of: "help", "info", "location", "faq", "goodbye", "unknown"
    And <CONFIDENCE> is a number between 0 and 1 indicating your confidence.
    
    Important: Return ONLY the JSON object and nothing else.
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["user_message"]
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

# Location help agent
def create_location_agent(llm):
    template = """
    You are a location assistance specialist. Your job is to help customers with location-related queries.
    
    User message: {user_message}
    
    Knowledge base:
    - Our headquarters is at 123 Main Street, New York, NY 10001
    - We have branches in Los Angeles, Chicago, Miami, and Seattle
    - Our European office is in London at 45 Oxford Street
    - All locations are open Monday-Friday 9am-5pm local time
    - Visitors must check in at the front desk with photo ID
    
    Respond with a JSON object in the following format:
    {{"message": "Your response to the user", "follow_up_question": "Optional follow-up question", "is_farewell": false}}
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["user_message"]
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

# FAQ agent
def create_faq_agent(llm):
    template = """
    You are a FAQ specialist. Your job is to answer frequently asked questions about our products and services.
    
    User message: {user_message}
    
    Knowledge base:
    - Our standard shipping takes 3-5 business days
    - Express shipping is available for an additional $15
    - Returns are accepted within 30 days with receipt
    - Our warranty covers manufacturing defects for 1 year
    - We accept all major credit cards and PayPal
    - Customer service is available 24/7 via chat and 8am-8pm ET by phone
    
    Respond with a JSON object in the following format:
    {{"message": "Your response to the user", "follow_up_question": "Optional follow-up question", "is_farewell": false}}
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["user_message"]
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

# Email validation
def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

# Parse JSON safely
def parse_json_response(response_text):
    try:
        # Find JSON object in the text
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)
        return {"message": response_text, "follow_up_question": None, "is_farewell": False}
    except json.JSONDecodeError:
        return {"message": response_text, "follow_up_question": None, "is_farewell": False}

# Parse intent JSON safely
def parse_intent_response(response_text):
    try:
        # Find JSON object in the text
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            result = json.loads(json_str)
            return Intent(intent_type=result.get("intent_type", "unknown"), 
                         confidence=float(result.get("confidence", 0.0)))
        return Intent(intent_type="unknown", confidence=0.0)
    except (json.JSONDecodeError, ValueError):
        return Intent(intent_type="unknown", confidence=0.0)

# Main Streamlit app
def main():
    st.title("Customer Support ChatBot")
    
    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi there! Welcome to ChatBot. How can I help you today?"}
        ]
    
    if "user_info" not in st.session_state:
        st.session_state.user_info = UserInfo()
    
    if "current_agent" not in st.session_state:
        st.session_state.current_agent = "general"
    
    if "llm" not in st.session_state:
        with st.spinner("Initializing AI model..."):
            st.session_state.llm = initialize_llm()
            st.session_state.general_agent = create_general_agent(st.session_state.llm)
            st.session_state.intent_agent = create_intent_agent(st.session_state.llm)
            st.session_state.location_agent = create_location_agent(st.session_state.llm)
            st.session_state.faq_agent = create_faq_agent(st.session_state.llm)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Process user input
    if user_input := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        # Process the message
        with st.spinner("Thinking..."):
            # Check if we need to collect user info first
            if not st.session_state.user_info.name:
                st.session_state.user_info.name = user_input
                response_text = f"Nice to meet you, {user_input}! Could you please provide your email address so I can better assist you?"
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                with st.chat_message("assistant"):
                    st.write(response_text)
                return
            
            if not st.session_state.user_info.email:
                if is_valid_email(user_input):
                    st.session_state.user_info.email = user_input
                    response_text = f"Thank you, {st.session_state.user_info.name}! How can I assist you today?"
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    with st.chat_message("assistant"):
                        st.write(response_text)
                else:
                    response_text = "That doesn't look like a valid email address. Could you please try again?"
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    with st.chat_message("assistant"):
                        st.write(response_text)
                return
            
            # Get conversation history for context
            conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" 
                                             for msg in st.session_state.messages[-5:]])
            
            # Determine intent for routing
            intent_response = st.session_state.intent_agent.run(user_message=user_input)
            intent = parse_intent_response(intent_response)
            
            # Route to appropriate agent based on intent or current agent
            if intent.intent_type == "goodbye" or "goodbye" in user_input.lower():
                st.session_state.current_agent = "general"
                response = st.session_state.general_agent.run(
                    user_message=user_input,
                    user_info=st.session_state.user_info.dict(),
                    conversation_history=conversation_history
                )
                response_data = parse_json_response(response)
                response_data["is_farewell"] = True
            elif intent.intent_type == "location" or st.session_state.current_agent == "location":
                st.session_state.current_agent = "location"
                response = st.session_state.location_agent.run(user_message=user_input)
                response_data = parse_json_response(response)
            elif intent.intent_type == "faq" or st.session_state.current_agent == "faq":
                st.session_state.current_agent = "faq"
                response = st.session_state.faq_agent.run(user_message=user_input)
                response_data = parse_json_response(response)
            else:
                # Default to general agent
                st.session_state.current_agent = "general"
                response = st.session_state.general_agent.run(
                    user_message=user_input,
                    user_info=st.session_state.user_info.dict(),
                    conversation_history=conversation_history
                )
                response_data = parse_json_response(response)
            
            # Add bot response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response_data["message"]})
            with st.chat_message("assistant"):
                st.write(response_data["message"])
            
            # Add follow-up question if provided
            if response_data.get("follow_up_question"):
                st.session_state.messages.append({"role": "assistant", "content": response_data["follow_up_question"]})
                with st.chat_message("assistant"):
                    st.write(response_data["follow_up_question"])
            
            # Check if this is a farewell
            if response_data.get("is_farewell", False):
                farewell_msg = "Thank you for chatting with us today! If you need help in the future, just come back. Goodbye!"
                st.session_state.messages.append({"role": "assistant", "content": farewell_msg})
                with st.chat_message("assistant"):
                    st.write(farewell_msg)
                
                # Reset for a new conversation
                if st.button("Start New Conversation"):
                    st.session_state.messages = [
                        {"role": "assistant", "content": "Hi there! Welcome to ChatBot. How can I help you today?"}
                    ]
                    st.session_state.user_info = UserInfo()
                    st.session_state.current_agent = "general"
                    st.experimental_rerun()

if __name__ == "__main__":
    main()