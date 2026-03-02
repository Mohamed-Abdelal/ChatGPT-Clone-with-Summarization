# =============================================================================
# ChatGPT Clone with Summarization - Project 5
# LangChain Memory Module Learning Project
# =============================================================================
# This project demonstrates different LangChain memory types:
# 1. ConversationBufferMemory - Stores full conversation history
# 2. ConversationSummaryMemory - Automatically summarizes conversation
# =============================================================================

import streamlit as st
from streamlit_chat import message
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.prompts import PromptTemplate


def get_llm(api_key):
    """
    Initialize the Language Model (LLM)
    
    We use Groq's free API with Llama model.
    The LLM is the core component that generates responses.
    """
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        groq_api_key=api_key
    )


def create_conversation_with_memory(api_key, memory_type):
    """
    Create a ConversationChain with the selected memory type.
    
    This demonstrates how LangChain memory works:
    - ConversationBufferMemory: Keeps ALL messages in memory (good for short chats)
    - ConversationSummaryMemory: Summarizes old messages (good for long chats, saves tokens)
    
    Args:
        api_key: The Groq API key
        memory_type: Either "buffer" or "summary"
    
    Returns:
        conversation: The ConversationChain object
        memory: The memory object (for inspection)
    """
    llm = get_llm(api_key)
    
    if memory_type == "buffer":
        # ConversationBufferMemory stores the complete conversation
        # Advantage: Full context available
        # Disadvantage: Token count grows with conversation length
        memory = ConversationBufferMemory(return_messages=True)
    else:
        # ConversationSummaryMemory uses LLM to summarize conversation
        # Advantage: Constant token usage regardless of conversation length
        # Disadvantage: May lose some details in summarization
        memory = ConversationSummaryMemory(llm=llm, return_messages=True)
    
    # ConversationChain combines LLM + Memory for stateful conversations
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True  # Set to True to see internal workings
    )
    
    return conversation, memory


def get_response(user_input):
    """
    Get response from the chatbot.
    
    The conversation.predict() method:
    1. Loads conversation history from memory
    2. Adds history to the prompt
    3. Sends to LLM
    4. Saves new exchange to memory
    """
    response = st.session_state.conversation.predict(input=user_input)
    return response


def generate_summary(api_key, messages):
    """
    Generate a summary of the conversation using LLMChain.
    
    This shows how to create custom chains with PromptTemplate.
    """
    if len(messages) == 0:
        return "No conversation to summarize yet."
    
    llm = get_llm(api_key)
    
    # Create a custom prompt template for summarization
    summary_template = PromptTemplate(
        template="""Summarize the following conversation between a human and an AI.
Focus on the key points discussed.

Conversation:
{conversation}

Summary:""",
        input_variables=["conversation"]
    )
    
    # Format conversation
    conversation_text = ""
    for i, msg in enumerate(messages):
        if i % 2 == 0:
            conversation_text += f"Human: {msg}\n"
        else:
            conversation_text += f"AI: {msg}\n"
    
    # Create and run the chain
    summary_chain = LLMChain(llm=llm, prompt=summary_template)
    result = summary_chain.invoke({"conversation": conversation_text})
    
    return result["text"]


# =============================================================================
# Initialize Session State Variables
# =============================================================================
# Session state persists data between Streamlit reruns

if "conversation" not in st.session_state:
    st.session_state.conversation = None

if "memory" not in st.session_state:
    st.session_state.memory = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory_type" not in st.session_state:
    st.session_state.memory_type = "buffer"

if "api_key" not in st.session_state:
    st.session_state.api_key = ""


# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="ChatGPT Clone - LangChain Memory Demo",
    page_icon="🤖"
)

st.markdown("# 🤖 How can I assist you?")
st.markdown("*LangChain Memory Module Learning Project*")


# =============================================================================
# Sidebar - Settings
# =============================================================================
st.sidebar.title("⚙️ Settings")

# API Key Input (like in Coursera videos)
st.sidebar.markdown("### 🔑 API Key")
api_key = st.sidebar.text_input(
    "Enter your Groq API Key:",
    type="password",
    value=st.session_state.api_key,
    help="Get free API key at: https://console.groq.com/keys"
)

if api_key:
    st.session_state.api_key = api_key

st.sidebar.markdown("---")

# =============================================================================
# Memory Type Selection (Core Learning Feature)
# =============================================================================
st.sidebar.markdown("### 🧠 Choose Memory Type")
st.sidebar.markdown("*This is the core concept of this project!*")

memory_options = {
    "1. Buffer Memory (Full History)": "buffer",
    "2. Summary Memory (Auto Summarization)": "summary"
}

selected_memory = st.sidebar.radio(
    "Select memory type:",
    options=list(memory_options.keys()),
    index=0 if st.session_state.memory_type == "buffer" else 1
)

new_memory_type = memory_options[selected_memory]

# Handle memory type change
if new_memory_type != st.session_state.memory_type:
    st.session_state.memory_type = new_memory_type
    st.session_state.conversation = None
    st.session_state.memory = None
    st.session_state.messages = []
    st.experimental_rerun()

# Display memory type explanation
st.sidebar.markdown("---")
if st.session_state.memory_type == "buffer":
    st.sidebar.info("""
    **📝 ConversationBufferMemory**
    
    - Stores COMPLETE conversation history
    - All messages kept in memory
    - Best for: Short conversations
    - Drawback: Token count grows
    """)
else:
    st.sidebar.info("""
    **📊 ConversationSummaryMemory**
    
    - Automatically SUMMARIZES conversation
    - Uses LLM to create running summary
    - Best for: Long conversations
    - Drawback: May lose some details
    """)

st.sidebar.markdown("---")

# =============================================================================
# Summarization Feature
# =============================================================================
st.sidebar.markdown("### 📋 Summarization")

if st.sidebar.button("📝 Generate Summary", key="summary_btn"):
    if st.session_state.api_key and len(st.session_state.messages) > 0:
        with st.spinner("Generating summary..."):
            summary = generate_summary(st.session_state.api_key, st.session_state.messages)
            st.sidebar.success("Summary generated!")
            st.sidebar.write(summary)
    elif not st.session_state.api_key:
        st.sidebar.warning("Please enter API key first!")
    else:
        st.sidebar.warning("No conversation to summarize!")

st.sidebar.markdown("---")

# =============================================================================
# Clear Conversation
# =============================================================================
st.sidebar.markdown("### 🗑️ Clear Conversation")

if st.sidebar.button("Clear Conversation", key="clear_btn"):
    if st.session_state.api_key:
        st.session_state.conversation, st.session_state.memory = create_conversation_with_memory(
            st.session_state.api_key, 
            st.session_state.memory_type
        )
        st.session_state.messages = []
        st.sidebar.success("Conversation cleared!")
        st.experimental_rerun()
    else:
        st.sidebar.warning("Please enter API key first!")

# Message count
st.sidebar.markdown("---")
st.sidebar.markdown(f"**💬 Messages:** {len(st.session_state.messages)}")


# =============================================================================
# Main Chat Interface
# =============================================================================

# Check for API key
if not st.session_state.api_key:
    st.warning("⚠️ Please enter your Groq API key in the sidebar to start chatting!")
    st.info("Get your free API key at: https://console.groq.com/keys")
    st.stop()

# Initialize conversation if needed
if st.session_state.conversation is None:
    st.session_state.conversation, st.session_state.memory = create_conversation_with_memory(
        st.session_state.api_key,
        st.session_state.memory_type
    )

# Chat containers
response_container = st.container()
input_container = st.container()

# Input form
with input_container:
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Your question goes here:",
            key="user_input",
            height=100
        )
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input:
        # Add user message
        st.session_state.messages.append(user_input)
        
        # Get AI response
        with st.spinner("Thinking..."):
            try:
                response = get_response(user_input)
                st.session_state.messages.append(response)
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.messages.pop()  # Remove user message if error

# Display conversation
with response_container:
    if len(st.session_state.messages) > 0:
        for i in range(len(st.session_state.messages)):
            if i % 2 == 0:
                message(st.session_state.messages[i], is_user=True, key=f"{i}_user")
            else:
                message(st.session_state.messages[i], is_user=False, key=f"{i}_ai")
