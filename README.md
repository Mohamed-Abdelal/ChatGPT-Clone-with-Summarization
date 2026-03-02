# 🤖 ChatGPT Clone with Summarization

## Project 5 - LangChain Memory Module

A ChatGPT-like chatbot built with **LangChain** that demonstrates different memory types for maintaining conversation context. This project is part of the **LangChain MasterClass** course.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-Memory-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)
![Groq](https://img.shields.io/badge/Groq-API-purple.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Concepts](#key-concepts)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [Memory Types Comparison](#memory-types-comparison)

---

## 🎯 Overview

This project demonstrates how to build a conversational AI chatbot using LangChain's memory modules. The main goal is to understand how LLMs (Large Language Models) can maintain context across multiple conversation turns.

### Why Memory Matters?

LLMs are **stateless** by default - they don't remember previous conversations. Each query is treated as a fresh request. LangChain's memory modules solve this problem by:

1. Storing conversation history
2. Automatically including context in prompts
3. Managing token limits efficiently

---

## 🧠 Key Concepts

### 1. ConversationBufferMemory
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)
```
- Stores **complete** conversation history
- Every message is kept in memory
- **Best for**: Short conversations
- **Drawback**: Token count grows with each message

### 2. ConversationSummaryMemory
```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=llm, return_messages=True)
```
- **Automatically summarizes** conversation
- Uses LLM to create a running summary
- **Best for**: Long conversations
- **Drawback**: May lose some details

### 3. ConversationChain
```python
from langchain.chains import ConversationChain

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)
```
- Combines LLM + Memory for stateful conversations
- Automatically manages history in prompts
- `verbose=True` shows internal workings

---

## 📁 Project Structure

```
chatgpt_clone/
├── app.py              # Streamlit UI application
├── main.py             # Console-based application
├── chat_memory.py      # ChatMemoryManager class
├── summarizer.py       # ChatSummarizer class
├── requirements.txt    # Project dependencies
├── .env                # Environment variables (API key)
└── README.md           # This file
```

### File Descriptions

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit application with UI for memory type selection |
| `main.py` | Console version with command-line interface |
| `chat_memory.py` | Wrapper class for managing different memory types |
| `summarizer.py` | Custom summarization using LLMChain and PromptTemplate |

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd chatgpt_clone
```

### 2. Create Virtual Environment
```bash
python -m venv langchain-env

# Windows
langchain-env\Scripts\activate

# macOS/Linux
source langchain-env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Get Groq API Key (Free)
1. Go to: https://console.groq.com/keys
2. Sign up / Log in
3. Create a new API key
4. Copy the key

---

## 💻 Usage

### Streamlit UI (Recommended)
```bash
streamlit run app.py
```

Then:
1. Enter your Groq API key in the sidebar
2. Select memory type (Buffer or Summary)
3. Start chatting!
4. Use "Generate Summary" to summarize the conversation
5. Use "Clear Conversation" to reset

### Console Version
```bash
python main.py
```

Commands:
- `/summary` - Get conversation summary
- `/clear` - Clear conversation
- `/help` - Show all commands
- `/quit` - Exit

---

## 📝 Code Explanation

### 1. chat_memory.py - Memory Management

```python
class ChatMemoryManager:
    def __init__(self, memory_type="buffer"):
        # Initialize LLM for summarization
        self.llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.0,
        )
        
        # Set up memory based on type
        if memory_type == "buffer":
            self.memory = ConversationBufferMemory(return_messages=True)
        elif memory_type == "summary":
            self.memory = ConversationSummaryMemory(llm=self.llm, return_messages=True)
```

**Key Methods:**
- `add_user_message(message)` - Add human message to memory
- `add_ai_message(message)` - Add AI response to memory
- `get_chat_history()` - Retrieve all messages
- `save_chat(filename)` - Save to JSON file
- `load_chat(filename)` - Load from JSON file

### 2. summarizer.py - Custom Summarization

```python
class ChatSummarizer:
    def __init__(self):
        self.llm = ChatGroq(...)
        
        # Custom prompt template
        self.summary_template = PromptTemplate(
            template="""
            Please provide a concise summary of the following conversation.
            
            Conversation:
            {conversation}
            
            Summary:
            """,
            input_variables=["conversation"]
        )
        
        # Create LLMChain
        self.summary_chain = LLMChain(llm=self.llm, prompt=self.summary_template)
```

**Key Methods:**
- `summarize(messages)` - Generate concise summary
- `summarize_bullet_points(messages)` - Generate bullet-point summary
- `extract_topics(messages)` - Extract main topics

### 3. app.py - Streamlit Application

```python
def create_conversation_with_memory(api_key, memory_type):
    llm = get_llm(api_key)
    
    if memory_type == "buffer":
        memory = ConversationBufferMemory(return_messages=True)
    else:
        memory = ConversationSummaryMemory(llm=llm, return_messages=True)
    
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    return conversation, memory
```

**Session State Variables:**
- `st.session_state.conversation` - The ConversationChain object
- `st.session_state.memory` - The memory object
- `st.session_state.messages` - List of messages for UI display
- `st.session_state.memory_type` - Current memory type selection

---

## ⚖️ Memory Types Comparison

| Feature | Buffer Memory | Summary Memory |
|---------|--------------|----------------|
| **Storage** | Full history | Running summary |
| **Token Usage** | Grows with conversation | Constant |
| **Context** | Complete | Summarized |
| **Best For** | Short chats | Long chats |
| **Speed** | Faster | Slower (needs summarization) |
| **Detail Loss** | None | Possible |

### When to Use Each:

**ConversationBufferMemory:**
- Short conversations (< 10 exchanges)
- When full context is critical
- When token limits are not a concern

**ConversationSummaryMemory:**
- Long conversations (> 10 exchanges)
- When saving tokens is important
- When general context is sufficient

---

## 🔧 Technologies Used

- **LangChain** - Framework for LLM applications
- **Groq API** - Fast, free LLM inference
- **Streamlit** - Web UI framework
- **Python 3.8+** - Programming language

---

## 📚 Learning Outcomes

After completing this project, you will understand:

1. ✅ Why memory is important for chatbots
2. ✅ How ConversationBufferMemory works
3. ✅ How ConversationSummaryMemory works
4. ✅ How to create ConversationChain
5. ✅ How to use PromptTemplate and LLMChain
6. ✅ How to build a Streamlit chat interface
7. ✅ How to manage session state in Streamlit

---

## 📄 License

This project is for educational purposes as part of the LangChain MasterClass course.

---

## 🙏 Acknowledgments

- LangChain MasterClass by Packt on Coursera
- Groq for providing free API access
