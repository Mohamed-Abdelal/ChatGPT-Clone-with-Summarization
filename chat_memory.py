# chat_memory.py - Conversation Memory Management
# Part 1: Imports and Setup
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.schema import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import json
import os

load_dotenv()


class ChatMemoryManager:
    def __init__(self, memory_type="buffer"):
        """Initialize chat memory manager
        
        Args:
            memory_type: Either "buffer" for full history or "summary" for summarized history
        """
        # Initialize Groq language model for summarization
        self.llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.7,  # Lower temperature for more consistent summaries
        )
        
        # Set up memory based on type
        if memory_type == "buffer":
            self.memory = ConversationBufferMemory(return_messages=True)
        elif memory_type == "summary":
            self.memory = ConversationSummaryMemory(llm=self.llm, return_messages=True)
        else:
            raise ValueError(f"Unsupported memory type: {memory_type}")
        
        self.memory_type = memory_type

    # Part 2: Memory Methods
    def add_user_message(self, message):
        """Add a user message to memory"""
        self.memory.chat_memory.add_user_message(message)

    def add_ai_message(self, message):
        """Add an AI message to memory"""
        self.memory.chat_memory.add_ai_message(message)

    def get_chat_history(self):
        """Get the full chat history"""
        return self.memory.chat_memory.messages

    def get_message_count(self):
        """Get the number of messages in history"""
        return len(self.memory.chat_memory.messages)

    def clear_memory(self):
        """Clear the chat history"""
        self.memory.chat_memory.clear()

    def get_formatted_history(self):
        """Get chat history as a formatted string"""
        formatted = []
        for message in self.memory.chat_memory.messages:
            if isinstance(message, HumanMessage):
                formatted.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                formatted.append(f"AI: {message.content}")
        return "\n".join(formatted)

    # Part 3: Save and Load Methods
    def save_chat(self, filename="chat_history.json"):
        """Save chat history to a file"""
        history = []
        for message in self.memory.chat_memory.messages:
            if isinstance(message, HumanMessage):
                history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"role": "assistant", "content": message.content})

        with open(filename, "w") as f:
            json.dump(history, f, indent=2)
        
        print(f"Chat history saved to {filename}")

    def load_chat(self, filename="chat_history.json"):
        """Load chat history from a file"""
        if not os.path.exists(filename):
            print(f"File {filename} not found.")
            return False

        self.clear_memory()

        with open(filename, "r") as f:
            history = json.load(f)

        for message in history:
            if message["role"] == "user":
                self.add_user_message(message["content"])
            elif message["role"] == "assistant":
                self.add_ai_message(message["content"])

        print(f"Loaded {len(history)} messages from {filename}")
        return True

    def get_memory_variables(self):
        """Get memory variables for use with LLM chains"""
        return self.memory.load_memory_variables({})


if __name__ == "__main__":
    # Test the memory manager
    manager = ChatMemoryManager(memory_type="buffer")
    
    manager.add_user_message("Hello, how are you?")
    manager.add_ai_message("I'm doing well, thank you! How can I help you today?")
    manager.add_user_message("Can you explain what Python is?")
    manager.add_ai_message("Python is a high-level, interpreted programming language known for its simplicity and readability.")
    
    print("Chat History:")
    print(manager.get_formatted_history())
    print(f"\nTotal messages: {manager.get_message_count()}")
    
    manager.save_chat("test_chat.json")
