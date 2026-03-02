# main.py - ChatGPT Clone with Summarization (Console Version)
# Using Groq API (Free and Fast!)

from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from chat_memory import ChatMemoryManager
from summarizer import ChatSummarizer
from dotenv import load_dotenv
import os

load_dotenv()


class ChatGPTClone:
    def __init__(self, model_name="llama-3.3-70b-versatile", temperature=0.7):
        """Initialize the ChatGPT Clone application
        
        Args:
            model_name: The Groq model to use (default: llama-3.3-70b-versatile)
            temperature: Controls randomness in responses (0.0-1.0)
        
        Available Groq Models (Free):
            - llama-3.3-70b-versatile (best quality)
            - llama-3.1-8b-instant (faster)
            - mixtral-8x7b-32768
            - gemma2-9b-it
        """
        # Initialize the Groq language model
        self.llm = ChatGroq(
            model_name=model_name,
            temperature=temperature,
        )
        
        # Initialize memory manager
        self.memory_manager = ChatMemoryManager(memory_type="buffer")
        
        # Initialize summarizer
        self.summarizer = ChatSummarizer()
        
        # Create conversation chain with memory
        self.memory = ConversationBufferMemory(return_messages=True)
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True
        )
        
        # System message for the chatbot
        self.system_message = """You are a helpful, friendly, and knowledgeable AI assistant. 
        You provide clear, accurate, and concise responses to user questions.
        You can help with a wide range of topics including coding, writing, analysis, and general knowledge.
        Always be respectful and professional in your responses."""

    def chat(self, user_input):
        """Process user input and generate a response
        
        Args:
            user_input: The user's message
            
        Returns:
            The AI's response
        """
        # Get response from conversation chain
        response = self.conversation.predict(input=user_input)
        
        # Also add to our memory manager for summarization
        self.memory_manager.add_user_message(user_input)
        self.memory_manager.add_ai_message(response)
        
        return response

    def get_summary(self):
        """Get a summary of the current conversation"""
        messages = self.memory_manager.get_chat_history()
        if not messages:
            return "No conversation to summarize yet."
        return self.summarizer.summarize(messages)

    def get_bullet_summary(self):
        """Get a bullet-point summary of the current conversation"""
        messages = self.memory_manager.get_chat_history()
        if not messages:
            return "No conversation to summarize yet."
        return self.summarizer.summarize_bullet_points(messages)

    def get_topics(self):
        """Get the main topics discussed in the conversation"""
        messages = self.memory_manager.get_chat_history()
        if not messages:
            return []
        return self.summarizer.extract_topics(messages)

    def save_conversation(self, filename="conversation.json"):
        """Save the current conversation to a file"""
        self.memory_manager.save_chat(filename)

    def load_conversation(self, filename="conversation.json"):
        """Load a conversation from a file"""
        return self.memory_manager.load_chat(filename)

    def clear_conversation(self):
        """Clear the current conversation history"""
        self.memory_manager.clear_memory()
        self.memory.clear()
        print("Conversation cleared.")

    def get_message_count(self):
        """Get the number of messages in the conversation"""
        return self.memory_manager.get_message_count()

    def get_conversation_history(self):
        """Get the formatted conversation history"""
        return self.memory_manager.get_formatted_history()


def print_menu():
    """Print the main menu options"""
    print("\n" + "=" * 50)
    print("ChatGPT Clone with Summarization (Powered by Groq)")
    print("=" * 50)
    print("\nCommands:")
    print("  /summary     - Get a summary of the conversation")
    print("  /bullets     - Get bullet-point summary")
    print("  /topics      - Get main topics discussed")
    print("  /history     - Show conversation history")
    print("  /save        - Save conversation to file")
    print("  /load        - Load conversation from file")
    print("  /clear       - Clear conversation history")
    print("  /count       - Show message count")
    print("  /help        - Show this menu")
    print("  /quit        - Exit the application")
    print("-" * 50)


def run_interactive_chat():
    """Run the interactive chat application"""
    print("\n" + "=" * 60)
    print("   Welcome to ChatGPT Clone with Summarization!")
    print("   Powered by Groq (Free & Fast!)")
    print("=" * 60)
    print("\nThis is an AI-powered chatbot with conversation summarization.")
    print("Type your message and press Enter to chat.")
    print("Type /help for available commands.\n")
    
    # Initialize chatbot
    chatbot = ChatGPTClone()
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() == "/quit":
                print("\nGoodbye! Thanks for chatting!")
                break
            
            elif user_input.lower() == "/help":
                print_menu()
            
            elif user_input.lower() == "/summary":
                print("\n--- Conversation Summary ---")
                print(chatbot.get_summary())
            
            elif user_input.lower() == "/bullets":
                print("\n--- Bullet Point Summary ---")
                print(chatbot.get_bullet_summary())
            
            elif user_input.lower() == "/topics":
                print("\n--- Main Topics ---")
                topics = chatbot.get_topics()
                if topics:
                    for i, topic in enumerate(topics, 1):
                        print(f"  {i}. {topic}")
                else:
                    print("No topics found yet.")
            
            elif user_input.lower() == "/history":
                print("\n--- Conversation History ---")
                history = chatbot.get_conversation_history()
                if history:
                    print(history)
                else:
                    print("No conversation history yet.")
            
            elif user_input.lower() == "/save":
                filename = input("Enter filename (default: conversation.json): ").strip()
                if not filename:
                    filename = "conversation.json"
                chatbot.save_conversation(filename)
            
            elif user_input.lower() == "/load":
                filename = input("Enter filename (default: conversation.json): ").strip()
                if not filename:
                    filename = "conversation.json"
                chatbot.load_conversation(filename)
            
            elif user_input.lower() == "/clear":
                confirm = input("Are you sure you want to clear the conversation? (y/n): ").strip().lower()
                if confirm == "y":
                    chatbot.clear_conversation()
            
            elif user_input.lower() == "/count":
                count = chatbot.get_message_count()
                print(f"\nTotal messages in conversation: {count}")
            
            else:
                # Regular chat message
                print("\nAssistant: ", end="")
                response = chatbot.chat(user_input)
                print(response)
        
        except KeyboardInterrupt:
            print("\n\nGoodbye! Thanks for chatting!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again or type /quit to exit.")


if __name__ == "__main__":
    run_interactive_chat()
