# summarizer.py - Chat Summarization Component
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()


class ChatSummarizer:
    def __init__(self):
        """Initialize the chat summarizer"""
        # Using Groq with Llama model (free and fast!)
        self.llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.3,
        )
        
        # Summary prompt template
        self.summary_template = PromptTemplate(
            template="""
            Please provide a concise summary of the following conversation.
            Focus on the main topics discussed, key points, and any conclusions reached.
            
            Conversation:
            {conversation}
            
            Summary:
            """,
            input_variables=["conversation"]
        )
        
        # Bullet point summary template
        self.bullet_summary_template = PromptTemplate(
            template="""
            Please provide a bullet-point summary of the following conversation.
            Include the main topics, key questions asked, and important answers given.
            
            Conversation:
            {conversation}
            
            Bullet Point Summary:
            """,
            input_variables=["conversation"]
        )
        
        # Topic extraction template
        self.topic_extraction_template = PromptTemplate(
            template="""
            Extract the main topics discussed in the following conversation.
            Return them as a comma-separated list.
            
            Conversation:
            {conversation}
            
            Main Topics:
            """,
            input_variables=["conversation"]
        )
        
        # Create chains
        self.summary_chain = LLMChain(llm=self.llm, prompt=self.summary_template)
        self.bullet_chain = LLMChain(llm=self.llm, prompt=self.bullet_summary_template)
        self.topic_chain = LLMChain(llm=self.llm, prompt=self.topic_extraction_template)

    def format_conversation(self, messages):
        """Format messages list into a conversation string"""
        formatted = []
        for message in messages:
            if isinstance(message, HumanMessage):
                formatted.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                formatted.append(f"AI: {message.content}")
            elif isinstance(message, dict):
                role = message.get("role", "unknown")
                content = message.get("content", "")
                if role == "user":
                    formatted.append(f"Human: {content}")
                elif role == "assistant":
                    formatted.append(f"AI: {content}")
        return "\n".join(formatted)

    def summarize(self, messages):
        """Generate a concise summary of the conversation"""
        conversation = self.format_conversation(messages)
        if not conversation:
            return "No conversation to summarize."
        
        result = self.summary_chain.invoke({"conversation": conversation})
        return result["text"].strip()

    def summarize_bullet_points(self, messages):
        """Generate a bullet-point summary of the conversation"""
        conversation = self.format_conversation(messages)
        if not conversation:
            return "No conversation to summarize."
        
        result = self.bullet_chain.invoke({"conversation": conversation})
        return result["text"].strip()

    def extract_topics(self, messages):
        """Extract main topics from the conversation"""
        conversation = self.format_conversation(messages)
        if not conversation:
            return []
        
        result = self.topic_chain.invoke({"conversation": conversation})
        topics = result["text"].strip().split(",")
        return [topic.strip() for topic in topics if topic.strip()]


if __name__ == "__main__":
    # Test the summarizer
    summarizer = ChatSummarizer()
    
    test_messages = [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."},
        {"role": "user", "content": "What are the main types of machine learning?"},
        {"role": "assistant", "content": "The three main types are: 1) Supervised learning - learning from labeled data, 2) Unsupervised learning - finding patterns in unlabeled data, and 3) Reinforcement learning - learning through trial and error with rewards."},
    ]
    
    print("=== Summary ===")
    print(summarizer.summarize(test_messages))
    
    print("\n=== Topics ===")
    print(summarizer.extract_topics(test_messages))
