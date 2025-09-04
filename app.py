# app.py

import asyncio
import chainlit as cl
from workflow_manager import workflow_manager
from faq_functions import FAQState

class DubaiFAQInterface:
    """Handles all Chainlit interface interactions"""
    
    @staticmethod
    def get_welcome_message() -> str:
        """Generate welcome message for new chat sessions"""
        return """
🏙️ **Welcome to Dubai FAQ Assistant!** 🇦🇪

I'm here to help you with questions about Dubai, including:

🎯 **Tourism & Attractions**
- Best places to visit and experiences
- Weather, timing, and seasonal information
- Popular landmarks and hidden gems

💼 **Business & Practical Info**
- Visas, documentation, and legal requirements
- Transportation and navigation
- Business customs and procedures

🏛️ **Culture & Lifestyle**
- Local customs, etiquette, and traditions
- Food, dining, and culinary experiences
- Shopping, entertainment, and nightlife

💡 **Pro Tips**: 
- Ask specific questions for the best answers
- I can help with both tourist and resident queries
- Try questions like: *"What's the best time to visit Dubai?"* or *"How do I get from the airport to downtown?"*

What would you like to know about Dubai? 🤔
"""
    
    @staticmethod
    def format_response(result: FAQState) -> str:
        """
        Format the workflow result for display in Chainlit
        
        Args:
            result: Final state from workflow execution
            
        Returns:
            Formatted message string for Chainlit
        """
        method = result.get('method', 'unknown')
        answer = result.get('answer', 'No answer available')
        
        # --- NEW --- Add a condition to handle SQL matches
        if method == "sql_match":
            # SQL Database match
            response_content = f"✅ **Found in Database**\n\n{answer}"
        
        elif method == "vector_match":
            # FAQ Database match
            similarity_score = result.get('similarity_score', 0)
            response_content = f"📊 **Found in FAQ Database** (Match: {similarity_score:.1%})\n\n{answer}"
            
            # Add matched question info if different from user's question
            matched_question = result.get('matched_question')
            user_question = result.get('question', '')
            
            if matched_question and matched_question.lower() != user_question.lower():
                response_content += f"\n\n💡 *This answer is for: \"{matched_question}\"*"
                
        elif method == "llm_generated":
            # AI Generated response
            response_content = f"🤖 **AI Generated Response**\n\n{answer}"
            
        elif method == "error":
            # Error occurred
            response_content = f"❌ **Service Issue**\n\n{answer}"
            
            # Add troubleshooting info for errors
            response_content += "\n\n🔧 **Troubleshooting:**\n"
            response_content += "- Check your internet connection\n"
            response_content += "- Try rephrasing your question\n"
            response_content += "- Wait a moment and try again"
            
        elif method == "fallback":
            # Fallback response
            response_content = f"⚠️ **Fallback Response**\n\n{answer}"
            
        else:
            # Unknown method
            response_content = f"🔍 **Response**\n\n{answer}"
        
        return response_content
    
    @staticmethod
    async def process_user_message(message_content: str) -> str:
        """
        Process user message through the workflow
        
        Args:
            message_content: User's question
            
        Returns:
            Formatted response string
        """
        try:
            # Clean the user input
            user_question = message_content.strip()
            
            if not user_question:
                return "Please ask a question about Dubai, and I'll be happy to help! 😊"
            
            # Process through workflow (run in executor to avoid blocking)
            result = await asyncio.get_event_loop().run_in_executor(
                None, workflow_manager.process_question, user_question
            )
            
            # Format and return response
            return DubaiFAQInterface.format_response(result)
            
        except Exception as e:
            print(f"🚨 Error in process_user_message: {e}")
            return (
                f"❌ **Unexpected Error**\n\n"
                f"I encountered an unexpected error while processing your question: {str(e)}\n\n"
                f"Please try asking your question again, or try rephrasing it differently."
            )

# Initialize the interface
faq_interface = DubaiFAQInterface()

# Chainlit Event Handlers
@cl.on_chat_start
async def on_chat_start():
    """Handle new chat session initialization"""
    try:
        # Send welcome message
        welcome_message = faq_interface.get_welcome_message()
        await cl.Message(content=welcome_message).send()
        
        print("💬 New chat session started")
        
    except Exception as e:
        print(f"🚨 Error in chat start: {e}")
        await cl.Message(
            content="❌ Sorry, there was an issue starting the chat session. Please refresh and try again."
        ).send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user messages"""
    try:
        user_question = message.content
        print(f"\n👤 User Question: {user_question}")
        
        # Show thinking indicator
        thinking_msg = cl.Message(content="🤔 Searching for the best answer...")
        await thinking_msg.send()
        
        # Process the message
        response_content = await faq_interface.process_user_message(user_question)
        
        # Send the response
        await cl.Message(content=response_content).send()
        
        print(f"✅ Response sent successfully")
        
    except Exception as e:
        print(f"🚨 Error handling message: {e}")
        await cl.Message(
            content=(
                f"❌ **Sorry, I encountered an error while processing your question.**\n\n"
                f"Error details: {str(e)}\n\n"
                f"Please try asking your question again, perhaps with different wording."
            )
        ).send()

@cl.on_chat_end
async def on_chat_end():
    """Handle chat session end"""
    print("💬 Chat session ended")

@cl.on_stop
async def on_stop():
    """Handle application stop"""
    print("🛑 Application stopping...")

if __name__ == "__main__":
    print("🎯 Dubai FAQ AI App - Structured Version")
    print("=" * 50)
    print("📁 Architecture:")
    print("  ├── app.py (Chainlit UI - this file)")
    print("  ├── faq_functions.py (Business Logic)")
    print("  ├── workflow_manager.py (LangGraph Workflow)")
    print("  ├── load_sql.py (Database Setup)")
    print("  └── dubai_faq.db (SQLite Database)")
    print("=" * 50)
    print(f"🌐 Starting Chainlit server...")
    print(f"📱 Access the app at: http://localhost:8000")
    print(f"\n💡 To run: chainlit run app.py")
    print(f"🔧 Make sure all environment variables are set in .env file")
    print("=" * 50)