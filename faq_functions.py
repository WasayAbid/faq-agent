# faq_functions.py

import os
import sqlite3 # --- NEW --- Import the sqlite3 library
from typing import TypedDict, Optional
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "dubai-faq-index")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
DB_FILE = "dubai_faq.db" # --- NEW --- Define database file name

# State definition
class FAQState(TypedDict):
    question: str
    answer: Optional[str]
    similarity_score: Optional[float]
    method: Optional[str]  # "sql_match", "vector_match", "llm_generated", etc. # --- MODIFIED ---
    matched_question: Optional[str]
    error: Optional[str]

class DubaiFAQService:
    """Service class containing all FAQ processing functions"""
    
    def __init__(self):
        """Initialize all services"""
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize Pinecone, Gemini, and embedding model"""
        try:
            # Validate environment variables
            if not PINECONE_API_KEY:
                raise ValueError("PINECONE_API_KEY not found in environment variables!")
            if not GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not found in environment variables!")
            
            # --- NEW --- Check if database file exists
            if not os.path.exists(DB_FILE):
                raise FileNotFoundError(f"Database file '{DB_FILE}' not found. Please run 'load_sql.py' first.")
            print(f"✔️ Database file '{DB_FILE}' found.")
            # --- END NEW ---

            # Initialize embedding model
            print("🧠 Loading embedding model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize Gemini
            print("🤖 Initializing Gemini LLM...")
            genai.configure(api_key=GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel('gemini-2.5-pro')
            
            # Initialize Pinecone
            print("📍 Connecting to Pinecone...")
            pc = Pinecone(api_key=PINECONE_API_KEY)
            self.pinecone_index = pc.Index(INDEX_NAME)
            
            print("✅ All services initialized successfully!")
            
        except Exception as e:
            print(f"❌ Service initialization failed: {e}")
            raise

    # --- NEW FUNCTION ---
    def query_sql_database(self, state: FAQState) -> FAQState:
        """
        Search the SQLite database for an exact match to the question.
        This is the first step in the workflow.
        """
        try:
            question = state["question"]
            print(f"🔍 Searching SQL database for: '{question}'")
            
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            
            # Query for an exact, case-insensitive match
            # The '?' is a placeholder to prevent SQL injection
            cursor.execute("SELECT answer FROM faqs WHERE lower(question) = lower(?)", (question.strip(),))
            result = cursor.fetchone()
            
            conn.close()
            
            if result:
                answer = result[0]
                state["answer"] = answer
                state["method"] = "sql_match"
                print(f"✅ Found exact match in SQL database.")
            else:
                print("❌ No exact match found in SQL database.")

        except Exception as e:
            print(f"🚨 SQL search error: {e}")
            state["error"] = f"SQL search failed: {str(e)}"
            
        return state
    # --- END NEW FUNCTION ---
    
    def search_vector_database(self, state: FAQState) -> FAQState:
        """
        Search Pinecone vector database for similar questions
        
        Args:
            state: Current FAQ state containing user question
            
        Returns:
            Updated state with search results
        """
        # --- NEW --- Skip this step if we already have an answer from SQL
        if state.get("answer") is not None:
            return state
        # --- END NEW ---

        try:
            question = state["question"]
            print(f"🔍 Searching vector database for: '{question}'")
            
            # Generate embedding for the query
            query_embedding = self.embedding_model.encode([question])
            
            # Search Pinecone
            results = self.pinecone_index.query(
                vector=query_embedding[0].tolist(),
                top_k=1,
                include_metadata=True
            )
            
            if results.matches and len(results.matches) > 0:
                best_match = results.matches[0]
                similarity_score = best_match.score
                
                print(f"📊 Best match similarity: {similarity_score:.3f}")
                print(f"📝 Matched question: {best_match.metadata['question'][:100]}...")
                
                if similarity_score >= SIMILARITY_THRESHOLD:
                    state["answer"] = best_match.metadata['answer']
                    state["similarity_score"] = similarity_score
                    state["method"] = "vector_match"
                    state["matched_question"] = best_match.metadata['question']
                    print(f"✅ Using vector match (score: {similarity_score:.3f})")
                else:
                    print(f"❌ Similarity too low: {similarity_score:.3f} < {SIMILARITY_THRESHOLD}")
                    state["similarity_score"] = similarity_score
            else:
                print("❌ No matches found in vector database")
                
        except Exception as e:
            print(f"🚨 Vector search error: {e}")
            state["error"] = f"Vector search failed: {str(e)}"
        
        return state
    
    def call_llm_for_answer(self, state: FAQState) -> FAQState:
        """
        Generate answer using Gemini LLM
        
        Args:
            state: Current FAQ state
            
        Returns:
            Updated state with LLM-generated answer
        """
        # Skip if we already have an answer
        if state.get("answer") is not None:
            return state
        
        try:
            question = state["question"]
            print("🤖 Generating response with Gemini LLM...")
            
            # Create Dubai-focused prompt
            prompt = self._create_dubai_prompt(question)
            
            # Generate response
            response = self.gemini_model.generate_content(prompt)
            
            state["answer"] = response.text
            state["method"] = "llm_generated"
            print("✅ Generated answer using Gemini LLM")
            
        except Exception as e:
            print(f"🚨 LLM generation error: {e}")
            state["answer"] = self._get_error_message(str(e))
            state["method"] = "error"
            state["error"] = f"LLM generation failed: {str(e)}"
        
        return state
    
    def finalize_response(self, state: FAQState) -> FAQState:
        """
        Final processing step - ensure we have a response
        
        Args:
            state: Current FAQ state
            
        Returns:
            Final state with guaranteed answer
        """
        if not state.get("answer"):
            state["answer"] = "I apologize, but I'm unable to provide an answer at the moment. Please try again later."
            state["method"] = "fallback"
        
        print(f"✅ Response finalized using method: {state.get('method', 'unknown')}")
        return state

    # --- NEW ROUTING FUNCTION ---
    def should_search_vector_db(self, state: FAQState) -> str:
        """
        Decision function for conditional routing after SQL search.
        
        Args:
            state: Current FAQ state
            
        Returns:
            Next node name to execute
        """
        if state.get("answer") is not None:
            # Answer found in SQL, so we can finalize
            print("🔀 Routing to: finalize_response (SQL match found)")
            return "finalize_response"
        else:
            # No answer from SQL, proceed to vector search
            print("🔀 Routing to: search_vector_database (no SQL match)")
            return "search_vector_database"
    # --- END NEW ROUTING FUNCTION ---

    def should_use_llm(self, state: FAQState) -> str:
        """
        Decision function for conditional routing after vector search.
        
        Args:
            state: Current FAQ state
            
        Returns:
            Next node name to execute
        """
        if state.get("answer") is not None:
            print("🔀 Routing to: finalize_response (vector match found)")
            return "finalize_response"
        else:
            print("🔀 Routing to: call_llm_for_answer (no vector match)")
            return "call_llm_for_answer"
    
    def _create_dubai_prompt(self, question: str) -> str:
        """Create a Dubai-focused prompt for the LLM"""
        return f"""
You are a helpful and knowledgeable assistant specializing in Dubai, UAE. You have extensive knowledge about Dubai's tourism, business, culture, lifestyle, and practical information.

Please provide a comprehensive, accurate, and helpful answer to the following question about Dubai:

Question: {question}

Guidelines for your response:
- Focus on providing specific, practical information about Dubai
- Include relevant details about locations, timings, costs, or procedures when applicable
- If the question involves recommendations, provide 2-3 good options
- Keep the tone friendly and informative
- If the question is not directly related to Dubai, try to relate it to Dubai context when possible
- If you cannot provide Dubai-specific information, politely explain and offer to help with Dubai-related topics instead

Please provide a detailed and helpful response:
"""
    
    def _get_error_message(self, error: str) -> str:
        """Generate user-friendly error message"""
        return (
            "I apologize, but I'm unable to generate an answer at the moment. "
            "This could be due to a temporary service issue. Please try rephrasing "
            "your question or try again in a moment."
        )

# Create global service instance
faq_service = DubaiFAQService()