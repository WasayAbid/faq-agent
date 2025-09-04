# workflow_manager.py

from typing import Literal
from faq_functions import FAQState, faq_service

# Try to import langgraph, fall back to simple implementation if not available
try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
    print("📊 Using LangGraph for workflow management")
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("⚠️ LangGraph not available, using simple workflow")

class WorkflowManager:
    """Manages the FAQ processing workflow using LangGraph or simple implementation"""
    
    def __init__(self):
        """Initialize the workflow manager"""
        self.workflow = self._build_workflow()
    
    def _build_workflow(self):
        """Build the appropriate workflow based on LangGraph availability"""
        if LANGGRAPH_AVAILABLE:
            return self._build_langgraph_workflow()
        else:
            return SimpleWorkflow()
    
    def _build_langgraph_workflow(self):
        """Build the LangGraph workflow"""
        print("🏗️ Building LangGraph workflow...")
        
        # Create the StateGraph
        workflow = StateGraph(FAQState)
        
        # --- MODIFIED --- Add all nodes, including the new SQL search node
        workflow.add_node("query_sql_database", faq_service.query_sql_database)
        workflow.add_node("search_vector_database", faq_service.search_vector_database)
        workflow.add_node("call_llm_for_answer", faq_service.call_llm_for_answer)
        workflow.add_node("finalize_response", faq_service.finalize_response)
        
        # --- MODIFIED --- Add edges to define the NEW flow
        
        # 1. The workflow now starts at the SQL search
        workflow.add_edge(START, "query_sql_database")
        
        # 2. After searching SQL, decide where to go next
        workflow.add_conditional_edges(
            "query_sql_database",           # From this node
            faq_service.should_search_vector_db, # Use our NEW decision function
            {
                "finalize_response": "finalize_response",  # If SQL match was found
                "search_vector_database": "search_vector_database" # If no SQL match
            }
        )
        
        # 3. The rest of the flow remains the same as before
        workflow.add_conditional_edges(
            "search_vector_database",
            faq_service.should_use_llm,
            {
                "finalize_response": "finalize_response",
                "call_llm_for_answer": "call_llm_for_answer"
            }
        )
        
        # 4. Both LLM and finalize paths end the workflow
        workflow.add_edge("call_llm_for_answer", "finalize_response")
        workflow.add_edge("finalize_response", END)
        
        print("✅ LangGraph workflow compiled successfully with new SQL step!")
        return workflow.compile()
    
    def process_question(self, question: str) -> FAQState:
        """
        Process a user question through the workflow
        
        Args:
            question: User's question string
            
        Returns:
            Final state with answer and metadata
        """
        print(f"\n{'='*50}")
        print("🔄 Starting workflow execution...")
        print(f"❓ Question: {question}")
        print("="*50)
        
        # Create initial state
        initial_state = FAQState(
            question=question,
            answer=None,
            similarity_score=None,
            method=None,
            matched_question=None,
            error=None
        )
        
        # Execute workflow
        try:
            result = self.workflow.invoke(initial_state)
            print(f"✅ Workflow completed successfully")
            return result
        except Exception as e:
            print(f"🚨 Workflow execution failed: {e}")
            return FAQState(
                question=question,
                answer=f"Workflow execution failed: {str(e)}",
                similarity_score=None,
                method="error",
                matched_question=None,
                error=str(e)
            )

class SimpleWorkflow:
    """Simple workflow implementation when LangGraph is not available"""
    
    def __init__(self):
        print("🔧 Initialized simple workflow (LangGraph fallback)")
    
    def invoke(self, initial_state: FAQState) -> FAQState:
        """Execute the workflow steps manually"""
        print("🔄 Executing simple workflow...")
        
        # --- NEW --- Step 1: Search SQL database first
        print("1️⃣ Searching SQL database...")
        state = faq_service.query_sql_database(initial_state)

        # --- MODIFIED --- Step 2: Search vector database (only if needed)
        if state.get("answer") is None:
            print("2️⃣ Searching vector database...")
            state = faq_service.search_vector_database(state)
        else:
            print("2️⃣ Skipping vector search (SQL match found)")

        # --- MODIFIED --- Step 3: Call LLM if needed
        if state.get("answer") is None:
            print("3️⃣ Calling LLM for answer...")
            state = faq_service.call_llm_for_answer(state)
        else:
            print("3️⃣ Skipping LLM (answer already found)")
        
        # Step 4: Finalize response
        print("4️⃣ Finalizing response...")
        state = faq_service.finalize_response(state)
        
        return state

# Global workflow manager instance
workflow_manager = WorkflowManager()