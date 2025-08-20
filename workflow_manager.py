"""
workflow_manager.py
==================
LangGraph workflow definition for Dubai FAQ system.
Contains only the workflow structure and graph management.
"""

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
        
        # Add nodes - these are the processing functions
        workflow.add_node("search_vector_database", faq_service.search_vector_database)
        workflow.add_node("call_llm_for_answer", faq_service.call_llm_for_answer)
        workflow.add_node("finalize_response", faq_service.finalize_response)
        
        # Add edges to define the flow
        # START -> search_vector_database (always start here)
        workflow.add_edge(START, "search_vector_database")
        
        # Conditional edge: search_vector_database -> finalize_response OR call_llm_for_answer
        workflow.add_conditional_edges(
            "search_vector_database",  # From this node
            faq_service.should_use_llm,  # Decision function
            {
                "finalize_response": "finalize_response",  # If vector match found
                "call_llm_for_answer": "call_llm_for_answer"  # If no vector match
            }
        )
        
        # Both LLM and finalize paths end the workflow
        workflow.add_edge("call_llm_for_answer", "finalize_response")
        workflow.add_edge("finalize_response", END)
        
        print("✅ LangGraph workflow compiled successfully!")
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
        
        # Step 1: Search vector database
        print("1️⃣ Searching vector database...")
        state = faq_service.search_vector_database(initial_state)
        
        # Step 2: Call LLM if needed
        if state.get("answer") is None:
            print("2️⃣ Calling LLM for answer...")
            state = faq_service.call_llm_for_answer(state)
        else:
            print("2️⃣ Skipping LLM (vector match found)")
        
        # Step 3: Finalize response
        print("3️⃣ Finalizing response...")
        state = faq_service.finalize_response(state)
        
        return state

# Global workflow manager instance
workflow_manager = WorkflowManager()