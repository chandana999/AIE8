"""
Utility functions for LangGraph MCP Tools
"""

import asyncio
from typing import List, Any, Dict
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

class GraphVisualizer:
    """Utility class for visualizing LangGraph workflows"""
    
    @staticmethod
    def print_graph_info(graph: StateGraph) -> None:
        """Print information about the graph structure"""
        print("\n📊 Graph Structure:")
        print(f"   - Nodes: {list(graph.nodes.keys())}")
        print(f"   - Edges: {len(graph.edges)}")
    
    @staticmethod
    def get_mermaid_diagram() -> str:
        """Generate Mermaid diagram for the LangGraph workflow"""
        return """
graph TD
    START([START]) --> call_model[Call Model]
    call_model --> tools_condition{Tools Needed?}
    tools_condition -->|Yes| tools[Execute Tools]
    tools_condition -->|No| END([END])
    tools --> call_model
    
    classDef startEnd fill:#e1f5fe
    classDef process fill:#f3e5f5
    classDef decision fill:#fff3e0
    
    class START,END startEnd
    class call_model,tools process
    class tools_condition decision
        """
    
    @staticmethod
    def display_mermaid_instructions() -> None:
        """Display instructions for rendering Mermaid diagrams"""
        print("\n🎨 To render the Mermaid diagram:")
        print("1. Copy the diagram code above")
        print("2. Go to https://mermaid.live/")
        print("3. Paste the code and view the diagram")
        print("4. Or use in Markdown: ```mermaid [diagram code] ```")

class QueryFormatter:
    """Utility class for formatting queries and responses"""
    
    @staticmethod
    def format_query(query: str, index: int) -> str:
        """Format query for display"""
        return f"[{index+1}] {query}"
    
    @staticmethod
    def format_response(response: str, query: str) -> str:
        """Format response for display"""
        return f"Query: {query}\nResponse: {response}\n" + "-"*50

class RateLimiter:
    """Utility class for rate limiting requests"""
    
    def __init__(self, delay: float = 1.0):
        self.delay = delay
    
    async def wait(self) -> None:
        """Wait for the specified delay"""
        await asyncio.sleep(self.delay)

class ErrorHandler:
    """Utility class for handling errors gracefully"""
    
    @staticmethod
    def handle_mcp_error(error: Exception) -> str:
        """Handle MCP-related errors"""
        if "Connection closed" in str(error):
            return "❌ MCP server connection lost. Please ensure server.py is running."
        elif "ModuleNotFoundError" in str(error):
            return "❌ Missing dependencies. Run 'uv sync' to install required packages."
        else:
            return f"❌ Unexpected error: {str(error)}"
    
    @staticmethod
    def handle_api_error(error: Exception) -> str:
        """Handle API-related errors"""
        if "API key" in str(error).lower():
            return "❌ API key issue. Check your environment variables."
        else:
            return f"❌ API error: {str(error)}"
