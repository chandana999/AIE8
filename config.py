"""
Configuration settings for LangGraph MCP Tools
"""

import os
from typing import Dict, Any

class Config:
    """Configuration class for the LangGraph MCP application"""
    
    # MCP Server Configuration
    MCP_SERVER_CONFIG = {
        "mcp-server": {
            "command": "uv",
            "args": ["run", "server.py"],
            "transport": "stdio",
        }
    }
    
    # Model Configuration
    MODEL_NAME = "openai:gpt-4o-mini"
    
    # Rate limiting
    QUERY_DELAY = 1  # seconds between queries
    
    # Demo Queries
    DEMO_QUERIES = [
        "Search the web for the latest news about AI agents",
        "Roll 2d20 dice for me",
        "Find the location of the LTE cell tower with MCC 404, MNC 45, LAC 1234, CID 5678901"
    ]
    
    CUSTOM_QUERIES = [
        "What are the current trends in machine learning?",
        "Roll 3d6 dice and keep the highest 2",
        "Get location for cell tower MCC 310, MNC 404, LAC 1, CID 5632016"
    ]
    
    @classmethod
    def validate_environment(cls) -> bool:
        """Validate that required environment variables are set"""
        required_vars = ["OPENAI_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
            return False
        
        return True
    
    @classmethod
    def get_mermaid_diagram(cls) -> str:
        """Get the Mermaid diagram for the LangGraph workflow"""
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
