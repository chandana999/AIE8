#!/usr/bin/env python3
"""
Complete LangGraph Workflow with All Connected Tools
Custom tools integration with LangGraph workflow:
- Web search (Tavily API)
- Dice rolling (custom tool)
- Cell tower location (Unwired Labs API)
"""

import asyncio
import os
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model

load_dotenv()

class LangGraphMCP:
    """Minimal LangGraph MCP implementation"""
    
    def __init__(self):
        self.model = None
        self.tools = None
        self.graph = None
    
    async def initialize(self):
        """Initialize model, tools, and graph"""
        if not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY not found")
            return False
        
        # Setup model
        self.model = init_chat_model("openai:gpt-4o-mini")
        
        # Setup MCP client
        client = MultiServerMCPClient({
            "mcp-server": {
                "command": "uv",
                "args": ["run", "server.py"],
                "transport": "stdio",
            }
        })
        
        self.tools = await client.get_tools()
        print(f"[+] Found {len(self.tools)} tools: {[t.name for t in self.tools]}")
        
        # Build graph
        await self._build_graph()
        return True
    
    async def _build_graph(self):
        """Build the LangGraph workflow"""
        def call_model(state: MessagesState):
            response = self.model.bind_tools(self.tools).invoke(state["messages"])
            return {"messages": [response]}
        
        builder = StateGraph(MessagesState)
        builder.add_node("call_model", call_model)
        builder.add_node("tools", ToolNode(self.tools))
        
        builder.add_edge(START, "call_model")
        builder.add_conditional_edges("call_model", tools_condition)
        builder.add_edge("tools", "call_model")
        
        self.graph = builder.compile()
    
    
    async def run_query(self, query: str):
        """Run a single query through the graph"""
        result = await self.graph.ainvoke({"messages": [("user", query)]})
        return result["messages"][-1].content
    
    async def demo(self):
        """Run demo queries"""
        queries = [
            "What are the current trends in artificial intelligence?"
            "Roll 4d6 and drop the lowest",
            "Roll 1d4 for a quick decision",
            "Find cell tower location MCC 404, MNC 45, LAC 1234, CID 5678901"
        ]
        
        for query in queries:
            print(f"\n[>] {query}")
            response = await self.run_query(query)
            print(f"[<] {response}")
            await asyncio.sleep(1)

async def main():
    """Main function"""
    print("Simple LangGraph Demo - Uses ALL MCP Tools")
    
    app = LangGraphMCP()
    if not await app.initialize():
        return
    
    
    # Run demo
    print("\n Running Demo...")
    await app.demo()

if __name__ == "__main__":
    asyncio.run(main())
