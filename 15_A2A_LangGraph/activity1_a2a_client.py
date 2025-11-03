# 🏗️ Activity #1: Simple LangGraph Agent that makes API calls to Agent Node via A2A protocol

import asyncio
import logging
from uuid import uuid4
from typing import Dict, Any, Optional
import httpx
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest
from langchain_openai import ChatOpenAI

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for required API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not OPENAI_API_KEY:
    logger.warning("⚠️ OPENAI_API_KEY not found in .env file")
else:
    logger.info("✅ OPENAI_API_KEY loaded from .env")

if TAVILY_API_KEY:
    logger.info("✅ TAVILY_API_KEY loaded from .env")
else:
    logger.info("ℹ️ TAVILY_API_KEY not set (optional)")

# ============================================================================
# LangGraph Agent State
# ============================================================================

class AgentState(TypedDict):
    """State for the LangGraph client agent."""
    messages: Annotated[list[BaseMessage], add_messages]
    a2a_client: Optional[A2AClient]
    a2a_task_id: Optional[str]
    a2a_context_id: Optional[str]

# ============================================================================
# A2A Tool - Makes API calls to Agent Node via A2A protocol
# ============================================================================

def create_a2a_tool(server_url: str = "http://localhost:10000"):
    """Create a tool that makes API calls to the Agent Node through A2A protocol."""
    
    @tool
    def call_a2a_agent(query: str) -> str:
        """
        Call the remote Agent Node via A2A protocol.
        This tool makes API calls to the server agent running on the A2A server.
        
        Args:
            query: The question or request to send to the agent node
            
        Returns:
            The response from the agent node
        """
        async def _call_agent():
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as httpx_client:
                try:
                    # Initialize A2A client connection
                    resolver = A2ACardResolver(httpx_client=httpx_client, base_url=server_url)
                    agent_card = await resolver.get_agent_card()
                    client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                    
                    # Send message via A2A protocol
                    payload = {
                        'message': {
                            'role': 'user',
                            'parts': [{'kind': 'text', 'text': query}],
                            'message_id': uuid4().hex,
                        }
                    }
                    request = SendMessageRequest(
                        id=str(uuid4()), 
                        params=MessageSendParams(**payload)
                    )
                    
                    # Make API call to Agent Node
                    response = await client.send_message(request)
                    
                    # Extract response
                    if response.root and response.root.result:
                        result = response.root.result
                        if hasattr(result, 'message') and result.message:
                            return result.message.get('parts', [{}])[0].get('text', 'No response received')
                        return str(result)
                    return str(response)
                    
                except Exception as e:
                    return f"Error calling agent node: {str(e)}"
        
        # Run async function
        return asyncio.run(_call_agent())
    
    return call_a2a_agent

# ============================================================================
# Build Simple LangGraph Agent
# ============================================================================

def build_simple_agent(server_url: str = "http://localhost:10000"):
    """Build a simple LangGraph agent that uses A2A protocol to call the Agent Node."""
    
    # Create the A2A tool
    a2a_tool = create_a2a_tool(server_url)
    tools = [a2a_tool]
    
    # Create LLM with tools (automatically uses OPENAI_API_KEY from .env)
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment. Please set it in .env file")
    
    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.1)
    llm_with_tools = llm.bind_tools(tools)
    
    def call_model(state: AgentState) -> Dict[str, Any]:
        """Invoke the model with messages."""
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(state: AgentState):
        """Route to tools if the last message has tool calls."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return END
    
    # Build graph
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)
    
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    
    return graph.compile()

# ============================================================================
# Main Demo
# ============================================================================

async def run_a2a_client_demo():
    """Run a simple demonstration of the LangGraph agent with A2A tool."""
    
    logger.info("=" * 70)
    logger.info("🚀 Activity #1: Simple LangGraph Agent with A2A Protocol")
    logger.info("=" * 70)
    logger.info("\n📋 This agent uses A2A protocol to make API calls to the Agent Node\n")
    
    server_url = "http://localhost:10000"
    
    # Build the LangGraph agent
    agent = build_simple_agent(server_url)
    
    # Demo queries
    demos = [
        "What are the key features of large language models?",
        "Explain how RAG improves AI responses",
        "What is the difference between fine-tuning and prompt engineering?"
    ]
    
    for i, query in enumerate(demos, 1):
        logger.info(f"\n📋 Demo {i}: {query}")
        logger.info("-" * 70)
        
        try:
            # Run the LangGraph agent
            state = {"messages": [HumanMessage(content=query)]}
            config = {"configurable": {"thread_id": f"demo-{i}"}}
            
            final_response = None
            async for event in agent.astream(state, config, stream_mode="values"):
                if "messages" in event:
                    final_response = event
            
            if final_response and final_response["messages"]:
                last_msg = final_response["messages"][-1]
                logger.info(f"✅ Agent Response: {last_msg.content[:200]}...")
            
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            logger.error("Make sure the A2A server is running: uv run python -m app")
        
        await asyncio.sleep(1)
    
    logger.info("\n✅ Demo complete!\n")

if __name__ == "__main__":
    asyncio.run(run_a2a_client_demo())

