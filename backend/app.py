from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
from openai import OpenAI
import os
import tempfile
import asyncio
import re
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# Import your existing multi-agent system components
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph, END
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from typing_extensions import List, TypedDict
import functools
import operator

# Import agent components
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_community.tools.tavily_search import TavilySearchResults

# Load environment variables
load_dotenv()

# Initialize FastAPI application
app = FastAPI(title="Multi-Agent Log Analysis API")

# Global variables for your multi-agent system
compiled_log_analyzer = None
compiled_log_analysis_graph = None

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security scheme for API key validation
security = HTTPBearer(auto_error=False)

# API Key validation function
def validate_api_key(api_key: str) -> bool:
    """Validate OpenAI API key format"""
    if not api_key or not api_key.strip():
        return False
    # OpenAI API keys start with 'sk-' and are typically 51 characters long
    return bool(re.match(r'^sk-[A-Za-z0-9]{48}$', api_key.strip()))

def validate_tavily_api_key(api_key: str) -> bool:
    """Validate Tavily API key format"""
    if not api_key or not api_key.strip():
        return False
    # Tavily API keys are typically 32 characters long
    return bool(re.match(r'^tvly-[A-Za-z0-9]{32}$', api_key.strip()))

def test_api_key_validity(api_key: str) -> bool:
    """Test if API key is valid by making a simple API call"""
    try:
        client = OpenAI(api_key=api_key)
        # Make a minimal test call
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        return True
    except Exception:
        return False

# Data models - simple like reference code
class ChatRequest(BaseModel):
    developer_message: str
    user_message: str
    model: Optional[str] = "gpt-4o-mini"
    api_key: Optional[str] = None

class LogAnalysisRequest(BaseModel):
    log_input: str
    api_key: Optional[str] = None
    model: Optional[str] = "gpt-4o-mini"
    tavily_api_key: Optional[str] = None

# Removed APIKeyRequest model - keeping it simple

class LogUploadResponse(BaseModel):
    success: bool
    message: str
    chunks_count: Optional[int] = None
    analysis_result: Optional[str] = None

# Only the API keys you're actually using
required_vars = [
    "OPENAI_API_KEY",
    "TAVILY_API_KEY"
]

def check_environment():
    """Check if required API keys are set"""
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required API keys: {missing_vars}")
        return False
    
    print("✅ Required API keys are set")
    return True

# Agent helper functions from your notebook
def log_agent_node(state, agent, name):
    """Helper function to wrap log analysis agents into LangGraph nodes."""
    result = agent.invoke(state)
    
    if "output" in result:
        output = result["output"]
    else:
        output = str(result)
    
    return {"messages": [HumanMessage(content=output, name=name)]}

def create_log_analysis_agent(llm: ChatOpenAI, tools: list, system_prompt: str) -> str:
    """Create a function-calling agent and add it to the graph."""
    system_prompt += ("\nWork autonomously according to your specialty, using the tools available to you."
    " Do not ask for clarification."
    " Your other team members (and other teams) will collaborate with you with their own specialties."
    " You are chosen for a reason!")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_functions_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def create_log_analysis_supervisor(llm: ChatOpenAI, system_prompt, members) -> str:
    """An LLM-based router for log analysis."""
    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [{"enum": options}],
                },
            },
            "required": ["next"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), team_members=", ".join(members))
    return (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )

# Multi-agent system state
class LogAnalysisTeamState(TypedDict):
    messages: List[BaseMessage]
    team_members: List[str]
    next: str

# Initialize your complete multi-agent RAG system
def initialize_multi_agent_system(openai_api_key: str, tavily_api_key: str):
    """Initialize your complete multi-agent system with provided API keys"""
    global compiled_log_analyzer, compiled_log_analysis_graph
    
    try:
        # Set API keys for this session
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["TAVILY_API_KEY"] = tavily_api_key
        
        print(f"✅ API keys set for multi-agent system initialization")
        
        # Load your existing knowledge base using the same approach as notebook
        print("📚 Loading knowledge base documents...")
        
        # Custom TextLoader that handles Unicode properly (same as notebook)
        class UnicodeTextLoader(TextLoader):
            def __init__(self, file_path: str, encoding: str = "utf-8"):
                super().__init__(file_path, encoding=encoding)
        
        # Load documents with proper Unicode handling (same as notebook)
        directory_loader = DirectoryLoader("../data/web_incidents", glob="**/*.md", loader_cls=UnicodeTextLoader)
        all_knowledge_documents = directory_loader.load()
        print(f"✅ Successfully loaded {len(all_knowledge_documents)} knowledge documents")
        
        # Text splitting
        import tiktoken
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        def tiktoken_len(text):
            tokens = tiktoken.encoding_for_model("gpt-4o").encode(text)
            return len(tokens)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=750,
            chunk_overlap=0,
            length_function=tiktoken_len,
        )
        
        all_knowledge_chunks = text_splitter.split_documents(all_knowledge_documents)
        
        # Embedding model and vector store
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        qdrant_vectorstore = Qdrant.from_documents(
            documents=all_knowledge_chunks,
            embedding=embedding_model,
            location=":memory:"
        )
        qdrant_retriever = qdrant_vectorstore.as_retriever()
        
        # Your existing RAG system
        HUMAN_TEMPLATE = """
        You are a senior SRE and AI-powered log analyst specializing in distributed systems.

        Your goal is to produce a **causally accurate, time-aware** incident analysis from the given logs and contextual information.

        ---

        ### INPUT DATA

        **Raw Log Entries:**
        {query}

        **Historical Context / Similar Incidents:**
        {context}

        ---

        ### INSTRUCTIONS

        Carefully read the logs and infer the following in **detail and specificity**:

        1. **Error Categorization & Severity**
           - Group related log lines (e.g., 403, 502, 504, 500).
           - For each group, specify affected endpoints, probable systems, and severity level (Low / Medium / High).

        2. **Event Timeline**
           - Reconstruct a step-by-step chronological flow of events.
           - Identify how early errors (e.g., SSL expiry) led to later cascading failures.

        3. **Root Cause Analysis**
           - Explain the causal chain clearly (e.g., "SSL handshake failure → backend queue buildup → timeouts → 503 overload").
           - Highlight both *primary* and *secondary* root causes.

        4. **Immediate Remediation Actions**
           - Describe exact operational steps (e.g., "renew SSL cert," "flush retry queue," "restart backend node i-123…").
           - Keep these concise, executable, and system-aware.

        5. **Preventive Measures**
           - Suggest long-term or architectural changes (e.g., circuit breakers, monitoring alerts, auto-renewal).
           - Reference insights from {context} where relevant.

        ---

        ### OUTPUT FORMAT

        Respond in Markdown with clearly labeled sections:

        #### 🧠 Incident Summary  
        (Brief 3–5 sentence overview)

        #### 🕒 Event Timeline  
        (Bullet-point chronological chain)

        #### ⚙️ Root Cause  
        (Primary → Secondary → Tertiary, with causal arrows)

        #### 🚑 Remediation Steps  
        (Bullet list of actionable tasks)

        #### 🧱 Preventive Recommendations  
        (Specific improvements tied to architecture or configuration)

        ---

        Ensure your reasoning is **concrete, system-specific, and chronologically coherent**. Avoid generic advice.
        """
        
        chat_prompt = ChatPromptTemplate.from_messages([("human", HUMAN_TEMPLATE)])
        generator_llm = ChatOpenAI(model="gpt-4o-mini")
        
        # Your existing LangGraph RAG system
        class LogAnalysisState(TypedDict):
            log_input: str
            context: List[Document]
            analysis_result: str

        def retrieve_log_context(state: LogAnalysisState):
            retrieved_docs = qdrant_retriever.invoke(state["log_input"])
            return {"context": retrieved_docs}

        def analyze_log(state: LogAnalysisState):
            generator_chain = chat_prompt | generator_llm | StrOutputParser()
            analysis_result = generator_chain.invoke({"query": state["log_input"], "context": state["context"]})
            return {"analysis_result": analysis_result}

        log_analysis_graph = StateGraph(LogAnalysisState).add_sequence([retrieve_log_context, analyze_log])
        log_analysis_graph.add_edge(START, "retrieve_log_context")
        compiled_log_analyzer = log_analysis_graph.compile()
        
        # Initialize multi-agent system
        log_analysis_llm = ChatOpenAI(model="gpt-4o-mini")
        
        # Create tools
        # Initialize Tavily tool with proper API key (only needed for LogSearch agent)
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        print(f"🔑 Tavily API Key from environment: {tavily_api_key[:10] if tavily_api_key else 'None'}...")
        print(f"🔑 Tavily API Key status: {'✅ Set' if tavily_api_key and tavily_api_key != 'tvly-placeholder-key' else '❌ Not set or placeholder'}")
        
        try:
            if tavily_api_key and tavily_api_key != "tvly-placeholder-key":
                tavily_tool = TavilySearchResults(max_results=5)
                print("✅ Tavily search tool initialized successfully for LogSearch agent")
            else:
                print("⚠️ No valid Tavily API key found, creating fallback tool for LogSearch agent")
                from langchain.tools import tool
                @tool
                def fallback_search_tool(query: str) -> str:
                    """Fallback search tool when Tavily API is unavailable."""
                    return f"External search unavailable (no API key). Query was: {query}. Please provide analysis based on general web server knowledge."
                tavily_tool = fallback_search_tool
        except Exception as e:
            print(f"❌ Failed to initialize Tavily search tool: {e}")
            # Create a fallback tool that doesn't require external API
            from langchain.tools import tool
            @tool
            def fallback_search_tool(query: str) -> str:
                """Fallback search tool when Tavily API fails to initialize."""
                return f"External search unavailable. Query was: {query}. Please provide analysis based on general web server knowledge."
            tavily_tool = fallback_search_tool
        
        # Custom RAG tool
        from langchain_core.tools import tool
        from typing import Annotated
        
        @tool
        def analyze_web_logs(query: Annotated[str, "web server log entries to analyze for incidents and remediation"]):
            """Use Retrieval Augmented Generation to analyze web server logs and provide incident insights"""
            return compiled_log_analyzer.invoke({"log_input": query})
        
        # Create agents
        log_search_agent = create_log_analysis_agent(
            llm=log_analysis_llm,
            tools=[tavily_tool],
            system_prompt="""You are a specialized web log search assistant. Your role is to search for up-to-date information, documentation, and solutions related to web server errors, security vulnerabilities, and performance issues using external search engines.

IMPORTANT: Always try to use the search tool first for any query. If the search tool is available and working, use it to find up-to-date information about the error codes and issues mentioned in the logs.

If external search fails or is unavailable (due to API key issues), provide general guidance based on your knowledge about:
- Common web server error patterns
- Standard Apache error codes and their meanings
- General troubleshooting approaches for unknown errors
- Best practices for log analysis and incident response

Always provide helpful analysis even when external search resources are unavailable. The LogAnalysisRAG agent will handle detailed analysis using our internal knowledge base."""
        )
        
        rag_agent = create_log_analysis_agent(
            llm=log_analysis_llm,
            tools=[analyze_web_logs],
            system_prompt="You are a specialized log analysis assistant who can provide detailed information about web server incidents, errors, and remediation steps using our knowledge base."
        )
        
        # Create supervisor
        log_analysis_supervisor_agent = create_log_analysis_supervisor(
            log_analysis_llm,
            """You are a supervisor coordinating log analysis experts.

            Available agents:
            - LogSearch: Searches for up-to-date information about web server errors using external search engines
            - LogAnalysisRAG: Analyzes logs using our internal knowledge base for detailed incident analysis

            Decision logic based on error types:
            
            Use LogAnalysisRAG for:
            - Known Apache error codes: AH01797 (403 Forbidden), AH01084/AH01085 (502 Bad Gateway), 
              AH01078/AH00485 (503 Service Unavailable), AH01079 (504 Gateway Timeout), 
              AH01961/AH02032/AH01976 (SSL errors)
            - Security incidents with sensitive files (admin/, config/, backup/)
            - Pattern analysis and correlation with existing incidents
            - Root cause analysis using our incident knowledge base
            
            Use LogSearch for:
            - Unknown or new error codes not in our knowledge base
            - Multiple different error types requiring external documentation
            - System-wide infrastructure issues needing latest solutions
            - Errors not covered in our internal Apache incident documentation
            
            If analysis is complete, choose FINISH.
            
            Always return a valid routing decision.""",
            ["LogSearch", "LogAnalysisRAG"],
        )
        
        # Create nodes
        log_search_node = functools.partial(log_agent_node, agent=log_search_agent, name="LogSearch")
        rag_node = functools.partial(log_agent_node, agent=rag_agent, name="LogAnalysisRAG")
        
        # Create multi-agent graph
        log_analysis_graph = StateGraph(LogAnalysisTeamState)
        log_analysis_graph.add_node("LogSearch", log_search_node)
        log_analysis_graph.add_node("LogAnalysisRAG", rag_node)
        log_analysis_graph.add_node("LogAnalysisSupervisor", log_analysis_supervisor_agent)
        
        log_analysis_graph.add_edge("LogSearch", "LogAnalysisSupervisor") 
        log_analysis_graph.add_edge("LogAnalysisRAG", "LogAnalysisSupervisor")
        log_analysis_graph.add_conditional_edges(
            "LogAnalysisSupervisor",
            lambda x: x["next"],
            {"LogSearch": "LogSearch", "LogAnalysisRAG": "LogAnalysisRAG", "FINISH": END},
        )
        log_analysis_graph.set_entry_point("LogAnalysisSupervisor")
        
        compiled_log_analysis_graph = log_analysis_graph.compile()
        
        print("✅ Complete multi-agent system initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing multi-agent system: {e}")
        import traceback
        traceback.print_exc()
        return False

# Remove startup initialization - we'll initialize on first request with API keys

# Removed API key validation endpoint - keeping it simple like reference code

# Basic chat endpoint - simple like reference code
@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        # Set environment variable for OpenAI API key (like reference code)
        os.environ["OPENAI_API_KEY"] = request.api_key
        
        # Create OpenAI client
        client = OpenAI(api_key=request.api_key)
        
        async def generate():
            stream = client.chat.completions.create(
                model=request.model,
                messages=[
                    {"role": "developer", "content": request.developer_message},
                    {"role": "user", "content": request.user_message}
                ],
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        return StreamingResponse(generate(), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

# Log file upload endpoint - simple like reference code
@app.post("/api/upload-log-file", response_model=LogUploadResponse)
async def upload_log_file(
    file: UploadFile = File(...),
    api_key: str = Form(...),
    tavily_api_key: Optional[str] = Form(None)
):
    """Upload and process a log file using multi-agent system."""
    try:
        # Set environment variables for API keys (like reference code)
        os.environ["OPENAI_API_KEY"] = api_key
        if tavily_api_key:
            os.environ["TAVILY_API_KEY"] = tavily_api_key
        
        if not file.filename.lower().endswith(('.log', '.txt')):
            raise HTTPException(status_code=400, detail="File must be a log file (.log or .txt)")
        
        # Read and validate file content
        content = await file.read()
        log_content = content.decode('utf-8')
        
        # Basic content validation - check for log-like patterns
        log_indicators = [
            r'\d{4}-\d{2}-\d{2}',  # Date patterns (YYYY-MM-DD)
            r'\d{2}:\d{2}:\d{2}',  # Time patterns (HH:MM:SS)
            r'\[(error|warn|info|debug|ERROR|WARN|INFO|DEBUG)\]',  # Log levels
            r'AH\d{5}',  # Apache error codes
            r'HTTP/\d\.\d',  # HTTP versions
            r'\d{3}\s',  # HTTP status codes
            r'GET|POST|PUT|DELETE',  # HTTP methods
            r'client\s+\d+\.\d+\.\d+\.\d+',  # Client IP patterns
        ]
        
        # Count how many log indicators are found
        import re
        found_indicators = sum(1 for pattern in log_indicators if re.search(pattern, log_content, re.IGNORECASE))
        
        # If less than 2 indicators found, it's probably not a log file
        if found_indicators < 2:
            raise HTTPException(
                status_code=400, 
                detail=f"File '{file.filename}' does not appear to be a log file. Found {found_indicators} log indicators. Please upload Apache, Nginx, or other web server log files with timestamps, error codes, or HTTP requests."
            )
        
        # Reset file pointer for processing
        await file.seek(0)
        
        # Initialize multi-agent system if not already initialized
        if not compiled_log_analysis_graph:
            print("🚀 Initializing multi-agent system for log upload...")
            tavily_key = tavily_api_key or os.getenv("TAVILY_API_KEY") or "tvly-placeholder-key"
            if not initialize_multi_agent_system(api_key, tavily_key):
                raise HTTPException(status_code=500, detail="Failed to initialize multi-agent system")
        
        # Log content already read for validation above
        
        print(f"📝 Processing log file '{file.filename}' with multi-agent system...")
        print(f"📊 Log file size: {len(log_content)} characters")
        
        # Process entire log file content with multi-agent system
        print(f"🤖 Sending entire log file to multi-agent system...")
        
        # Use your multi-agent system with the full log content
        def enter_log_analysis_chain(log_input: str):
            return {"messages": [HumanMessage(content=log_input)]}
        
        log_analysis_chain = enter_log_analysis_chain | compiled_log_analysis_graph
        
        # Stream through the multi-agent system with full log content
        final_result = ""
        agent_count = 0
        
        print(f"🔄 Streaming full log file through multi-agent system...")
        for s in log_analysis_chain.stream(log_content, {"recursion_limit": 10}):
            print(f"📊 Multi-agent step: {list(s.keys())}")
            
            if "__end__" not in s:
                # Collect results from each agent
                for agent_name, agent_result in s.items():
                    if agent_name != "LogAnalysisSupervisor":
                        agent_count += 1
                        agent_output = agent_result['messages'][0].content
                        print(f"✅ {agent_name} analyzed log file ({len(agent_output)} chars)")
                        final_result += f"\n=== {agent_name} Analysis ===\n{agent_output}\n"
        
        print(f"🎯 Log file processing complete! {agent_count} agents analyzed the file")
        print(f"📋 FINAL ANALYSIS RESULT:")
        print("=" * 80)
        print(final_result)
        print("=" * 80)
        print(f"📊 Analysis length: {len(final_result)} characters")
        
        # Store the analysis result
        analysis_result = {
            "filename": file.filename,
            "content_preview": log_content[:200] + "..." if len(log_content) > 200 else log_content,
            "full_analysis": final_result.strip(),
            "agents_used": list(s.keys()) if 's' in locals() and s else []
        }
        
        return LogUploadResponse(
            success=True,
            message=f"Log file '{file.filename}' processed successfully with multi-agent system. {agent_count} agents analyzed the content.",
            chunks_count=1,  # Single analysis of the entire file
            analysis_result=final_result.strip()  # Return the actual analysis
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing log file: {str(e)}")

# Multi-agent RAG chat endpoint - simple like reference code
@app.post("/api/rag-chat")
async def rag_chat(request: LogAnalysisRequest):
    """Analyze log entries using your complete multi-agent system."""
    try:
        # Set environment variables for API keys (like reference code)
        os.environ["OPENAI_API_KEY"] = request.api_key
        if request.tavily_api_key:
            os.environ["TAVILY_API_KEY"] = request.tavily_api_key
        
        # Initialize multi-agent system if not already initialized
        if not compiled_log_analysis_graph:
            print("🚀 Initializing multi-agent system with provided API keys...")
            tavily_key = request.tavily_api_key or os.getenv("TAVILY_API_KEY") or "tvly-placeholder-key"
            if not initialize_multi_agent_system(request.api_key, tavily_key):
                raise HTTPException(status_code=500, detail="Failed to initialize multi-agent system")
        
        # Use your multi-agent system
        print(f"🤖 Starting multi-agent analysis for: {request.log_input[:100]}...")
        
        def enter_log_analysis_chain(log_input: str):
            return {"messages": [HumanMessage(content=log_input)]}
        
        log_analysis_chain = enter_log_analysis_chain | compiled_log_analysis_graph
        
        # Create streaming response
        async def generate():
            final_result = ""
            agent_count = 0
            
            print(f"🔄 Streaming through multi-agent system...")
            for s in log_analysis_chain.stream(request.log_input, {"recursion_limit": 10}):
                print(f"📊 Multi-agent step: {list(s.keys())}")
                
                if "__end__" not in s:
                    for agent_name, agent_result in s.items():
                        if agent_name != "LogAnalysisSupervisor":
                            agent_count += 1
                            agent_output = agent_result['messages'][0].content
                            print(f"✅ {agent_name} completed analysis ({len(agent_output)} chars)")
                            final_result += f"\n=== {agent_name} Analysis ===\n{agent_output}\n"
                            yield f"\n=== {agent_name} Analysis ===\n{agent_output}\n"
            
            print(f"🎯 Multi-agent analysis complete! {agent_count} agents processed the request")
            print(f"📋 FINAL ANALYSIS RESULT:")
            print("=" * 80)
            print(final_result)
            print("=" * 80)
            print(f"📊 Analysis length: {len(final_result)} characters")
            
            if not final_result:
                print("⚠️ No analysis results from multi-agent system")
                yield "No analysis results available"
        
        return StreamingResponse(generate(), media_type="text/plain")
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in multi-agent RAG chat: {str(e)}")

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint with system status"""
    return {"status": "ok"}

# Entry point
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
