# 🔧 Multi-Agent Log Analyzer - Backend

FastAPI backend for the Multi-Agent Log Analyzer system, featuring intelligent routing between specialized agents for comprehensive Apache log analysis.

## 🏗️ **Architecture**

The backend implements a sophisticated multi-agent system using LangGraph:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Multi-Agent    │    │   Knowledge     │
│   Endpoints     │◄──►│   System         │◄──►│   Base          │
│                 │    │                  │    │                 │
│ • File Upload   │    │ • Supervisor     │    │ • Incident Docs │
│ • RAG Chat      │    │ • LogSearch      │    │ • Error Patterns│
│ • Health Check  │    │ • LogAnalysisRAG │    │ • Solutions     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🧠 **Multi-Agent System**

### **Supervisor Agent**
- **Role**: Intelligent routing coordinator
- **Function**: Analyzes log input and routes to appropriate specialist
- **Routing Logic**:
  - Known Apache errors → LogAnalysisRAG
  - Unknown errors → LogSearch
  - Multiple error types → LogSearch

### **LogSearch Agent**
- **Role**: External research specialist
- **Tools**: Tavily search integration
- **Use Case**: Unknown error codes, new issues, external documentation
- **Output**: Web search results with current solutions

### **LogAnalysisRAG Agent**
- **Role**: Internal knowledge specialist
- **Tools**: Curated incident knowledge base
- **Use Case**: Known Apache errors, documented incidents
- **Output**: Detailed analysis from internal knowledge

## 🚀 **API Endpoints**

### **POST /api/upload-log-file**
Upload and analyze log files using the multi-agent system.

**Request:**
- `file`: Log file (.log or .txt)
- `api_key`: OpenAI API key
- `tavily_api_key`: Tavily API key (optional)

**Response:**
```json
{
  "success": true,
  "message": "Log file processed successfully",
  "chunks_count": 1,
  "analysis_result": "Multi-agent analysis output"
}
```

### **POST /api/rag-chat**
Direct chat interface with the multi-agent system.

**Request:**
```json
{
  "log_input": "Log entries to analyze",
  "model": "gpt-4o-mini",
  "api_key": "your-openai-key",
  "tavily_api_key": "your-tavily-key"
}
```

**Response:** Streaming text response with analysis

### **GET /api/health**
Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

## 🛠️ **Installation**

### **Prerequisites**
- Python 3.11+
- OpenAI API Key
- Tavily API Key (optional)

### **Setup**
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables (optional)
export OPENAI_API_KEY="your-openai-key"
export TAVILY_API_KEY="your-tavily-key"

# Run the application
python app.py
```

The server will start on `http://localhost:8000`

## 📋 **Dependencies**

### **Core Framework**
- `fastapi==0.104.1` - Web framework
- `uvicorn==0.24.0` - ASGI server

### **AI/ML Libraries**
- `langchain==0.3.27` - LangChain framework
- `langchain-community==0.3.30` - Community integrations
- `langchain-core==0.3.78` - Core LangChain components
- `langchain-openai==0.3.35` - OpenAI integration
- `langgraph==0.2.56` - Multi-agent framework

### **Evaluation & Search**
- `ragas==0.2.5` - RAG evaluation framework
- `tavily-python==0.3.3` - External search API

### **Vector Storage**
- `qdrant-client==1.7.3` - Vector database client

### **Utilities**
- `pydantic>=2.7.4` - Data validation
- `python-multipart==0.0.6` - File upload support

## 🔧 **Configuration**

### **Environment Variables**
```bash
# Required
OPENAI_API_KEY=your-openai-api-key

# Optional
TAVILY_API_KEY=your-tavily-api-key
PORT=8000
```

### **Knowledge Base**
The system uses a curated knowledge base located in `../data/web_incidents/` containing:
- Apache 403 Forbidden incidents
- Apache 502 Bad Gateway incidents
- Apache 503 Service Unavailable incidents
- Apache 504 Gateway Timeout incidents
- Apache SSL/TLS incidents
- Apache 500 Internal Server Error incidents

## 🎯 **File Validation**

The system includes intelligent file validation:

### **Extension Validation**
- Accepts: `.log`, `.txt` files
- Rejects: Other file types

### **Content Validation**
Validates log content using pattern matching:
- Date patterns (YYYY-MM-DD)
- Time patterns (HH:MM:SS)
- Log levels (error, warn, info, debug)
- Apache error codes (AH#####)
- HTTP versions and status codes
- HTTP methods (GET, POST, PUT, DELETE)
- Client IP patterns

Files with fewer than 2 log indicators are rejected with detailed error messages.

## 🔄 **Multi-Agent Workflow**

1. **Input Processing**: Log file or text input received
2. **Supervisor Analysis**: Determines routing strategy
3. **Agent Selection**: Routes to LogSearch or LogAnalysisRAG
4. **Analysis Execution**: Specialized agent processes the input
5. **Result Compilation**: Combines analysis results
6. **Response Streaming**: Returns formatted analysis

## 🚀 **Deployment**

### **Render Deployment**
1. Connect GitHub repository to Render
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Set environment variables
5. Deploy

### **Environment Variables for Production**
```bash
PORT=10000
OPENAI_API_KEY=your-production-openai-key
TAVILY_API_KEY=your-production-tavily-key
```

## 📊 **Monitoring & Logging**

The application includes comprehensive logging:
- Multi-agent system initialization
- API key validation status
- File upload and validation
- Agent routing decisions
- Analysis progress and results
- Error handling and debugging

## 🧪 **Testing**

### **API Testing**
```bash
# Health check
curl http://localhost:8000/api/health

# Upload test file
curl -X POST "http://localhost:8000/api/upload-log-file" \
  -F "file=@sample.log" \
  -F "api_key=your-openai-key"
```

### **Interactive API Documentation**
Visit `http://localhost:8000/docs` for interactive Swagger UI documentation.

## 🔒 **Security**

- API key validation and secure storage
- File type and content validation
- CORS configuration for frontend integration
- Input sanitization and error handling

## 📈 **Performance**

- Streaming responses for real-time feedback
- Efficient multi-agent routing
- Optimized knowledge base retrieval
- Caching for improved response times

---

**Built with FastAPI, LangChain, and LangGraph for production-ready log analysis**