# Multi-Agent Log Analysis API

A FastAPI-based web service that provides intelligent log analysis using a multi-agent RAG system with LangGraph.

## Features

- **Multi-Agent Architecture**: LogSearch + LogAnalysisRAG + Supervisor agents
- **RAG System**: Retrieval Augmented Generation with Qdrant vector store
- **External Search**: Tavily search integration for up-to-date information
- **Streaming Responses**: Real-time analysis results
- **Log File Upload**: Batch processing of log files
- **Chat Interface**: Interactive log analysis

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment

Copy the example environment file:
```bash
cp .env.example .env
```

### 3. Prepare Data

Ensure you have log incident documents in `../data/web_incidents/` directory:
```
data/web_incidents/
├── incident1.md
├── incident2.md
└── ...
```

### 4. Run the API

```bash
python app.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check
- **GET** `/api/health` - Check API status and system initialization

### Basic Chat
- **POST** `/api/chat` - Basic OpenAI chat (no RAG)

### Log Analysis
- **POST** `/api/rag-chat` - Analyze log entries using multi-agent system
- **POST** `/api/upload-log-file` - Upload and process log files

## Usage Examples

### 1. Analyze Single Log Entry

```bash
curl -X POST "http://localhost:8000/api/rag-chat" \
  -H "Content-Type: application/json" \
  -d '{
    "log_input": "[error] AH01797: client denied by server configuration",
    "api_key": "your_openai_api_key"
  }'
```

### 2. Upload Log File

```bash
curl -X POST "http://localhost:8000/api/upload-log-file" \
  -F "file=@server.log" \
  -F "api_key=your_openai_api_key"
```

### 3. Basic Chat

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "developer_message": "You are a helpful assistant",
    "user_message": "Hello, how are you?",
    "api_key": "your_openai_api_key"
  }'
```

## Multi-Agent System

### Agents

1. **LogSearch Agent**
   - Searches external sources using Tavily
   - Handles unknown error codes
   - Provides up-to-date documentation

2. **LogAnalysisRAG Agent**
   - Uses internal knowledge base
   - Analyzes known Apache error codes
   - Provides detailed incident analysis

3. **Supervisor Agent**
   - Routes requests between agents
   - Decides which agent to use based on log type
   - Coordinates multi-agent collaboration

### Decision Logic

- **Use LogAnalysisRAG for:**
  - Known Apache error codes (AH01797, AH01084, etc.)
  - Security incidents
  - Pattern analysis with existing incidents

- **Use LogSearch for:**
  - Unknown error codes
  - Multiple error types
  - Infrastructure issues needing latest solutions

## Required API Keys

Users must provide these API keys through the frontend:

- **OPENAI_API_KEY**: For LLM and embeddings
- **TAVILY_API_KEY**: For external search functionality

## Data Requirements

The system expects log incident documents in markdown format in the `../data/web_incidents/` directory. Each document should contain:

- Incident description
- Error patterns
- Root cause analysis
- Remediation steps

## Architecture

```
Frontend → FastAPI → Multi-Agent System → RAG System
                     ↓
                 Knowledge Base (Qdrant)
                     ↓
                 External Search (Tavily)
```

## Development

### Running in Development Mode

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

## Dependencies

- FastAPI 0.104.1
- LangChain 0.3.27
- LangGraph 0.2.0
- OpenAI 1.3.0
- Qdrant Client 1.6.0
- Tavily Python 0.3.0

## License

This project is part of the AI Engineer Challenge.


