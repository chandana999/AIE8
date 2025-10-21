# LogAnalyzer Frontend

A modern, responsive web interface for Apache log analysis powered by AI.

## Features

- **Apache Log Upload**: Upload Apache2 or Apache web logs for analysis
- **Multi-Agent Analysis**: Uses LangGraph multi-agent system for comprehensive log analysis
- **Real-time Streaming**: Get instant, streaming responses from AI agents
- **Security Insights**: Identify security threats and vulnerabilities
- **Error Analysis**: Find and analyze critical errors and performance issues
- **Quick Questions**: Pre-built questions for common log analysis tasks

## Quick Start

### Prerequisites

- Node.js 18+ 
- npm or yarn
- Backend API running on port 8000

### Installation

```bash
# Install dependencies
npm install

# Run development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the application.

### Configuration

The frontend connects to your backend API at `http://localhost:8000` by default. 

To change the API URL, set the environment variable:
```bash
NEXT_PUBLIC_API_URL=http://your-api-url:port
```

## Usage

1. **Enter API Keys**: Add your OpenAI API key (required) and Tavily API key (optional)
2. **Select Log Type**: Choose between Apache2 or Apache Web logs
3. **Upload Log File**: Upload your log file (.log or .txt)
4. **Start Analysis**: Use the chat interface or quick questions to analyze your logs

## API Integration

The frontend integrates with your FastAPI backend endpoints:

- `POST /api/upload-log-file` - Upload and process log files
- `POST /api/rag-chat` - Multi-agent log analysis
- `POST /api/chat` - General chat
- `GET /api/health` - Health check

## Tech Stack

- **Next.js 14** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Lucide React** - Icons
- **FastAPI** - Backend integration

## Development

```bash
# Run development server
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Run linting
npm run lint
```

## Project Structure

```
frontend/
├── app/
│   ├── globals.css      # Global styles
│   ├── layout.tsx       # Root layout
│   └── page.tsx         # Main application page
├── package.json         # Dependencies
├── tailwind.config.js   # Tailwind configuration
├── next.config.js       # Next.js configuration
└── README.md           # This file
```

## License

MIT License


