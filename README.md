<p align = "center" draggable=”false” ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719" 
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">AI Makerspace: MCP Session Repo for Session 13</h1>

This project is a demonstration of the MCP (Model Context Protocol) server, which utilizes the Tavily API for web search capabilities. The server is designed to run in a standard input/output (stdio) transport mode.

## Project Overview

The MCP server is set up to handle web search queries using the Tavily API. It is built with the following key components:

- **TavilyClient**: A client for interacting with the Tavily API to perform web searches.

## Prerequisites

- Python 3.13 or higher
- A valid Tavily API key

## ⚠️NOTE FOR WINDOWS:⚠️

You'll need to install this on the *Windows* side of your OS. 

This will require getting two CLI tool for Powershell, which you can do as follows:

- `winget install astral-sh.uv`
- `winget install --id Git.Git -e --source winget`

After you have those CLI tools, please open Cursor *into Windows*.

Then, you can clone the repository using the following command in your Cursor terminal:

```bash
git clone https://AI-Maker-Space/AIE8-MCP-Session.git
```

After that, you can follow from Step 2. below!

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Configure environment variables**:
Copy the `.env.sample` to `.env` and add your Tavily API key:
   ```
   TAVILY_API_KEY=your_api_key_here
   ```

3. 🏗️ **Add a new tool to your MCP Server** 🏗️

Create a new tool in the `server.py` file, that's it!

## Running the MCP Server

To start the MCP server, you will need to add the following to your MCP Profile in Cursor:

> NOTE: To get to your MCP config. you can use the Command Pallete (CMD/CTRL+SHIFT+P) and select "View: Open MCP Settings" and replace the contents with the JSON blob below.

```
{
    "mcpServers":  {
        "mcp-server": {
            "command" : "uv",
            "args" : ["--directory", "/PATH/TO/REPOSITORY", "run", "server.py"]
        }
    }
}
```

The server will start and listen for commands via standard input/output.

## Usage

The server provides a `web_search` tool that can be used to search the web for information about a given query. This is achieved by calling the `web_search` function with the desired query string.

## Activities: 

There are a few activities for this assignment!

### 🏗️ Activity #1: Build MCP Server

Choose an API that you enjoy using - and build an MCP server for it!

**Code Location**: `server.py` - Contains the MCP server implementation with custom tools

### 🏗️ Activity #2: LangGraph Integration

Build a simple LangGraph application that interacts with your MCP Server.

**Code Location**: `langgraph_tools.py` - Contains the LangGraph workflow that uses MCP tools

You can find more details [here](https://github.com/langchain-ai/langchain-mcp-adapters)!

## Custom Tools Available

This MCP server includes several custom tools that demonstrate different types of functionality:

### 📡 **Cell Tower Location Tool (`get_cell_location`)**

The `get_cell_location` tool uses the **Unwired Labs API** to find the approximate geographic location of a cell tower based on cellular network parameters.

**What it does:**
- Takes cellular network identifiers as input:
  - **MCC** (Mobile Country Code) - Identifies the country
  - **MNC** (Mobile Network Code) - Identifies the carrier/network
  - **LAC** (Location Area Code) - Identifies the location area
  - **CID** (Cell ID) - Identifies the specific cell tower
- Returns the approximate location including:
  - **Latitude and Longitude** coordinates
  - **Accuracy radius** in meters
  - **Human-readable address** of the location

**Example Usage:**
```
Find cell tower location MCC 404, MNC 45, LAC 1234, CID 5678901
```

**Output:**
```
- Latitude: 16.906413
- Longitude: 76.142876  
- Accuracy: 44,521 meters
- Address: Somapura, Sindagi taluku, Vijayapura, Karnataka, 586115, India
```

**Use Cases:**
- Network troubleshooting and optimization
- Location-based services development
- Emergency services coordination
- Network coverage analysis

**API Provider:** [Unwired Labs](https://unwiredlabs.com/) - Provides cell tower location services
