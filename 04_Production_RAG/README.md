<p align = "center" draggable=”false” ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719" 
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">Production RAG with LangGraph and LangChain</h1>

| 📰 Session Sheet | ⏺️ Recording     | 🖼️ Slides        | 👨‍💻 Repo         | 📝 Homework      | 📁 Feedback       |
|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| [Session 4: RAG with LangGraph, OSS Local Models, & Eval w/ LangSmith  ](https://www.notion.so/Session-4-Production-Grade-RAG-with-LangChain-and-LangSmith-26acd547af3d80838d5beba464d7e701) |[Recording!](https://us02web.zoom.us/rec/share/jEs9TS_re1f9X3y2T61Dgv_bEp6EmVzVkiYDOC-cEU8WA2tR5jMI1bwsn4L_Al1n.msDqlCRCROFBaRCH) (78y?PRTg) | [Session 4 Slides](https://www.canva.com/design/DAGzMO1y0FQ/oJaw4HMIFecP3oX9jSO4fw/edit?utm_content=DAGzMO1y0FQ&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) | You are here! | [Session 4 Assignment: Prod. RAG](https://forms.gle/i2SdxgWX4ahFwNrCA) | [AIE8 Feedback 9/18](https://forms.gle/ymYqK5MBLAG11jDB9)

# Build 🏗️

If running locally:

1. `uv sync`
2. **NEW**: Set up Ollama by running the `Ollama_Setup_and_Testing.ipynb` notebook first to verify your local LLM setup
3. Open the main assignment notebook
4. Select the venv created by `uv sync` as your kernel

## Setting up Ollama (Local LLM)

Before starting the main assignment, run the `Ollama_Setup_and_Testing.ipynb` notebook to:
- Verify Ollama is installed and running
- Test embeddings with LangChain connectors
- Test model inference with LangChain connectors
- Ensure all models are properly downloaded

Run the preparation notebook and complete the contained tasks:

- 🤝 Breakout Room #1:
    1. Install and run Ollama
    2. Ensure all the required models are pulled
    3. Test them!

Next, run the Assignment notebook and complete the contained tasks:

- 🤝 Breakout Room #2:
    1. LangChain and LCEL Concepts
    2. Understanding States and Nodes
    3. Introduction to QDrant Vector Databases
    4. Building a Basic Graph

# Ship 🚢

- The completed notebook. 
- 5min. Loom Video

# Share 🚀
- Walk through your notebook and explain what you've completed in the Loom video
- Make a social media post about your final application and tag @AIMakerspace
- Share 3 lessons learned
- Share 3 lessons not learned

# Submitting Your Homework

Follow these steps to prepare and submit your homework:
1. Create a branch of your `AIE7` repo to track your changes. Example command: `git checkout -b s04-assignment`
2. Responding to the activities and questions inline in the `Assignment_Introduction_to_LCEL_and_LangGraph_LangChain_Powered_RAG.ipynb`

> _NOTE on the `Assignment_Introduction_to_LCEL_and_LangGraph_LangChain_Powered_RAG` notebook: You will also need to enter your response to Question #1 in the code cell directly below it which contains this line of code:_
    ```
    embedding_dim =  # YOUR ANSWER HERE
    ```
