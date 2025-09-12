<p align = "center" draggable=”false” ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719" 
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">Session 2: Embeddings and RAG</h1>

### [Quicklinks](https://github.com/AI-Maker-Space/AIE7/tree/main/00_AIM_Quicklinks)

| 📰 Session Sheet | ⏺️ Recording     | 🖼️ Slides        | 👨‍💻 Repo         | 📝 Homework      | 📁 Feedback       |
|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| [Session 2: 🗃️ Context Engineering I: RAG](https://www.notion.so/Session-2-Context-Engineering-I-Retrieval-Augmented-Generation-RAG-26acd547af3d8041a75bfa162d1ab600) |[Recording!](https://us02web.zoom.us/rec/share/cDb3c4VCrHUACpHdOl1C0u092-_GOSBb3emPfeuwZDAtUV3FTdBVMt6DGSzkU6gQ.N6_xNZ9FOiG4SYyB) (GZm4&jx*) | [Session 2 Slides](https://www.canva.com/design/DAGrSqNpTZ0/Ra4psYmdCwBeL8o1-L69WA/edit?utm_content=DAGrSqNpTZ0&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) | You are here! | [Session 2 Assignment: RAG](https://forms.gle/98s98cdS8vt1m5g67) | [AIE8 Feedback 9/11](https://forms.gle/hGJjwZLDm6Mcb2Ki7)


### Outline:

🤜 BREAKOUT ROOM #1:
- Task 1: Imports and Utilities
- Task 2: Documents
- Task 3: Embeddings and Vectors
- Task 4: Prompts
- Task 5: Retrieval Augmented Generation
     - 🚧 ACTIVITY #1: Augment RAG

### Steps to Run:

1. Install UV - which you can do through [this resource](https://docs.astral.sh/uv/#getting-started)
2. Run the command `uv sync`
3. Open your Jupyter notebook and select `.venv` for your kernel. 

# Build 🏗️

Run the notebook!

# Ship 🚢

- Add one of the following "extras" (or whatever augmentations suit your use-case) to the RAG pipeline:
     - Allow it to work with PDF files
     - Implement a new distance metric
     - Add metadata support to the vector database
     - Use a different embedding model
     - Add the capability to ingest a YouTube link
- Make a simple diagram of your RAG process
- Run the notebook
- When you're finished with augmentations to your RAG application - vibe check it against the old one - see if you can "feel the improvement!
- Record a Loom walking through the notebook, the questions in the notebook, and your addition!

# Share 🚀
- Show your App in a loom video and explain the diagram
- Make a social media post about your final application and tag @AIMakerspace
- Share 3 lessons learned
- Share 3 lessons not learned

Here's a template to get your post started!

```
🚀 Exciting News! 🎉

I just built and shipped my very first Retrieval Augmented Generation QA Application using Chainlit and the OpenAI API! 🤖💼 

🔍 Three Key Takeaways:
1️⃣ The power of combining traditional search methods with state-of-the-art generative models is mind-blowing. 🧠✨
2️⃣ Collaboration and leveraging community resources like AI Makerspace can greatly accelerate the learning curve. 🌱📈
3️⃣ Dive deep, keep iterating, and never stop learning. Each project brings a new set of challenges and equally rewarding lessons. 🔄📚

A huge shoutout to the @AI Makerspace for their invaluable resources and guidance. 🙌

Looking forward to more AI-driven adventures! 🌟 Feel free to connect if you'd like to chat more about it! 🤝

#OpenAI #AIPowered #Innovation #TechJourney
```

# Submitting Your Homework
## Main Assignment
Follow these steps to prepare and submit your homework:
1. Pull the latest updates from upstream into the main branch of your AIE8 repo:
    - _(You should have completed this process already):_ For your initial repo setup see [00_Setting Up Git/README.md](https://github.com/AI-Maker-Space/AIE8/tree/main/00_Setting%20Up%20Git)
    - To get the latest updates from AI Makerspace into your own AIE8 repo, run the following commands:
    ```
    git checkout main
    git pull upstream main
    git push origin main
    ```
2. **IMPORTANT:** Start Cursor from the `02_Embeddings_and_RAG` folder (you can also use the _File -> Open Folder_ menu option of an existing Cursor window)
3. Create a branch of your `AIE8` repo to track your changes. Example command: `git checkout -b s02-assignment`
4. Answer Questions 1 - 4 using the `✅ Answer:` markdown cell below them.
5. Complete Activity #1 _(Enhance your RAG application in some way!)_ at the end of the notebook:
   + "your RAG application" refers to the code cells of this notebook, as well as the code in the `aimakerspace` library.
   + At the end of the file is a Python code cell where you will enter the code to enhance the application
   + If you so desire, you can also implement some of the code in new `.py` files and `import` the functionality into that final code cell.
6. Add, commit and push your modified `Pythonic_RAG_Assignment.ipynb` to your origin repository.

>(NOTE: You should not merge this commit into origin's main branch. This will spare you from update challenges for each future session.)

When submitting your homework, provide the GitHub URL to the tracking branch (for example: `s02-assignment`) you created on your AIE8 repo.
