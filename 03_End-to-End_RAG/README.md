<p align = "center" draggable=”false” ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719" 
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">Session 3: End-to-End RAG</h1>

### [Quicklinks](https://github.com/AI-Maker-Space/AIE7/tree/main/00_AIM_Quicklinks)

 📰 Session Sheet | ⏺️ Recording     | 🖼️ Slides        | 👨‍💻 Repo         | 📝 Homework      | 📁 Feedback       |
|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| [Session 3: E2E & Use Cases](https://www.notion.so/Session-3-End-to-End-AI-Applications-OSS-Models-I-and-2025-Industry-Use-Cases-26acd547af3d80b4b646e2fd6f1fd31c) |[Recording!](https://us02web.zoom.us/rec/share/7UJErmXFPnBQmIxWoeHVtCVcjtF1c_XmzAybJLGgei5Xrju_Q2jgPzgjYI8YT06o.pRQgg0m-t4-HHAmV) (*6zDd0%S) | [Session 3 Slides](https://www.canva.com/design/DAGzJw-3i34/1UdGr5HlXlPjFtabOAWdkw/edit?utm_content=DAGzJw-3i34&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) | You are here! | [Session 3 Assignment: E2E](https://forms.gle/ZVvwkbg4jEpHKpCY9) | [AIE8 Feedback 9/16](https://forms.gle/9SnYW7vgNLeGpkh47)

Building off last week, we're going to take our Pythonic RAG application to the next level!

This will be our updated system diagram for our application after the changes you make below are completed!

![image](https://i.imgur.com/FsNSG9T.png)

## 🏗️ Activity #1:

This week, we'll be working out of your challenge repository!

1. You can begin by copying and pasting the new `aimakerspace` library into your challenge repository at the ROOT LEVEL (top level directory).

> NOTE: You can use the following command to do the same, just be sure to replace the paths with the correct ones based on your local environment.

```bash
cp /PATH/TO/THIS/FOLDER/aimakerspace /PATH/TO/YOUR/CHALLENGE
```

2. Create a new rule in your `.cursor` folder with the following text, make this a global rule that applies at all times:

![image](https://i.imgur.com/uWeyoHC.png)

```
You always prefer to use branch development. Before writing any code - you create a feature branch to hold those changes. 

After you are done - provide instructions in a "MERGE.md" file that explains how to merge the changes back to main with both a GitHub PR route and a GitHub CLI route.
```

3. Use Cursor (or your own mind, of course) to include the RAG functionality we discussed last week to allow users to upload PDFs and "chat with them" (interact with a RAG pipeline). 

> NOTE: You can do this with the following prompt as a way to get started: 

```
Modify the application (frontend and backend) to: 

- Allow the user to upload a PDF
  - Your PDF should be used to build your RAG's context
  - The LLM should only answer questions using information from the provided context
- Index the PDF using the `aimakerspace` library
- Chat with the PDF using a simple RAG system built with the `aimakerspace` library
```

> NOTE: You need to do this from your challenge repo.

4. Deploy the application with Vercel.

## 🏗️ Activity #2:

Determine a specific use-case for RAG, and adapt your challenge application to that new use-case. 

Think about how people from that domain or expertise area may interact with your application - and ensure it's suited to them. 

This may involve:

- Modifying the UI
- Adding additional file-types to be ingested by RAG

Once done, deploy your customized application to Vercel!

### 🚧 Advanced Build:

<details>
<summary>🚧 Advanced Build 🚧 (OPTIONAL - <i>open this section for the requirements</i>)</summary>

Leverage Together API's endpoints to power the LLM in your deployed application - this will require a small change to the `aimakerspace/openai_utils/chatmodel.py` file. 

> NOTE: Also describe the process of modifying the endpoints used - and what made it difficult or not to make this change.

</details>

## Submitting Your Homework

### Main Assignment
Follow these steps to prepare and submit your homework:
1. Verify that Activity #1 was completed by validating that:
    + Cursor/Claude followed the global rules you added to the .cursor file:
      + It created a new branch prior to generating code
      + It created a MERGE.md file with appropriate instructions for merging this new branch
      > NOTE: If you "used your own mind" rather than vibe coding with Cursor/Claude then it is your responsibility to do these two things, or ask Claude directly to do them for you.
    + The deployed application is able to upload a PDF, index it, and then chat about the contents of the PDF.
2. Verify that Activity #2 was completed by validating that:
    + Cursor/Claude followed the global rules you added to the .cursor file (same as in Step 1 above)
    + The deployed application is still able to upload, process, and chat about a PDF (the functionality of Activity #1)
    + The deployed application meets your new functionality requirements for the RAG-specific use-case that you implemented
3. Create a _**5 minute or less**_ Loom video about the assignment and your modified challenge application
4. Post on social media:
    - LinkedIn or X ⬅️ 2 extra credit points! _OR_
    - [Discord's #build-ship-share-🏗️-🚢-🚀 channel](https://discord.com/channels/1135695983720792216/1415130012394192896) ⬅️ 1 extra credit point!
5. Complete the Homework Form!

### Advanced Build (Alternative to the Main Assignment)
Follow these steps to prepare and submit your homework:
1. Deploy your updated application to Vercel
2. Create a _**5 minute or less**_ Loom video about modifying the endpoints, including a discussion of what made this change difficult and/or easy to implement.
3. Post on social media:
    - LinkedIn or X ⬅️ 2 extra credit points! _OR_
    - [Discord's #build-ship-share-🏗️-🚢-🚀 channel](https://discord.com/channels/1135695983720792216/1415130012394192896) ⬅️ 1 extra credit point!
4. Complete the Homework Form!
