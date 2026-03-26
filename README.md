# YouTube Chatbot using RAG (LangChain + FAISS)

A simple **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about a **YouTube video** by:
1) fetching the video transcript,  
2) chunking it into smaller pieces,  
3) embedding the chunks with OpenAI embeddings,  
4) storing/searching them in a **FAISS** vector store, and  
5) generating answers with an OpenAI chat model **using only retrieved transcript context**.

This repo includes:
- `YOUTUBE_Chatbot_rag_using_langchain.ipynb` — the original notebook workflow
- `youtube_chatbot_rag_using_langchain.py` — a Python script export of the notebook

---

## How it works (pipeline)

### 1) Transcript ingestion (YouTube)
Uses `youtube-transcript-api` to fetch the transcript for a given `video_id`, then flattens it into a single `transcript_text` string.

### 2) Text splitting
Uses LangChain’s `RecursiveCharacterTextSplitter`:
- `chunk_size=1000`
- `chunk_overlap=200`

### 3) Embeddings + Vector store (FAISS)
Creates embeddings with:
- `OpenAIEmbeddings(model="text-embedding-3-small")`

Stores the embedded chunks in:
- `FAISS.from_documents(chunks, embeddings)`

### 4) Retrieval
Creates a retriever using similarity search:
- `k = 4` (top 4 chunks returned)

### 5) Generation (grounded answers only)
Uses:
- `ChatOpenAI(model="gpt-4o-mini", temperature=0.2)`

Prompt behavior:
- Answer **only** from the provided transcript context
- If context is insufficient, say **"I don't know"**

### 6) Chain (LangChain Runnable pipeline)
Builds a runnable chain using:
- `RunnableParallel` + `RunnablePassthrough` + `RunnableLambda`
- `StrOutputParser`

---

## Repository structure

- `README.md` — project overview (this file)
- `YOUTUBE_Chatbot_rag_using_langchain.ipynb` — notebook version
- `youtube_chatbot_rag_using_langchain.py` — script version

---

## Requirements

You’ll need:
- Python 3.9+ (recommended)
- An OpenAI API key

Python packages used in the repo:
- `langchain-text-splitters`
- `langchain-openai`
- `langchain-community`
- `faiss-cpu`
- `youtube-transcript-api`

---

## Setup

### 1) Install dependencies
```bash
pip install -U langchain-text-splitters langchain-openai langchain-community faiss-cpu youtube-transcript-api
```

### 2) Set your OpenAI API key

**Recommended (environment variable):**
```bash
export OPENAI_API_KEY="your_api_key_here"
```

**Note:** the current script contains a placeholder line:
```python
os.environ["OPENAI_API_KEY"] = "Paste API HERE"
```
It’s better to avoid hardcoding secrets into code.

---

## Usage

### Option A — Run the notebook
Open and run:
- `YOUTUBE_Chatbot_rag_using_langchain.ipynb`

### Option B — Run the Python script
The `.py` file is a Colab-export style script and includes notebook-style `!pip install ...` commands. It works best in **Colab/Jupyter**.

If you want to run it as a normal Python script locally, you should remove the `!pip ...` lines and install dependencies via pip (as shown above).

---

## Configuration

### Change the target YouTube video
In `youtube_chatbot_rag_using_langchain.py`, update:
```python
video_id = "Gfr50f6ZBvo"
```

### Ask questions
Example questions used in the script:
- `"What is deepmind"`
- `"Can you summarize the video"`
- `"is the topic of nuclear fusion discussed in this video? if yes then what was discussed"`

---

## Notes / Limitations

- Transcript availability depends on the video (some videos have transcripts disabled).
- Answers are only as good as the transcript content + retrieval quality.
- This is a minimal demo; it doesn’t include a web UI or persistent storage.

---

## Next improvements (ideas)

- Convert the script into a clean CLI app (no notebook `!pip` commands).
- Add support for multiple videos / playlists.
- Persist the FAISS index to disk and reload it.
- Add a simple Streamlit/Gradio UI.
- Add source citations (timestamps + transcript segments) in answers.

---

## License

No license file is currently included in the repository. If you want others to reuse the code, consider adding a license (e.g., MIT).
