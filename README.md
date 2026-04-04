# YouTube Chatbot using RAG (LangChain + FAISS)

A simple **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about a **YouTube video** by:
1) fetching the video transcript,  
2) chunking it into smaller pieces,  
3) embedding the chunks with OpenAI embeddings,  
4) storing/searching them in a **FAISS** vector store, and  
5) generating answers with an OpenAI chat model **using only retrieved transcript context**.

This repo includes:
- `YOUTUBE_Chatbot_rag_using_langchain.ipynb` — the original notebook workflow
- `youtube_chatbot_rag_using_langchain.py` — the legacy notebook-export script
- `main.py` — the runnable entrypoint for the modularized project
- `youtube_chatbot_rag/` — reusable package modules for ingestion, timestamps, prompts, vector store, and chain assembly

---

## How it works (pipeline)

### 1) Transcript ingestion (YouTube)
Uses `youtube-transcript-api` to fetch transcript segments for one or more resolved YouTube videos.

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
- Include clickable YouTube source citations with timestamps in the answer output

### 6) Chain (LangChain Runnable pipeline)
Builds a runnable chain using:
- `RunnableParallel` + `RunnablePassthrough` + `RunnableLambda`
- `StrOutputParser`

### 7) Persistent FAISS index
The modular app now saves the FAISS index to disk and reloads it on later runs.

Default cache location:
- `.vectorstores/faiss/<corpus_hash>/`

Behavior:
- If the index exists, it is loaded from disk
- If not, the app rebuilds it from the transcript and saves it

### 8) Multiple videos / playlists
The modular CLI accepts multiple sources and playlist URLs.

If you provide a playlist link, the app resolves all videos from that playlist and attempts to fetch transcripts for each video.

Examples:
```bash
python main.py --source Gfr50f6ZBvo --source https://www.youtube.com/watch?v=dQw4w9WgXcQ
python main.py --source https://www.youtube.com/playlist?list=PL1234567890ABCDEF
python main.py --source Gfr50f6ZBvo --source https://www.youtube.com/playlist?list=PL1234567890ABCDEF --question "What themes are shared across these videos?"
```

The app resolves every source into a flat set of video IDs, builds one combined corpus, and caches the resulting FAISS index using a stable hash of the full source set.

---

## Repository structure

- `README.md` — project overview (this file)
- `YOUTUBE_Chatbot_rag_using_langchain.ipynb` — notebook version
- `youtube_chatbot_rag_using_langchain.py` — legacy script export
- `main.py` — simple entrypoint for the modular version
- `youtube_chatbot_rag/` — package with reusable modules

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
Run the modularized project with:
```bash
python main.py
```

After launch, you can paste the full YouTube video or playlist link when prompted, then keep asking multiple questions in the same running session.

The first run for a given source set will build and save the FAISS index. Later runs reuse the cached index from `.vectorstores/faiss/`.

To pass a full YouTube video link directly:
```bash
python main.py --video-link https://www.youtube.com/watch?v=Gfr50f6ZBvo
```

To pass more than one source, repeat `--source`:
```bash
python main.py --source Gfr50f6ZBvo --source https://www.youtube.com/playlist?list=PL1234567890ABCDEF
```

You can also repeat `--video-link` for multiple videos:
```bash
python main.py --video-link https://www.youtube.com/watch?v=Gfr50f6ZBvo --video-link https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

To start directly in persistent chat mode:
```bash
python main.py --interactive --video-link https://www.youtube.com/watch?v=Gfr50f6ZBvo
```

In chat mode, type `exit` to stop.

The legacy `.py` export is still in the repository for reference, but the modular entrypoint is now the recommended way to run the project locally.

If you want to run the notebook export directly, you should remove the `!pip ...` lines and install dependencies via pip (as shown above).

---

## Configuration

### Change the default source used when no input is provided
In `youtube_chatbot_rag/config.py`, update:
```python
DEFAULT_VIDEO_SOURCE = "https://www.youtube.com/watch?v=..."
```

You can also skip config changes and pass source links directly at runtime via prompt, `--video-link`, or `--source`.

### Ask questions
Example questions used in the script:
- `"What is deepmind"`
- `"Can you summarize the video"`
- `"is the topic of nuclear fusion discussed in this video? if yes then what was discussed"`

Answers now include clickable source links pulled from the retrieved transcript chunks, for example `https://www.youtube.com/watch?v=VIDEO_ID&t=720s`.

---

## Notes / Limitations

- Transcript availability depends on the video (some videos have transcripts disabled).
- Some cloud/devcontainer IPs are blocked by YouTube for transcript requests. If that happens, set proxy variables before running:
	- `YOUTUBE_TRANSCRIPT_PROXY_HTTP`
	- `YOUTUBE_TRANSCRIPT_PROXY_HTTPS`
	- or standard `HTTP_PROXY` / `HTTPS_PROXY`
- Answers are only as good as the transcript content + retrieval quality.
- This is a minimal demo; it doesn’t include a web UI or persistent storage.

Example:
```bash
export YOUTUBE_TRANSCRIPT_PROXY_HTTP="http://user:pass@proxy-host:port"
export YOUTUBE_TRANSCRIPT_PROXY_HTTPS="http://user:pass@proxy-host:port"
python main.py --interactive --video-link "https://youtu.be/y1fGlAECNFM"
```

---



## License

No license file is currently included in the repository. If you want others to reuse the code, consider adding a license (for example MIT).