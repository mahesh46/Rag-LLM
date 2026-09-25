# Rag-LLM.

Upload document then query - RaggLLM example

Below steps to setup and run;
1.
pip3 install gradio langchain langchain-community langchain-openai chromadb pypdf

pip3 install -U langchain langchain-core langchain-openai langchain-chroma langchain-community gradio pypdf

How to Run and Test
Save maheshLad_cv.txt into the same directory or upload it directly through the UI.   

Run the Python application:
2.

export OPENAI_API_KEY="your-api-key-here"

3.
Bash
python3 app.py
Open the URL printed in the terminal (usually [http://127.0.0.1:7860](http://127.0.0.1:7860)).

Upload maheshLad_cv.txt in the screen interface and click Build RAG Index.   

Use the chat interface to query details from the CV (e.g., "What projects did Mahesh handle at Deloitte?" or "What certifications does he hold?").



4.
What changed in app.py:

Embeddings (app.py:59): tries OpenAI, falls back to local fastembed on failure.
LLM (app.py:38): probes OpenAI; if the account has credits it uses gpt-4o-mini, otherwise falls back to local Ollama gemma4:latest (via freshly installed langchain-ollama).
Status text now shows which provider is answering.

