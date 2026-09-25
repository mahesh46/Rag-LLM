import os
import gradio as gr

# Modern LangChain imports
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

LOCAL_LLM_MODEL = "gemma4:latest"

def load_env(path=".env"):
    if not os.path.exists(path):
        return
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                key, _, value = line.partition("=")
                os.environ.setdefault(key, value.strip().strip("\"'"))
    except OSError as e:
        print(f"Warning: could not read {path}: {e}")

load_env()

# Global variables
retrieval_chain = None

def format_docs(docs):
    """Helper to join retrieved document chunks into a single string context."""
    return "\n\n".join(doc.page_content for doc in docs)

def create_llm():
    """Use OpenAI if the account has credits, otherwise fall back to local Ollama."""
    try:
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        llm.invoke("ping")
        return llm, "OpenAI (gpt-4o-mini)"
    except Exception:
        return ChatOllama(model=LOCAL_LLM_MODEL, temperature=0), f"Local Ollama ({LOCAL_LLM_MODEL})"

def process_file(file_obj):
    global retrieval_chain
    if file_obj is None:
        return "Please upload a valid file."
    
    file_path = file_obj.name
    
    # 1. Load document based on file extension
    if file_path.endswith('.txt'):
        loader = TextLoader(file_path)
    elif file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        return "Unsupported file type. Please upload a .txt or .pdf file."
    
    documents = loader.load()
    
    # 2. Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    # 3. Vectorstore and Retriever using langchain-chroma
    embedder = None
    try:
        embedder = OpenAIEmbeddings()
    except Exception:
        embedder = None

    if embedder is None:
        from langchain_community.embeddings import FastEmbedEmbeddings
        embedder = FastEmbedEmbeddings()

    try:
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embedder)
    except Exception:
        from langchain_community.embeddings import FastEmbedEmbeddings
        vectorstore = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 4. Define LLM and LCEL (LangChain Expression Language) Chain
    llm, provider = create_llm()
    
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )
    
    prompt = ChatPromptTemplate.from_template(system_prompt)
    
    # Construct modern retrieval pipeline
    retrieval_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return f"File '{os.path.basename(file_path)}' processed successfully! You can now ask questions. (Answering via {provider})"

def chat_response(message, history):
    global retrieval_chain
    if retrieval_chain is None:
        return "Please upload and process a text file first."
    
    return retrieval_chain.invoke(message)

# Gradio Interface
with gr.Blocks(title="Document RAG Assistant") as demo:
    gr.Markdown("# 📄 Interactive RAG Document Assistant")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload Text/CV File", file_types=[".txt", ".pdf"])
            upload_btn = gr.Button("Build RAG Index", variant="primary")
            status_output = gr.Textbox(label="Status", interactive=False)
            
        with gr.Column(scale=2):
            chatbot = gr.ChatInterface(
                fn=chat_response,
                description="Ask questions about the uploaded document.",
                examples=[
                    "What is Mahesh's total experience in iOS development?",
                    "Which companies has Mahesh worked for?",
                    "What are his core technical skills?"
                ]
            )
            
    upload_btn.click(fn=process_file, inputs=[file_input], outputs=[status_output])

if __name__ == "__main__":
    demo.launch()