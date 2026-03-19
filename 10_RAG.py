"""
PDF RAG — ask questions about any PDF using OpenAI + Chroma
Usage:
    python pdf_rag.py                        # interactive mode
    python pdf_rag.py --pdf doc/myfile.pdf   # specify PDF at launch
"""

import os
import argparse
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

load_dotenv()

# ── 1. Validate API key ────────────────────────────────────────────────────────
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError(
        "OPENAI_API_KEY not found. "
        "Add it to your .env file: OPENAI_API_KEY=sk-..."
    )

# ── 2. Build RAG pipeline from a PDF path ─────────────────────────────────────
def build_rag(pdf_path: str):
    print(f"\n Loading: {pdf_path}")

    # Load
    loader = PyPDFLoader(pdf_path)
    data = loader.load()
    print(f" Pages loaded: {len(data)}")

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    docs = splitter.split_documents(data)
    print(f" Chunks created: {len(docs)}")

    # Embed → Chroma vector store (in-memory, no persistence needed for Q&A)
    print(" Embedding chunks with OpenAI...")
    embedding_fn = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(docs, embedding=embedding_fn)

    # Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions about a document.
Use ONLY the context below to answer. If the answer is not in the context, say so.

Context:
{context}

Question: {question}

Answer:""")

    # LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Format retrieved docs
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return chain


# ── 3. Interactive Q&A loop ────────────────────────────────────────────────────
def interactive_loop(chain):
    print("\n RAG ready! Type your questions below.")
    print(" Type 'exit' or 'quit' to stop.\n")
    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        response = chain.invoke(question)
        print(f"\nAssistant: {response.content}\n")


# ── 4. Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF RAG with OpenAI")
    parser.add_argument("--pdf", type=str, help="Path to the PDF file")
    args = parser.parse_args()

    pdf_path = args.pdf
    if not pdf_path:
        pdf_path = input("Enter path to your PDF file: ").strip()

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    chain = build_rag(pdf_path)
    interactive_loop(chain)
