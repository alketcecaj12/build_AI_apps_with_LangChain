# 🦜 Building AI Applications with LangChain

A hands-on learning project documenting my journey building LangChain applications from scratch — debugging real errors, fixing environment conflicts, and progressively building toward a full RAG pipeline.

---

## 🎯 Project Goal

Build a complete **RAG (Retrieval-Augmented Generation) pipeline** using LangChain, starting from simple LLM chains and incrementally adding document loading, embeddings, vector stores, and retrieval.

---

## 🧠 Learning Approach: The Scientific Method

Every bug and error in this project was debugged using the **5-step scientific method**:

1. **Observation** — what the code is trying to do
2. **Hypothesis** — what might be wrong
3. **Prediction** — what error to expect
4. **Experiment** — the fix to apply
5. **Conclusion** — why it works

---

## ✅ What's Been Built So Far

### 1. LLM Backends
Three different LLM backends working via LangChain:

```python
# OpenAI (cloud)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# HuggingFace (local, small model)
from langchain_huggingface import HuggingFacePipeline
llm = HuggingFacePipeline.from_model_id(
    model_id="crumb/nano-mistral",
    task="text-generation",
    device=-1,
    pipeline_kwargs={"max_new_tokens": 50, "repetition_penalty": 1.3}
)

# Ollama (local, full-size model)
from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="llama3.2")
```

### 2. Prompt Templates

```python
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Basic prompt
template = "Answer the question briefly.\nQuestion: {question}\nAnswer:"
prompt = PromptTemplate.from_template(template)

# Chat prompt with few-shot example
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a geography expert that returns flag colors."),
    ("human", "France"),
    ("ai", "blue, white, red"),
    ("human", "{country}")
])
```

### 3. LCEL Chains (LangChain Expression Language)

```python
# The pipe operator chains components together
chain = prompt | llm | StrOutputParser()
response = chain.invoke({"question": "What is LangChain?"})
print(response)
```

### 4. Document Loading

```python
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader

loader = PyPDFLoader("doc/your_file.pdf")
documents = loader.load()
print(documents[0].page_content)
print(documents[0].metadata)
```

---

## 🗺️ RAG Pipeline Roadmap

```
✅ Step 1 — LLM Backends        (OpenAI, HuggingFace, Ollama)
✅ Step 2 — Prompt Templates     (PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate)
✅ Step 3 — LCEL Chains          (prompt | llm | StrOutputParser)
✅ Step 4 — Document Loading     (PDF, TXT, Web)
🔄 Step 5 — Text Splitting       (chunks with overlap)
⏳ Step 6 — Embeddings           (convert text to vectors)
⏳ Step 7 — Vector Store         (store & search chunks)
⏳ Step 8 — Retriever            (find relevant chunks)
⏳ Step 9 — Full RAG Chain       (retriever + prompt + llm) 🎯 GOAL
```

---

## 🛠️ Setup

### Prerequisites
- Python 3.10+
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Ollama](https://ollama.com) (for local models)

### Create Environment

```bash
conda create -n myenv python=3.10
conda activate myenv
```

### Install Dependencies

```bash
pip install langchain langchain-core langchain-community langchain-openai \
            langchain-huggingface langchain-ollama langchain-chroma \
            transformers==4.44.0 torch==2.4.0 torchvision==0.19.0 \
            python-dotenv pypdf
```

### API Key Setup

Create a `.env` file in the project root:

```bash
touch .env
echo "OPENAI_API_KEY=sk-your-key-here" > .env
echo ".env" >> .gitignore  # ⚠️ never commit your key!
```

Load it in Python:

```python
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())
api_key = os.getenv("OPENAI_API_KEY")
```

### Local Models via Ollama

```bash
ollama pull llama3.2
ollama pull nomic-embed-text  # for embeddings
```

---

## ⚠️ Known Environment Gotchas

Hard-won lessons from real debugging sessions:

| Issue | Cause | Fix |
|---|---|---|
| `langchain.verbose` AttributeError | LangChain version mismatch | Upgrade all LangChain packages together |
| `transformers` import error | `transformers 5.x` too new | Pin to `transformers==4.44.0` |
| `torchvision::nms` missing | `torch`/`torchvision` mismatch | Use matched pair: `torch==2.4.0` + `torchvision==0.19.0` |
| `device` TypeError | `"mps"` not supported in older versions | Use `device=-1` for CPU |
| `is_mlu_available` missing | `accelerate` outdated | `pip install --upgrade accelerate` |
| `.env` not loading | `load_dotenv()` missing or wrong path | Use `load_dotenv(find_dotenv())` |
| Chroma dimension mismatch | Switching between OpenAI (1536d) and Ollama (4096d) embeddings | Delete `chroma_db/` folder and re-embed |
| File not found | Hyphen vs underscore in filename | Always run `os.listdir('.')` to verify exact filename |

---

## 📦 Tech Stack

| Tool | Purpose |
|---|---|
| [LangChain](https://langchain.com) | LLM application framework |
| [OpenAI GPT-4o-mini](https://openai.com) | Cloud LLM |
| [Ollama + LLaMA 3.2](https://ollama.com) | Local LLM |
| [HuggingFace](https://huggingface.co) | Local model hub |
| [Chroma](https://trychroma.com) | Vector database (coming soon) |
| [Python 3.10](https://python.org) | Runtime |

---

## 📚 What I've Learned

- The LCEL `prompt | llm | parser` pattern and why it's powerful
- How to swap LLM backends in one line — OpenAI, HuggingFace, Ollama
- Why `torch` and `torchvision` must always be installed as a matched pair
- How virtual environment conflicts silently break imports
- Why API keys must never be hardcoded — always use `.env` + `.gitignore`
- How few-shot prompting works in `ChatPromptTemplate`
- How `StrOutputParser` extracts clean text from LLM response objects

---

## 🔐 Security Notes

- **Never** hardcode API keys in source code
- **Always** add `.env` to `.gitignore` before first commit
- Use `os.getenv("OPENAI_API_KEY")` — never paste keys directly
- If you accidentally expose a key, **invalidate it immediately** at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

---

## 📁 Project Structure

```
.
├── scripts/
│   └── *.py              # LangChain experiment scripts
├── doc/
│   └── *.pdf             # Source documents for RAG
├── chroma_db/            # Vector store (not committed)
├── .env                  # API keys (not committed)
├── .gitignore
└── README.md
```

---

*Built with curiosity, debugged with the scientific method. 🔬*
