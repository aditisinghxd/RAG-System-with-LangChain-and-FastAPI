# RAG System with LangChain and FastAPI

A simple end-to-end **Retrieval-Augmented Generation (RAG)** application built with LangChain, Hugging Face, FAISS, and FastAPI.

The project demonstrates how external documents can be loaded, split into chunks, converted into vector embeddings, retrieved based on semantic similarity, and supplied as context to a language model before generating an answer.

The final RAG pipeline is exposed through a FastAPI endpoint.

---

## Architecture

```text
User Question
      |
      v
FastAPI /query
      |
      v
   ask_rag()
      |
      v
   Retriever
      |
      v
     FAISS
      |
      v
Relevant Document Chunks
      |
      v
    Context
      |
      v
     Prompt
      |
      v
  FLAN-T5 Base
      |
      v
Generated Answer
      |
      v
 JSON Response
```

---

## How the RAG Pipeline Works

### 1. Document Loading

The source text is loaded using LangChain's `TextLoader`.

```python
loader = TextLoader(
    "data/my_document.txt",
    encoding="utf-8"
)

documents = loader.load()
```

---

### 2. Text Chunking

The document is divided into smaller chunks using `RecursiveCharacterTextSplitter`.

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=180,
    chunk_overlap=30
)

document_chunks = splitter.split_documents(documents)
```

Splitting documents into smaller pieces improves retrieval because the system can search for the most relevant section instead of processing the complete document for every question.

---

### 3. Embeddings

The project uses the free Hugging Face embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Each text chunk is converted into a numerical vector representing its semantic meaning.

```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

---

### 4. Vector Storage with FAISS

The document embeddings are stored in a FAISS vector store.

```python
vector_store = FAISS.from_documents(
    document_chunks,
    embeddings
)
```

FAISS enables efficient similarity search between the user's question and the stored document vectors.

---

### 5. Retrieval

The FAISS vector store is converted into a LangChain retriever.

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)
```

For every question, the two most relevant document chunks are retrieved.

---

### 6. Context Construction

The retrieved chunks are combined into a single context:

```python
context = "\n\n".join(
    doc.page_content for doc in retrieved_docs
)
```

The context is then included in the prompt sent to the language model.

---

### 7. Language Model

The project uses the free Hugging Face model:

```text
google/flan-t5-base
```

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
```

The prompt instructs the model to answer using only the retrieved context.

```text
Context:
<retrieved document chunks>

Question:
<user question>

Answer:
```

---

## FastAPI

The RAG pipeline is exposed through a FastAPI application.

Two endpoints are available.

### Health Check

```http
GET /
```

Response:

```json
{
  "message": "RAG API is running"
}
```

### Ask the RAG System

```http
GET /query?question=<your-question>
```

Example:

```http
GET /query?question=Why is climate change a problem for polar bears?
```

Example response:

```json
{
  "question": "Why is climate change a problem for polar bears?",
  "answer": "This reduces the amount of time polar bears have available for hunting and can make it harder for them to obtain enough food"
}
```

---

## Project Structure

```text
rag-system-with-langchain-and-fastapi/
│
├── main.py
├── rag.py
├── test_main.py
├── README.md
│
└── data/
    └── my_document.txt
```

### `rag.py`

Contains the complete RAG pipeline:

```text
Document Loading
      ↓
Chunking
      ↓
Embeddings
      ↓
FAISS
      ↓
Retriever
      ↓
Context
      ↓
FLAN-T5
      ↓
Answer
```

### `main.py`

Contains the FastAPI application and exposes the RAG system through HTTP endpoints.

### `test_main.py`

Contains API tests using FastAPI's `TestClient`.

### `data/my_document.txt`

Contains the example knowledge-base document used by the RAG system.

---

## Technologies Used

- Python
- LangChain
- Hugging Face Transformers
- Sentence Transformers
- FAISS
- FastAPI
- Uvicorn
- PyTorch
- Pytest

---

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd rag-system-with-langchain-and-fastapi
```

Install the required dependencies:

```bash
pip install \
    langchain \
    langchain-community \
    langchain-text-splitters \
    langchain-huggingface \
    sentence-transformers \
    faiss-cpu \
    transformers \
    sentencepiece \
    accelerate \
    fastapi \
    uvicorn \
    pytest
```

The Hugging Face embedding and language models will be downloaded automatically when the application is first started.

---

## Running the Application

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

---

## Swagger API Documentation

FastAPI automatically generates interactive API documentation.

After starting the server, open:

```text
http://127.0.0.1:8000/docs
```

You can test the `/query` endpoint directly through Swagger UI.

---

## Running the Tests

Run:

```bash
pytest -q test_main.py
```

The tests verify:

- the API health endpoint returns a successful response
- the `/query` endpoint accepts a question
- the RAG system returns a non-empty generated answer

---

## Example

Question:

```text
Why is climate change a problem for polar bears?
```

The retriever identifies the relevant document section:

```text
Climate change is causing Arctic sea ice to decline.
This reduces the amount of time polar bears have available
for hunting and can make it harder for them to obtain enough food.
```

The retrieved information is supplied to the language model as context, producing an answer grounded in the source document.

---

## Key Concepts Demonstrated

This project helped me understand the complete RAG workflow rather than treating RAG as a black box.

I implemented and explored:

- document loading
- text chunking
- embeddings
- semantic similarity
- vector databases
- FAISS similarity search
- LangChain retrievers
- context augmentation
- prompt construction
- local Hugging Face language models
- FastAPI endpoints
- Uvicorn
- Swagger/OpenAPI
- API testing with Pytest

---

## Why RAG?

A language model normally generates responses using knowledge contained in the model itself.

RAG adds an external retrieval step:

```text
Without RAG:

Question
   ↓
LLM
   ↓
Answer


With RAG:

Question
   ↓
Search external knowledge
   ↓
Retrieve relevant information
   ↓
Add information to prompt
   ↓
LLM
   ↓
Grounded answer
```

This allows the model to answer questions using information from a specific knowledge base.

---

## Future Improvements

Possible extensions include:

- support for PDF documents
- support for multiple documents
- returning document sources with answers
- persistent FAISS indexes
- configurable chunk sizes
- configurable retrieval `k`
- POST endpoints
- asynchronous request handling
- stronger local language models
- Docker containerisation
- cloud deployment
- retrieval and generation evaluation
- conversation history

---

## Interview Summary

A concise way to describe this project:

> I built an end-to-end Retrieval-Augmented Generation system using LangChain. I loaded and chunked a source document, generated semantic embeddings using a Hugging Face Sentence Transformer, and stored them in FAISS. For every user query, the system embeds the question and retrieves the most relevant document chunks using vector similarity search. Those chunks are combined into context and passed with the question to a FLAN-T5 language model to generate a grounded response. I then exposed the pipeline through FastAPI, served it using Uvicorn, tested the endpoints with Pytest, and validated the API through Swagger.

---

## Learning Project

This repository was created as a hands-on learning project to understand how the individual components of a RAG application work together from document ingestion to retrieval, generation, and API serving.
