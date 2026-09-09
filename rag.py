
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ---------------------------------
# 1. Load the document
# ---------------------------------

loader = TextLoader(
    "data/my_document.txt",
    encoding="utf-8"
)

documents = loader.load()


# ---------------------------------
# 2. Split document into chunks
# ---------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=180,
    chunk_overlap=30
)

document_chunks = splitter.split_documents(documents)


# ---------------------------------
# 3. Create free embeddings
# ---------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ---------------------------------
# 4. Store embeddings in FAISS
# ---------------------------------

vector_store = FAISS.from_documents(
    document_chunks,
    embeddings
)


# ---------------------------------
# 5. Create retriever
# ---------------------------------

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)


# ---------------------------------
# 6. Load free LLM
# ---------------------------------

model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# ---------------------------------
# 7. Complete RAG function
# ---------------------------------

def ask_rag(question: str):

    # Retrieve relevant document chunks
    retrieved_docs = retriever.invoke(question)

    # Combine chunks into context
    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    # Build prompt
    prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    # Convert prompt into model tokens
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    # Generate answer
    output_ids = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False
    )

    # Decode generated tokens
    answer = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    return answer
