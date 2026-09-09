
from fastapi import FastAPI
from rag import ask_rag

app = FastAPI()


@app.get("/")
def home():
    return {"message": "RAG API is running"}


@app.get("/query")
def query_rag(question: str):

    answer = ask_rag(question)

    return {
        "question": question,
        "answer": answer
    }
