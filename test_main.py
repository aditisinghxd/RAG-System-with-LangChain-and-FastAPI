
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "RAG API is running"
    }


def test_query():
    question = "Why is climate change a problem for polar bears?"

    response = client.get(
        "/query",
        params={"question": question}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == question
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0

    # The temporary rag.py must be replaced
    assert not data["answer"].startswith("Received question:")
