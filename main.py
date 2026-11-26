
import json
import re
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Create the API
# --------------------------------------------------

app = FastAPI(title="RAG Colosseum")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# 2. Load the knowledge base
# --------------------------------------------------

KB_FILE = Path(__file__).parent / "knowledge_base.json"

with open(KB_FILE, "r", encoding="utf-8") as file:
    raw_data = json.load(file)


def extract_text(item):
    """Support common JSON passage formats."""

    if isinstance(item, str):
        return item

    if isinstance(item, dict):
        for key in ["text", "passage", "content", "body"]:
            if key in item:
                return str(item[key])

    return str(item)


if isinstance(raw_data, dict):
    if "passages" in raw_data:
        raw_data = raw_data["passages"]
    else:
        raw_data = list(raw_data.values())

passages = [extract_text(item) for item in raw_data]

if len(passages) != 50:
    print(f"Warning: Expected 50 passages, found {len(passages)}")


# --------------------------------------------------
# 3. Load the embedding model
# --------------------------------------------------

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Convert all passages into vectors only once.
passage_embeddings = model.encode(
    passages,
    normalize_embeddings=True,
    convert_to_numpy=True
)

# Lock protects model inference during concurrent requests.
model_lock = threading.Lock()


# --------------------------------------------------
# 4. Define the request format
# --------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=3, ge=1, le=10)


# --------------------------------------------------
# 5. Extract useful factual sentences
# --------------------------------------------------

STOPWORDS = {
    "what", "which", "who", "where", "when", "why", "how",
    "is", "are", "was", "were", "the", "a", "an", "of",
    "to", "for", "in", "on", "and", "or", "does", "do",
    "did", "can", "could", "would", "should", "used", "use",
    "with", "by", "from", "that", "this", "it", "its",
    "about", "explain", "tell", "me", "give", "please"
}


def get_keywords(text):
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {word for word in words if word not in STOPWORDS}


def extract_answer(question, selected_passages):
    """
    Select factual sentences from retrieved passages.
    Retrieved text is treated as data, not instructions.
    """

    question_words = get_keywords(question)

    candidates = []

    for passage in selected_passages:
        sentences = re.split(r"(?<=[.!?])\s+", passage)

        for sentence in sentences:
            sentence_words = get_keywords(sentence)

            overlap = len(question_words & sentence_words)

            if overlap > 0:
                candidates.append((overlap, sentence.strip()))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)

    # Keep up to two useful sentences for simple synthesis.
    best_sentences = []

    for _, sentence in candidates:
        if sentence not in best_sentences:
            best_sentences.append(sentence)

        if len(best_sentences) == 2:
            break

    return " ".join(best_sentences)


# --------------------------------------------------
# 6. Query endpoint
# --------------------------------------------------

@app.post("/query")
def query(request: QueryRequest):

    question = request.question.strip()

    if not question:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.0
        }

    # Embed the question.
    with model_lock:
        question_embedding = model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0]

    # Calculate cosine similarities.
    similarities = passage_embeddings @ question_embedding

    # Select the most relevant passages.
    top_k = min(request.top_k, len(passages))

    ranked_indices = np.argsort(similarities)[::-1][:top_k]

    # Convert cosine similarity into the required 0-1 range.
    ranked_scores = [
        float(np.clip(similarities[index], 0.0, 1.0))
        for index in ranked_indices
    ]

    # Original passage numbers must be preserved.
    sources = [int(index) for index in ranked_indices]

    # Abstain if the best passage is too weak.
    best_score = ranked_scores[0] if ranked_scores else 0.0

    if best_score < 0.35:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.2
        }

    selected_passages = [
        passages[index] for index in ranked_indices
    ]

    answer = extract_answer(
        question,
        selected_passages
    )

    if not answer:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.2
        }

    confidence = float(
        np.clip(best_score, 0.0, 1.0)
    )

    return {
        "answer": answer,
        "sources": sources,
        "scores": ranked_scores,
        "confidence": confidence
    }


# --------------------------------------------------
# 7. Health check
# --------------------------------------------------

@app.get("/")
def home():
    return {"status": "RAG Colosseum is running"}
