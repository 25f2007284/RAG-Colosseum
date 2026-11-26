import json
import re
import math
from collections import Counter

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# --------------------------------------------------
# App setup
# --------------------------------------------------

app = FastAPI(title="RAG Colosseum API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Load knowledge base
# --------------------------------------------------

with open("knowledge_base.json", "r", encoding="utf-8") as file:
    raw_data = json.load(file)


def extract_passages(data):
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("passages", list(data.values()))
    else:
        items = []

    passages = []

    for index, item in enumerate(items):
        if isinstance(item, str):
            text = item
            source_id = index
        elif isinstance(item, dict):
            text = (
                item.get("text")
                or item.get("passage")
                or item.get("content")
                or item.get("body")
                or ""
            )
            source_id = item.get("id", index)
        else:
            continue

        if text:
            passages.append({
                "id": source_id,
                "text": str(text)
            })

    return passages


passages = extract_passages(raw_data)


# --------------------------------------------------
# Lightweight text processing
# --------------------------------------------------

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "what", "which", "who", "where", "when", "why",
    "how", "does", "do", "did", "can", "could",
    "would", "should", "and", "or", "but", "to",
    "of", "in", "on", "for", "with", "by", "from",
    "about", "this", "that", "these", "those",
    "it", "its", "their", "they", "we", "you",
    "i", "as", "at", "be", "been", "being"
}


def tokenize(text):
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [word for word in words if word not in STOP_WORDS]


def make_vector(tokens, idf):
    counts = Counter(tokens)
    vector = {}

    for word, count in counts.items():
        if word in idf:
            vector[word] = (1 + math.log(count)) * idf[word]

    return vector


def cosine_similarity(vector_a, vector_b):
    dot = sum(
        value * vector_b.get(word, 0.0)
        for word, value in vector_a.items()
    )

    norm_a = math.sqrt(sum(value * value for value in vector_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vector_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


# Prepare passage vectors once at startup
passage_tokens = [tokenize(p["text"]) for p in passages]

document_frequency = Counter()

for tokens in passage_tokens:
    document_frequency.update(set(tokens))

total_documents = max(len(passages), 1)

idf = {
    word: math.log(1 + total_documents / frequency)
    for word, frequency in document_frequency.items()
}

passage_vectors = [
    make_vector(tokens, idf)
    for tokens in passage_tokens
]


# --------------------------------------------------
# Request and response
# --------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=3, ge=1, le=10)


# --------------------------------------------------
# Query endpoint
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

    question_tokens = tokenize(question)

    if not question_tokens:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.0
        }

    question_vector = make_vector(question_tokens, idf)

    ranked = []

    for index, vector in enumerate(passage_vectors):
        score = cosine_similarity(question_vector, vector)

        if score > 0:
            ranked.append((score, index))

    ranked.sort(reverse=True)

    if not ranked or ranked[0][0] < 0.10:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.0
        }
    
    selected = ranked[:request.top_k]

    # Keep only passages with meaningful query overlap
    selected = [
        (score, index)
        for score, index in selected
        if score >= max(0.10, ranked[0][0] * 0.45)
    ]

    if not selected:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.0
        }

    # Extract relevant sentences from retrieved passages
    question_words = set(question_tokens)
    answer_sentences = []

    for score, index in selected:
        text = passages[index]["text"]

        sentences = re.split(r"(?<=[.!?])\s+", text)

        scored_sentences = []

        for sentence in sentences:
            sentence_tokens = set(tokenize(sentence))

            overlap = len(question_words & sentence_tokens)

            if overlap > 0:
                sentence_score = overlap / max(len(question_words), 1)
                scored_sentences.append((sentence_score, sentence))

        scored_sentences.sort(reverse=True)

        if scored_sentences:
            best_sentence = scored_sentences[0][1]

            if best_sentence not in answer_sentences:
                answer_sentences.append(best_sentence)

    if not answer_sentences:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.2
        }

    answer = " ".join(answer_sentences[:3])

    source_ids = [
        passages[index]["id"]
        for score, index in selected
    ]

    scores = [
        round(float(score), 4)
        for score, index in selected
    ]

    confidence = 1.0

    return {
        "answer": answer,
        "sources": source_ids,
        "scores": scores,
        "confidence": confidence
    }


@app.get("/")
def home():
    return {
        "status": "RAG Colosseum API is running",
        "passages_loaded": len(passages)
    }
