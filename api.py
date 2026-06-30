from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import random
import re
import os
from dotenv import load_dotenv

load_dotenv("Backend/.env")

import sys
import os
sys.path.append(os.getcwd())

from Backend.document_processor import extract_text
from Backend.embeddings import create_chunks
from Backend.rag_graph import store_vectors, retrieve_docs
from  Backend.evaluation import evaluate_answer
from  Backend.question_generator import generate_question
from  Backend.question_utils import extract_questions

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

questions_store = []

@app.get("/")
def home():
    return {"message": "AI Mock Interview API Running"}

@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...), difficulty: str = Form("Beginner")):
    global questions_store

    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        temp_path = tmp.name

    with open(temp_path, "rb") as f:
        text_data = extract_text(f)

    os.remove(temp_path)

    existing_questions = extract_questions(text_data)

    if len(existing_questions) >= 3:
        questions = existing_questions
    else:
        chunks = create_chunks(text_data)
        store_vectors(chunks)
        docs = retrieve_docs("interview questions")
        questions = []

        for doc in docs:
            q = generate_question(doc, difficulty)
            questions.append(q)

    questions = list(set(questions))
    random.shuffle(questions)

    questions_store = questions

    return {
        "message": "Resume processed",
        "total_questions": len(questions)
    }

@app.get("/question/{index}")
def get_question(index: int):
    global questions_store

    if index < len(questions_store):
        return {
            "question": questions_store[index],
            "index": index
        }

    return {"message": "Interview completed"}

@app.post("/evaluate")
def evaluate(answer: str = Form(...)):
    feedback = evaluate_answer(answer)

    match = re.search(r"(?i)score[\s:*]*\(?(\d+)\)?", feedback)
    score = int(match.group(1)) if match else 0

    return {
        "score": score,
        "feedback": feedback
    }