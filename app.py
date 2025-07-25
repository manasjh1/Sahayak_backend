from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt, qa_prompt

from langchain_pinecone import PineconeVectorStore
from groq import Groq
from pymongo import MongoClient

from datetime import datetime
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware


# ----------------------------
# FastAPI Setup
# ----------------------------
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://sahayak-cizr.vercel.app/"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings = download_hugging_face_embeddings()
docsearch = PineconeVectorStore.from_existing_index(index_name="bot", embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
client = Groq(api_key=GROQ_API_KEY)

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[MONGO_DB_NAME]
collection = db[COLLECTION_NAME]

# ----------------------------
# Utility
# ----------------------------
def generate_completion(prompt_template, user_message: str):
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            prompt_template,
            {"role": "user", "content": user_message}
        ],
        temperature=0.9,
        max_completion_tokens=600,
        top_p=1,
        stream=False,
        stop=None,
    )
    return completion.choices[0].message.content.strip()

def retrieve_context(query: str) -> str:
    docs = retriever.invoke(query)
    if not docs:
        return ""
    return "\n".join(doc.page_content for doc in docs)

# ----------------------------
# ROUTES
# ----------------------------

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/worksheet")
async def generate_worksheet(msg: str = Form(...)):
    print("Worksheet Topic:", msg)
    context = retrieve_context(msg)
    user_prompt = f"""
Topic: {msg}
Context: {context}
"""
    response = generate_completion(system_prompt, user_prompt)

    chat_entry = {
        "type": "worksheet",
        "user_message": msg,
        "bot_response": response,
        "timestamp": datetime.utcnow()
    }
    collection.insert_one(chat_entry)

    return JSONResponse(content={"worksheet": response})

@app.get("/history")
async def chat_history():
    history = list(collection.find().sort("timestamp", -1).limit(10))
    for h in history:
        h["_id"] = str(h["_id"])
        h["timestamp"] = h["timestamp"].isoformat()
    return JSONResponse(content={"history": history})

# ----------------------------
# Q/A ENDPOINT
# ----------------------------

class QARequest(BaseModel):
    question: str

@app.post("/qa")
async def generate_answer(data: QARequest):
    context = retrieve_context(data.question)
    user_prompt = f"""
Question: {data.question}
Context: {context}
"""
    response = generate_completion(qa_prompt, user_prompt)
    return JSONResponse(content={"answer": response})
