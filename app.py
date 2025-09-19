# app.py

import os
from dotenv import load_dotenv
from datetime import datetime
from fastapi import FastAPI, Request, Form, Depends, HTTPException, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_pinecone import PineconeVectorStore
from pymongo import MongoClient
from groq import Groq
import google.generativeai as genai
import uvicorn
from src.helper import load_pdf_file, text_split
from src.prompt import system_prompt, qa_prompt, video_script_prompt
from pinecone.grpc import PineconeGRPC as Pinecone

# Import only the necessary auth components
from auth_routes import auth_router
# We no longer need get_current_user or get_current_active_user for endpoint protection

app = FastAPI(
    title="Sahayak - AI Educational Platform (Public API)",
    description="AI-Powered Educational Revolution for 260 Million Students - Now with Public Endpoints",
    version="1.1.0"
)

# CORS CONFIGURATION (No changes here)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://sahayak.me",
        "https://front-eight-murex.vercel.app",
        "https://sahayak-cizr.vercel.app",
        "https://www.sahayak.me/",
        "https://sahayak-lac.vercel.app/"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Include authentication routes (for OTP login, but not for endpoint protection)
app.include_router(auth_router)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

# --- GeminiEmbedding Class (No changes here) ---
class GeminiEmbedding:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
    
    def embed_documents(self, texts):
        return self._embed_texts(texts, task_type="retrieval_document")
    
    def embed_query(self, text):
        result = self._embed_texts([text], task_type="retrieval_query")
        return result[0] if result else [0.0] * 768
    
    def _embed_texts(self, texts, task_type="retrieval_query"):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for text in texts:
            try:
                res = genai.embed_content(
                    model="models/embedding-001",
                    content=text,
                    task_type=task_type
                )
                embeddings.append(res["embedding"])
            except Exception as e:
                print(f"Gemini embedding error: {e}")
                embeddings.append([0.0] * 768)
        return embeddings

embedding = GeminiEmbedding(GEMINI_API_KEY)
docsearch = PineconeVectorStore.from_existing_index(index_name="bot", embedding=embedding)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

client = Groq(api_key=GROQ_API_KEY)

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[MONGO_DB_NAME]
collection = db[COLLECTION_NAME]

# --- Helper functions (generate_completion, retrieve_context, save_to_database) - No changes here ---
def generate_completion(prompt_template, user_message: str):
    # ... (same as before)
    try:
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
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq API error: {e}")
        return "I'm sorry, I encountered an error while generating the response."

def retrieve_context(query: str) -> str:
    # ... (same as before)
    try:
        docs = retriever.invoke(query)
        if not docs:
            return ""
        return "\n".join(doc.page_content for doc in docs)
    except Exception as e:
        print(f"Context retrieval error: {e}")
        return ""

def save_to_database(entry_type: str, user_message: str, bot_response: str, user_id: str = "public_user"):
    # ... (same as before, but with a default user_id)
    try:
        chat_entry = {
            "type": entry_type,
            "user_message": user_message,
            "bot_response": bot_response,
            "user_id": user_id,
            "timestamp": datetime.utcnow()
        }
        collection.insert_one(chat_entry)
        print(f"Saved {entry_type} interaction to database for user: {user_id}")
    except Exception as e:
        print(f"Database save error: {e}")
        
# --- Root and Health endpoints (No changes here) ---
@app.get("/")
async def root():
    # ... (same as before)
    return { "message": "🎓 Sahayak API - AI-Powered Educational Revolution" }

@app.get("/health")
async def health_check():
    # ... (same as before)
    return { "status": "healthy" }


# --- REMOVED JWT PROTECTION FROM ENDPOINTS ---

@app.post("/worksheet")
async def generate_worksheet(msg: str = Form(...), difficulty: str = Form("Medium")):
    """Generate educational worksheet (Public Endpoint)"""
    print(f"Worksheet Topic: {msg}, Difficulty: {difficulty}, User: public_user")
    
    try:
        context = retrieve_context(msg)
        if not context:
            raise HTTPException(status_code=404, detail="No relevant content found")
            
        user_prompt = f"Topic: {msg}\nDifficulty Level: {difficulty}\nContext: {context}"
        response = generate_completion(system_prompt, user_prompt)
        
        save_to_database("worksheet", f"{msg} ({difficulty})", response)
        
        return JSONResponse(content={"worksheet": response, "topic": msg, "difficulty": difficulty})
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/video-script")
async def generate_video_script(msg: str = Form(...)):
    """Generate 30-second video script (Public Endpoint)"""
    print(f"Video Script Topic: {msg}, User: public_user")
    
    try:
        context = retrieve_context(msg)
        if not context:
            raise HTTPException(status_code=404, detail="No relevant content found")
        
        user_prompt = f"Topic: {msg}\nContext: {context}"
        response = generate_completion(video_script_prompt, user_prompt)
        
        save_to_database("video_script", msg, response)
        
        return JSONResponse(content={"script": response, "topic": msg})
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class QARequest(BaseModel):
    question: str

@app.post("/qa")
async def generate_answer(data: QARequest):
    """Generate answer for user questions using RAG (Public Endpoint)"""
    print(f"Q/A Question: {data.question}, User: public_user")
    
    try:
        context = retrieve_context(data.question)
        user_prompt = f"Question: {data.question}\nContext: {context}"
        response = generate_completion(qa_prompt, user_prompt)
        
        save_to_database("qa", data.question, response)
        
        return JSONResponse(content={"answer": response, "question": data.question})
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- History and Stats endpoints now operate without user context or are simplified ---

@app.get("/history")
async def chat_history(limit: int = 10):
    """Retrieve recent chat history (Public, not user-specific)"""
    try:
        history = list(collection.find().sort("timestamp", -1).limit(limit))
        for h in history:
            h["_id"] = str(h["_id"])
            h["timestamp"] = h["timestamp"].isoformat()
        return JSONResponse(content={"history": history})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get global usage statistics (Public Endpoint)"""
    try:
        total_interactions = collection.count_documents({})
        total_worksheets = collection.count_documents({"type": "worksheet"})
        total_qa = collection.count_documents({"type": "qa"})
        total_videos = collection.count_documents({"type": "video_script"})
        
        return JSONResponse(content={
            "global_stats": {
                "total_interactions": total_interactions,
                "worksheets_generated": total_worksheets,
                "questions_answered": total_qa,
                "video_scripts_generated": total_videos
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
# The demo endpoint can now be removed or merged with the main /qa endpoint
# if you want to keep it, it will work as is.

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
