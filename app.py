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

# Import authentication components
from auth_routes import auth_router
from auth import get_current_user, get_current_active_user, auth_manager

app = FastAPI(
    title="Sahayak - AI Educational Platform",
    description="AI-Powered Educational Revolution for 260 Million Students",
    version="1.0.0"
)

# CORS CONFIGURATION
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",                     # Local development
        "http://127.0.0.1:3000",
        "https://sahayak.me",
        "https://front-eight-murex.vercel.app",     # Your current frontend URL
        "https://sahayak-cizr.vercel.app",          # Keep old one just in case
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Include authentication routes
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

class GeminiEmbedding:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
    
    def embed_documents(self, texts):
        """For document embedding during indexing"""
        return self._embed_texts(texts, task_type="retrieval_document")
    
    def embed_query(self, text):
        """For query embedding during search"""
        result = self._embed_texts([text], task_type="retrieval_query")
        return result[0] if result else [0.0] * 768
    
    def embed(self, texts):
        """Backward compatibility method"""
        return self._embed_texts(texts)
    
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

def generate_completion(prompt_template, user_message: str):
    """Generate completion using Groq API"""
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
    """Retrieve relevant context from Pinecone using Gemini embeddings"""
    try:
        docs = retriever.invoke(query)
        if not docs:
            return ""
        return "\n".join(doc.page_content for doc in docs)
    except Exception as e:
        print(f"Context retrieval error: {e}")
        return ""

def save_to_database(entry_type: str, user_message: str, bot_response: str, user_id: str = None):
    """Save interaction to MongoDB with user tracking"""
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

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🎓 Sahayak API - AI-Powered Educational Revolution",
        "status": "healthy",
        "version": "1.0.0",
        "endpoints": {
            "auth": "/auth/",
            "qa": "/qa",
            "worksheet": "/worksheet",
            "video_script": "/video-script",
            "stats": "/stats",
            "docs": "/docs"
        },
        "description": "Transforming Education for 260 Million Students Across India"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": "connected",
            "pinecone": "connected",
            "groq": "connected",
            "gemini": "connected"
        }
    }

# Protected endpoints requiring authentication
@app.post("/worksheet")
async def generate_worksheet(
    msg: str = Form(...), 
    difficulty: str = Form("Medium"),
    current_user: dict = Depends(get_current_active_user)
):
    """Generate educational worksheet based on topic and difficulty level (Protected)"""
    print(f"Worksheet Topic: {msg}, Difficulty: {difficulty}, User: {current_user['_id']}")
    
    try:
        # Retrieve context from vector store
        context = retrieve_context(msg)
        
        if not context:
            return JSONResponse(
                content={"error": "No relevant content found for this topic"},
                status_code=404
            )
        
        # Format the prompt
        user_prompt = f"""
Topic: {msg}
Difficulty Level: {difficulty}
Context: {context}
"""
        
        # Generate worksheet
        response = generate_completion(system_prompt, user_prompt)
        
        # Save to database with user tracking
        save_to_database("worksheet", f"{msg} ({difficulty})", response, current_user["_id"])
        
        return JSONResponse(content={
            "worksheet": response,
            "topic": msg,
            "difficulty": difficulty,
            "user_id": current_user["_id"],
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        print(f"Worksheet generation error: {e}")
        return JSONResponse(
            content={"error": "Failed to generate worksheet"},
            status_code=500
        )

@app.post("/video-script")
async def generate_video_script(
    msg: str = Form(...),
    current_user: dict = Depends(get_current_active_user)
):
    """Generate 30-second video script for educational content (Protected)"""
    print(f"Video Script Topic: {msg}, User: {current_user['_id']}")
    
    try:
        # Retrieve context from vector store
        context = retrieve_context(msg)
        
        if not context:
            return JSONResponse(
                content={"error": "No relevant content found for this topic"},
                status_code=404
            )
        
        # Format the prompt
        user_prompt = f"""
Topic: {msg}
Context: {context}
"""
        
        # Generate video script
        response = generate_completion(video_script_prompt, user_prompt)
        
        # Save to database with user tracking
        save_to_database("video_script", msg, response, current_user["_id"])
        
        return JSONResponse(content={
            "script": response,
            "topic": msg,
            "user_id": current_user["_id"],
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        print(f"Video script generation error: {e}")
        return JSONResponse(
            content={"error": "Failed to generate video script"},
            status_code=500
        )

class QARequest(BaseModel):
    question: str

@app.post("/qa")
async def generate_answer(
    data: QARequest,
    current_user: dict = Depends(get_current_active_user)
):
    """Generate answer for user questions using RAG (Protected)"""
    print(f"Q/A Question: {data.question}, User: {current_user['_id']}")
    
    try:
        # Retrieve relevant context
        context = retrieve_context(data.question)
        
        # Format the prompt
        user_prompt = f"""
Question: {data.question}
Context: {context}
"""
        
        # Generate answer
        response = generate_completion(qa_prompt, user_prompt)
        
        # Save to database with user tracking
        save_to_database("qa", data.question, response, current_user["_id"])
        
        return JSONResponse(content={
            "answer": response,
            "question": data.question,
            "user_id": current_user["_id"],
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        print(f"Q/A generation error: {e}")
        return JSONResponse(
            content={"error": "Failed to generate answer"},
            status_code=500
        )

# History endpoints (protected)
@app.get("/history")
async def chat_history(
    limit: int = 10,
    current_user: dict = Depends(get_current_active_user)
):
    """Retrieve user's chat history from database (Protected)"""
    try:
        # Get history for the current user only
        history = list(
            collection.find({"user_id": current_user["_id"]})
            .sort("timestamp", -1)
            .limit(limit)
        )
        
        # Convert ObjectId and datetime to string for JSON serialization
        for h in history:
            h["_id"] = str(h["_id"])
            h["timestamp"] = h["timestamp"].isoformat()
        
        return JSONResponse(content={
            "history": history,
            "count": len(history),
            "user_id": current_user["_id"]
        })
        
    except Exception as e:
        print(f"History retrieval error: {e}")
        return JSONResponse(
            content={"error": "Failed to retrieve history"},
            status_code=500
        )

@app.delete("/history")
async def clear_history(current_user: dict = Depends(get_current_active_user)):
    """Clear user's chat history (Protected)"""
    try:
        # Only clear history for the current user
        result = collection.delete_many({"user_id": current_user["_id"]})
        return JSONResponse(content={
            "message": f"Cleared {result.deleted_count} records for user {current_user['_id']}",
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        print(f"History clearing error: {e}")
        return JSONResponse(
            content={"error": "Failed to clear history"},
            status_code=500
        )

# Stats endpoints
@app.get("/stats")
async def get_stats(current_user: dict = Depends(get_current_active_user)):
    """Get user's usage statistics (Protected)"""
    try:
        user_id = current_user["_id"]
        
        # User-specific stats
        user_interactions = collection.count_documents({"user_id": user_id})
        user_worksheets = collection.count_documents({"user_id": user_id, "type": "worksheet"})
        user_qa = collection.count_documents({"user_id": user_id, "type": "qa"})
        user_videos = collection.count_documents({"user_id": user_id, "type": "video_script"})
        
        # Global stats (for admins or general info)
        total_interactions = collection.count_documents({})
        total_worksheets = collection.count_documents({"type": "worksheet"})
        total_qa = collection.count_documents({"type": "qa"})
        total_videos = collection.count_documents({"type": "video_script"})
        
        return JSONResponse(content={
            "user_stats": {
                "total_interactions": user_interactions,
                "worksheets_generated": user_worksheets,
                "questions_answered": user_qa,
                "video_scripts_generated": user_videos
            },
            "global_stats": {
                "total_interactions": total_interactions,
                "worksheets_generated": total_worksheets,
                "questions_answered": total_qa,
                "video_scripts_generated": total_videos
            },
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        print(f"Stats retrieval error: {e}")
        return JSONResponse(
            content={"error": "Failed to retrieve stats"},
            status_code=500
        )

# Admin-only stats endpoint
@app.get("/admin/stats")
async def get_admin_stats(current_user: dict = Depends(get_current_active_user)):
    """Get comprehensive admin statistics (Admin only)"""
    try:
        if current_user.get("role") != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        # Comprehensive admin statistics
        total_users = auth_manager.users_collection.count_documents({})
        verified_users = auth_manager.users_collection.count_documents({"is_verified": True})
        email_users = auth_manager.users_collection.count_documents({"login_type": "email"})
        phone_users = auth_manager.users_collection.count_documents({"login_type": "phone"})
        
        # Interaction statistics
        total_interactions = collection.count_documents({})
        worksheets = collection.count_documents({"type": "worksheet"})
        qa_sessions = collection.count_documents({"type": "qa"})
        video_scripts = collection.count_documents({"type": "video_script"})
        
        # Recent activity (last 24 hours)
        from datetime import timedelta
        yesterday = datetime.utcnow() - timedelta(days=1)
        recent_interactions = collection.count_documents({"timestamp": {"$gte": yesterday}})
        
        return JSONResponse(content={
            "admin_stats": {
                "users": {
                    "total": total_users,
                    "verified": verified_users,
                    "email_login": email_users,
                    "phone_login": phone_users
                },
                "interactions": {
                    "total": total_interactions,
                    "worksheets": worksheets,
                    "qa_sessions": qa_sessions,
                    "video_scripts": video_scripts,
                    "recent_24h": recent_interactions
                }
            },
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        print(f"Admin stats retrieval error: {e}")
        return JSONResponse(
            content={"error": "Failed to retrieve admin stats"},
            status_code=500
        )

# Public demo endpoint (no authentication required)
@app.post("/demo/qa")
async def demo_qa(data: QARequest):
    """Demo Q&A endpoint (No authentication required)"""
    try:
        context = retrieve_context(data.question)
        user_prompt = f"""
Question: {data.question}
Context: {context}
"""
        
        response = generate_completion(qa_prompt, user_prompt)
        
        # Save to database without user tracking (demo usage)
        save_to_database("demo_qa", data.question, response, "demo_user")
        
        return JSONResponse(content={
            "answer": response,
            "question": data.question,
            "demo": True,
            "timestamp": datetime.utcnow().isoformat(),
            "note": "This is a demo. Register for full features!"
        })
        
    except Exception as e:
        print(f"Demo Q/A error: {e}")
        return JSONResponse(
            content={"error": "Failed to generate demo answer"},
            status_code=500
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)