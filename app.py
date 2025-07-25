import os
from dotenv import load_dotenv
from datetime import datetime
from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_pinecone import PineconeVectorStore
from pymongo import MongoClient
from groq import Groq
import google.generativeai as genai
from src.helper import load_pdf_file, text_split
from src.prompt import system_prompt, qa_prompt, video_script_prompt
from pinecone.grpc import PineconeGRPC as Pinecone

# ----------------------------
# Load Environment
# ----------------------------
<<<<<<< HEAD
=======
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
>>>>>>> bfcdded (change in cors link)
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://sahayak-cizr.vercel.app",
        "http://localhost:3000",
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Improved Gemini Embedding Wrapper
# ----------------------------
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

# ----------------------------
# Setup: Pinecone + Embeddings + Groq + Mongo
# ----------------------------
embedding = GeminiEmbedding(GEMINI_API_KEY)
docsearch = PineconeVectorStore.from_existing_index(index_name="bot", embedding=embedding)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

client = Groq(api_key=GROQ_API_KEY)

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[MONGO_DB_NAME]
collection = db[COLLECTION_NAME]

# ----------------------------
# Utility Functions
# ----------------------------
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

def save_to_database(entry_type: str, user_message: str, bot_response: str):
    """Save interaction to MongoDB"""
    try:
        chat_entry = {
            "type": entry_type,
            "user_message": user_message,
            "bot_response": bot_response,
            "timestamp": datetime.utcnow()
        }
        collection.insert_one(chat_entry)
        print(f"Saved {entry_type} interaction to database")
    except Exception as e:
        print(f"Database save error: {e}")

# ----------------------------
# ROUTES
# ----------------------------
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Sahayak API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# ----------------------------
# Worksheet Generation Endpoint
# ----------------------------
@app.post("/worksheet")
async def generate_worksheet(msg: str = Form(...), difficulty: str = Form("Medium")):
    """Generate educational worksheet based on topic and difficulty level"""
    print(f"Worksheet Topic: {msg}, Difficulty: {difficulty}")
    
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
        
        # Save to database
        save_to_database("worksheet", f"{msg} ({difficulty})", response)
        
        return JSONResponse(content={
            "worksheet": response,
            "topic": msg,
            "difficulty": difficulty,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        print(f"Worksheet generation error: {e}")
        return JSONResponse(
            content={"error": "Failed to generate worksheet"},
            status_code=500
        )

# ----------------------------
# Video Script Generation Endpoint
# ----------------------------
@app.post("/video-script")
async def generate_video_script(msg: str = Form(...)):
    """Generate 30-second video script for educational content"""
    print(f"Video Script Topic: {msg}")
    
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
        
        # Save to database
        save_to_database("video_script", msg, response)
        
        return JSONResponse(content={
            "script": response,
            "topic": msg,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        print(f"Video script generation error: {e}")
        return JSONResponse(
            content={"error": "Failed to generate video script"},
            status_code=500
        )

# ----------------------------
# Chat History Endpoint
# ----------------------------
@app.get("/history")
async def chat_history(limit: int = 10):
    """Retrieve chat history from database"""
    try:
        history = list(collection.find().sort("timestamp", -1).limit(limit))
        
        # Convert ObjectId and datetime to string for JSON serialization
        for h in history:
            h["_id"] = str(h["_id"])
            h["timestamp"] = h["timestamp"].isoformat()
        
        return JSONResponse(content={
            "history": history,
            "count": len(history)
        })
        
    except Exception as e:
        print(f"History retrieval error: {e}")
        return JSONResponse(
            content={"error": "Failed to retrieve history"},
            status_code=500
        )

# ----------------------------
# Q/A Endpoint
# ----------------------------
class QARequest(BaseModel):
    question: str

@app.post("/qa")
async def generate_answer(data: QARequest):
    """Generate answer for user questions using RAG"""
    print(f"Q/A Question: {data.question}")
    
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
        
        # Save to database
        save_to_database("qa", data.question, response)
        
        return JSONResponse(content={
            "answer": response,
            "question": data.question,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        print(f"Q/A generation error: {e}")
        return JSONResponse(
            content={"error": "Failed to generate answer"},
            status_code=500
        )

# ----------------------------
# Database Management Endpoints
# ----------------------------
@app.delete("/history")
async def clear_history():
    """Clear all chat history"""
    try:
        result = collection.delete_many({})
        return JSONResponse(content={
            "message": f"Cleared {result.deleted_count} records",
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        print(f"History clearing error: {e}")
        return JSONResponse(
            content={"error": "Failed to clear history"},
            status_code=500
        )

@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    try:
        total_interactions = collection.count_documents({})
        worksheet_count = collection.count_documents({"type": "worksheet"})
        qa_count = collection.count_documents({"type": "qa"})
        video_script_count = collection.count_documents({"type": "video_script"})
        
        return JSONResponse(content={
            "total_interactions": total_interactions,
            "worksheets_generated": worksheet_count,
            "questions_answered": qa_count,
            "video_scripts_generated": video_script_count,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        print(f"Stats retrieval error: {e}")
        return JSONResponse(
            content={"error": "Failed to retrieve stats"},
            status_code=500
        )

# ----------------------------
# Run the application
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
