# app.py - Complete updated version with conversation memory and user context

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

app = FastAPI(
    title="Sahayak - AI Educational Platform (Public API)",
    description="AI-Powered Educational Revolution for 260 Million Students - Now with Memory & Context",
    version="1.2.0"
)

# CORS CONFIGURATION
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://sahayak.me",
        "https://front-eight-murex.vercel.app",
        "https://sahayak-cizr.vercel.app",
        "https://www.sahayak.me"
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

# --- GeminiEmbedding Class ---
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
user_profiles = db["user_profiles"]  # New collection for user data

# --- Helper functions ---
def generate_completion(prompt_template, user_message: str):
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
    try:
        docs = retriever.invoke(query)
        if not docs:
            return ""
        return "\n".join(doc.page_content for doc in docs)
    except Exception as e:
        print(f"Context retrieval error: {e}")
        return ""

def save_to_database(entry_type: str, user_message: str, bot_response: str, user_id: str = "public_user"):
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

def get_conversation_history(user_id: str = "public_user", limit: int = 5) -> str:
    """Get recent conversation history for context"""
    try:
        history = list(collection.find(
            {"user_id": user_id, "type": "qa"}
        ).sort("timestamp", -1).limit(limit))
        
        if not history:
            return ""
        
        context_parts = []
        for chat in reversed(history):  # Reverse to show chronological order
            context_parts.append(f"Human: {chat['user_message']}")
            context_parts.append(f"Assistant: {chat['bot_response']}")
        
        return "\n".join(context_parts)
        
    except Exception as e:
        print(f"History retrieval error: {e}")
        return ""

def get_user_profile(user_id: str) -> dict:
    """Get user profile information"""
    try:
        profile = user_profiles.find_one({"user_id": user_id})
        return profile if profile else {}
    except Exception as e:
        print(f"User profile retrieval error: {e}")
        return {}

def update_user_profile(user_id: str, update_data: dict):
    """Update user profile information"""
    try:
        user_profiles.update_one(
            {"user_id": user_id},
            {"$set": {**update_data, "updated_at": datetime.utcnow()}},
            upsert=True
        )
        print(f"Updated profile for user: {user_id}")
    except Exception as e:
        print(f"User profile update error: {e}")

def extract_user_info(message: str, response: str) -> dict:
    """Extract user information from conversation"""
    info = {}
    
    # Extract name
    name_patterns = [
        "my name is", "i'm", "i am", "call me", "name's"
    ]
    
    for pattern in name_patterns:
        if pattern in message.lower():
            words = message.lower().split()
            try:
                idx = words.index(pattern.split()[-1])
                if idx + 1 < len(words):
                    potential_name = words[idx + 1].strip(".,!?").title()
                    if potential_name.isalpha() and len(potential_name) > 1:
                        info["name"] = potential_name
                        break
            except (ValueError, IndexError):
                continue
    
    # Extract grade/class
    grade_patterns = ["grade", "class", "standard", "std"]
    for pattern in grade_patterns:
        if pattern in message.lower():
            words = message.lower().split()
            for i, word in enumerate(words):
                if pattern in word and i + 1 < len(words):
                    try:
                        grade_num = int(words[i + 1])
                        if 1 <= grade_num <= 12:
                            info["grade"] = grade_num
                            break
                    except ValueError:
                        continue
    
    # Extract interests and hobbies
    interest_keywords = ["like", "love", "enjoy", "interested in", "favorite", "play", "watch"]
    
    # Academic subjects
    subjects = ["math", "science", "physics", "chemistry", "biology", "english", "history"]
    
    # Sports and activities
    sports = ["cricket", "football", "basketball", "tennis", "badminton", "swimming", "running"]
    
    # Tech and gaming
    tech = ["gaming", "computer", "coding", "programming", "phone", "apps", "youtube"]
    
    # Entertainment
    entertainment = ["movies", "music", "dancing", "singing", "drawing", "painting"]
    
    # Food and cooking
    food = ["cooking", "baking", "food", "eating"]
    
    all_interests = subjects + sports + tech + entertainment + food
    
    for keyword in interest_keywords:
        if keyword in message.lower():
            for interest in all_interests:
                if interest in message.lower():
                    if "interests" not in info:
                        info["interests"] = []
                    if interest not in info["interests"]:
                        info["interests"].append(interest)
    
    # Extract location/city for local examples
    city_keywords = ["from", "live in", "city", "town"]
    indian_cities = ["delhi", "mumbai", "bangalore", "chennai", "kolkata", "pune", "hyderabad", "ahmedabad"]
    
    for keyword in city_keywords:
        if keyword in message.lower():
            for city in indian_cities:
                if city in message.lower():
                    info["city"] = city.title()
                    break
    
    return info

def generate_personalized_context(user_profile: dict) -> str:
    """Generate personalized context string for the prompt"""
    context_parts = []
    
    if user_profile.get("name"):
        context_parts.append(f"User's name: {user_profile['name']}")
    
    if user_profile.get("grade"):
        grade = user_profile['grade']
        context_parts.append(f"User's grade: {grade}")
        
        # Add age-appropriate context
        if grade <= 5:
            context_parts.append("Use simple examples like toys, cartoons, family activities")
        elif grade <= 8:
            context_parts.append("Use examples like school activities, sports, simple technology")
        else:
            context_parts.append("Use examples like social media, complex technology, current events")
    
    if user_profile.get("interests"):
        interests = user_profile['interests']
        context_parts.append(f"User's interests: {', '.join(interests)}")
        
        # Add specific example guidance based on interests
        example_guidance = []
        
        if any(sport in interests for sport in ["cricket", "football", "basketball", "tennis", "badminton"]):
            example_guidance.append("Use sports examples (ball physics, momentum, energy)")
        
        if any(tech in interests for tech in ["gaming", "computer", "coding", "phone", "apps"]):
            example_guidance.append("Use technology examples (circuits, electricity, algorithms)")
        
        if any(subject in interests for subject in ["math", "science", "physics", "chemistry"]):
            example_guidance.append("Use academic examples and experiments they might enjoy")
        
        if any(food in interests for food in ["cooking", "baking", "food"]):
            example_guidance.append("Use cooking/kitchen examples (heat, chemical reactions, mixing)")
        
        if example_guidance:
            context_parts.append(f"Example types to use: {'; '.join(example_guidance)}")
    
    if user_profile.get("city"):
        context_parts.append(f"User's city: {user_profile['city']} (use local examples when relevant)")
    
    return "\n".join(context_parts) if context_parts else ""

# --- Pydantic Models ---
class QARequest(BaseModel):
    question: str
    user_id: str = "public_user"

class UserProfileRequest(BaseModel):
    user_id: str
    name: str = None
    grade: int = None
    interests: list = None

# --- Root and Health endpoints ---
@app.get("/")
async def root():
    return {"message": "🎓 Sahayak API - AI-Powered Educational Revolution with Memory"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# --- MAIN QA ENDPOINT WITH MEMORY AND CONTEXT ---
@app.post("/qa")
async def generate_answer(data: QARequest):
    """Generate answer with conversation history and user context"""
    print(f"Q/A Question: {data.question}, User: {data.user_id}")
    
    try:
        # Get RAG context from documents
        rag_context = retrieve_context(data.question)
        
        # Get conversation history for context
        conversation_history = get_conversation_history(data.user_id, limit=4)
        
        # Get user profile for personalization
        user_profile = get_user_profile(data.user_id)
        
        # Generate personalized context
        personalized_context = generate_personalized_context(user_profile)
        
        # Build comprehensive prompt with all contexts
        if conversation_history or personalized_context:
            user_prompt = f"""User Information & Personalization Guidelines:
{personalized_context}

Previous Conversation:
{conversation_history}

Current Question: {data.question}

Document Context: {rag_context}

Remember: Use the user's name, interests, and grade level to give relatable real-world examples!"""
        else:
            user_prompt = f"Question: {data.question}\nContext: {rag_context}"
        
        response = generate_completion(qa_prompt, user_prompt)
        
        # Extract and save user information if found
        extracted_info = extract_user_info(data.question, response)
        if extracted_info:
            current_profile = get_user_profile(data.user_id)
            # Merge with existing profile
            for key, value in extracted_info.items():
                if key == "interests":
                    existing_interests = current_profile.get("interests", [])
                    new_interests = list(set(existing_interests + value))
                    extracted_info[key] = new_interests
            update_user_profile(data.user_id, extracted_info)
        
        # Save conversation to database
        save_to_database("qa", data.question, response, data.user_id)
        
        return JSONResponse(content={
            "answer": response, 
            "question": data.question,
            "has_history": bool(conversation_history),
            "user_profile": get_user_profile(data.user_id)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- USER PROFILE ENDPOINTS ---
@app.get("/profile/{user_id}")
async def get_profile(user_id: str):
    """Get user profile"""
    try:
        profile = get_user_profile(user_id)
        return JSONResponse(content={"profile": profile, "user_id": user_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/profile")
async def update_profile(data: UserProfileRequest):
    """Update user profile"""
    try:
        update_data = {}
        if data.name:
            update_data["name"] = data.name
        if data.grade:
            update_data["grade"] = data.grade
        if data.interests:
            update_data["interests"] = data.interests
        
        if update_data:
            update_user_profile(data.user_id, update_data)
        
        return JSONResponse(content={"message": "Profile updated successfully"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- CONVERSATION HISTORY ENDPOINT ---
@app.get("/history/{user_id}")
async def get_user_history(user_id: str, limit: int = 10):
    """Get conversation history for specific user"""
    try:
        history = list(collection.find(
            {"user_id": user_id, "type": "qa"}
        ).sort("timestamp", -1).limit(limit))
        
        for h in history:
            h["_id"] = str(h["_id"])
            h["timestamp"] = h["timestamp"].isoformat()
            
        return JSONResponse(content={"history": history, "user_id": user_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- OTHER ENDPOINTS (Worksheet, Video Script, etc.) ---
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
        total_users = user_profiles.count_documents({})
        
        return JSONResponse(content={
            "global_stats": {
                "total_interactions": total_interactions,
                "worksheets_generated": total_worksheets,
                "questions_answered": total_qa,
                "video_scripts_generated": total_videos,
                "registered_users": total_users
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)