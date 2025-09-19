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
import uvicorn
from src.helper import load_pdf_file, text_split
from src.prompt import system_prompt, qa_prompt, video_script_prompt
from pinecone.grpc import PineconeGRPC as Pinecone
from fastapi import BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import asyncio
import uuid

# Add these imports for video generation
from src.video_generator import video_service
from src.video_tasks import create_video_task_manager



app = FastAPI()

# FIXED CORS CONFIGURATION
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

video_task_manager = create_video_task_manager(video_service)

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

try:
    from src.video_generator import video_service
    VIDEO_GENERATION_AVAILABLE = True
    print("✅ Video generation service loaded")
except ImportError as e:
    VIDEO_GENERATION_AVAILABLE = False
    print(f"⚠️ Video generation not available: {e}")

# Add these models after your existing Pydantic models
class VideoGenerationRequest(BaseModel):
    topic: str
    duration: int = 30
    style: str = "educational"

class VideoStatusResponse(BaseModel):
    job_id: str
    status: str
    progress_percentage: int
    topic: str
    video_title: Optional[str] = None
    final_video_url: Optional[str] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    learning_objective: Optional[str] = None

# Add these endpoints after your existing endpoints

@app.get("/check-video-dependencies")
async def check_video_dependencies():
    """Check which video generation dependencies are available"""
    if not VIDEO_GENERATION_AVAILABLE:
        return JSONResponse(content={
            "video_generation_available": False,
            "error": "Video generation service not loaded",
            "install_commands": [
                "pip install moviepy>=1.0.3",
                "pip install Pillow>=9.0.0", 
                "pip install supabase>=1.0.0",
                "pip install pyttsx3>=2.90"
            ]
        })
    
    try:
        dependencies = video_service.check_dependencies()
        
        missing_deps = []
        if not dependencies["moviepy"]:
            missing_deps.append("pip install moviepy>=1.0.3")
        if not dependencies["pillow"]:
            missing_deps.append("pip install Pillow>=9.0.0")
        if not dependencies["supabase"]:
            missing_deps.append("pip install supabase>=1.0.0")
        if not dependencies["tts"]:
            missing_deps.append("pip install pyttsx3>=2.90")
        
        return JSONResponse(content={
            "video_generation_available": VIDEO_GENERATION_AVAILABLE,
            "dependencies": dependencies,
            "missing_dependencies": missing_deps,
            "ready_for_video_generation": all(dependencies.values()),
            "can_generate_scripts": dependencies["gemini"],
            "can_create_images": dependencies["pillow"],
            "can_generate_audio": dependencies["tts"],
            "can_assemble_video": dependencies["moviepy"],
            "has_storage": dependencies["supabase"],
            "has_database": dependencies["mongodb"]
        })
        
    except Exception as e:
        return JSONResponse(
            content={"error": f"Dependency check failed: {str(e)}"},
            status_code=500
        )

@app.post("/generate-video")
async def generate_video(request: VideoGenerationRequest):
    """Generate educational video (with graceful degradation)"""
    if not VIDEO_GENERATION_AVAILABLE:
        return JSONResponse(
            content={
                "error": "Video generation not available",
                "install_command": "pip install moviepy Pillow supabase pyttsx3"
            },
            status_code=503
        )
    
    print(f"🎬 Video generation request: {request.topic}")
    
    try:
        # Check what's available
        dependencies = video_service.check_dependencies()
        
        # Retrieve context from your existing RAG
        context = retrieve_context(request.topic)
        
        if not context:
            return JSONResponse(
                content={"error": "No relevant educational content found for this topic"},
                status_code=404
            )
        
        # Generate script (this should work even with missing dependencies)
        script_data = await video_service.generate_video_script(request.topic, context)
        
        if not script_data:
            return JSONResponse(
                content={"error": "Failed to generate video script"},
                status_code=500
            )
        
        # Create job ID
        job_id = str(uuid.uuid4())
        
        # Save to your existing database
        save_to_database("video_generation", request.topic, f"Script generated with ID: {job_id}")
        
        response_data = {
            "job_id": job_id,
            "status": "script_generated",
            "topic": request.topic,
            "video_title": script_data.get("video_title"),
            "learning_objective": script_data.get("learning_objective"),
            "segments_count": len(script_data.get("segments", [])),
            "timestamp": datetime.utcnow().isoformat(),
            "available_features": dependencies
        }
        
        # Add warnings for missing features
        warnings = []
        if not dependencies["moviepy"]:
            warnings.append("Video assembly disabled - install moviepy")
        if not dependencies["pillow"]:
            warnings.append("Image generation disabled - install Pillow")
        if not dependencies["tts"]:
            warnings.append("Audio generation disabled - install pyttsx3")
        if not dependencies["supabase"]:
            warnings.append("Cloud storage disabled - configure Supabase")
        
        if warnings:
            response_data["warnings"] = warnings
            response_data["install_commands"] = [
                "pip install moviepy>=1.0.3",
                "pip install Pillow>=9.0.0",
                "pip install pyttsx3>=2.90",
                "pip install supabase>=1.0.0"
            ]
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        print(f"Video generation error: {e}")
        return JSONResponse(
            content={"error": f"Video generation failed: {str(e)}"},
            status_code=500
        )

@app.post("/test-video-pipeline")
async def test_video_pipeline():
    """Test video generation pipeline with current dependencies"""
    if not VIDEO_GENERATION_AVAILABLE:
        return JSONResponse(content={
            "status": "service_unavailable",
            "message": "Video generation service not loaded",
            "fix": "Install missing dependencies and restart"
        })
    
    try:
        # Test basic functionality
        test_topic = "photosynthesis"
        test_context = "Photosynthesis is the process by which plants make food using sunlight, water, and carbon dioxide."
        
        # Check dependencies
        dependencies = video_service.check_dependencies()
        
        # Test script generation
        script = await video_service.generate_video_script(test_topic, test_context)
        
        result = {
            "status": "partial_success" if script else "failed",
            "dependencies_status": dependencies,
            "script_generation": "✅ Working" if script else "❌ Failed",
            "services_available": {
                "gemini_api": "✅ Working" if dependencies["gemini"] else "❌ Not configured",
                "image_processing": "✅ Ready" if dependencies["pillow"] else "❌ Install Pillow",
                "audio_generation": "✅ Ready" if dependencies["tts"] else "❌ Install pyttsx3", 
                "video_assembly": "✅ Ready" if dependencies["moviepy"] else "❌ Install MoviePy",
                "cloud_storage": "✅ Ready" if dependencies["supabase"] else "⚠️ Local storage only",
                "database": "✅ Working" if dependencies["mongodb"] else "⚠️ Limited functionality"
            },
            "next_steps": []
        }
        
        if script:
            result["script_preview"] = {
                "title": script.get("video_title"),
                "segments_count": len(script.get("segments", [])),
                "learning_objective": script.get("learning_objective")
            }
        
        # Add recommendations
        if not dependencies["moviepy"]:
            result["next_steps"].append("Install MoviePy: pip install moviepy>=1.0.3")
        if not dependencies["pillow"]:
            result["next_steps"].append("Install Pillow: pip install Pillow>=9.0.0")
        if not dependencies["tts"]:
            result["next_steps"].append("Install TTS: pip install pyttsx3>=2.90")
        if not dependencies["supabase"]:
            result["next_steps"].append("Configure Supabase for cloud storage")
        
        if not result["next_steps"]:
            result["next_steps"].append("All dependencies ready! Try generating a full video.")
            result["status"] = "fully_ready"
        
        return JSONResponse(content=result)
        
    except Exception as e:
        return JSONResponse(
            content={
                "status": "error",
                "error": f"Pipeline test failed: {str(e)}",
                "fix": "Check your environment variables and dependencies"
            },
            status_code=500
        )

@app.get("/video-status/{job_id}")
async def get_video_status(job_id: str):
    """Get the current status of video generation job"""
    if not VIDEO_GENERATION_AVAILABLE:
        return JSONResponse(
            content={"error": "Video generation service not available"},
            status_code=503
        )
    
    try:
        job_data = video_service.get_video_job(job_id)
        
        if not job_data:
            return JSONResponse(
                content={"error": "Job not found"},
                status_code=404
            )
        
        # Check dependencies for current status
        dependencies = video_service.check_dependencies()
        
        response = {
            "job_id": job_data.get("job_id"),
            "status": job_data.get("status", "unknown"),
            "progress_percentage": job_data.get("progress_percentage", 0),
            "topic": job_data.get("topic"),
            "video_title": job_data.get("video_title"),
            "final_video_url": job_data.get("final_video_url"),
            "error_message": job_data.get("error_message"),
            "created_at": job_data.get("created_at"),
            "learning_objective": job_data.get("learning_objective"),
            "available_features": dependencies
        }
        
        # Add helpful messages based on status and dependencies
        if response["status"] == "script_generated" and not dependencies["moviepy"]:
            response["message"] = "Script ready! Install MoviePy to continue with video generation."
            response["install_command"] = "pip install moviepy>=1.0.3"
        
        return JSONResponse(content=response)
        
    except Exception as e:
        print(f"Status check error: {e}")
        return JSONResponse(
            content={"error": "Failed to get job status"},
            status_code=500
        )
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

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Sahayak API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

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

# @app.post("/generate-video")
# async def generate_video(request: VideoGenerationRequest, background_tasks: BackgroundTasks):
#     """
#     Generate educational video from topic using RAG + AI
#     Returns immediately with job_id for status tracking
#     """
#     print(f"🎬 Video generation request: {request.topic}")
    
#     try:
#         # Retrieve context from RAG
#         context = retrieve_context(request.topic)
        
#         if not context:
#             return JSONResponse(
#                 content={"error": "No relevant educational content found for this topic"},
#                 status_code=404
#             )
        
#         # Start video generation in background
#         job_id = await video_task_manager.generate_full_video(request.topic, context)
        
#         # Save initial request to your existing database
#         save_to_database("video_generation", request.topic, f"Job started with ID: {job_id}")
        
#         return JSONResponse(content={
#             "job_id": job_id,
#             "status": "processing",
#             "message": "Video generation started successfully",
#             "topic": request.topic,
#             "estimated_time": "2-3 minutes",
#             "timestamp": datetime.utcnow().isoformat()
#         })
        
#     except Exception as e:
#         print(f"Video generation request error: {e}")
#         return JSONResponse(
#             content={"error": "Failed to start video generation"},
#             status_code=500
#         )

# @app.get("/video-status/{job_id}")
# async def get_video_status(job_id: str):
#     """
#     Get the current status of video generation job
#     """
#     try:
#         job_data = video_service.get_video_job(job_id)
        
#         if not job_data:
#             return JSONResponse(
#                 content={"error": "Job not found"},
#                 status_code=404
#             )
        
#         response = VideoStatusResponse(
#             job_id=job_data["job_id"],
#             status=job_data["status"],
#             progress_percentage=job_data.get("progress_percentage", 0),
#             topic=job_data["topic"],
#             video_title=job_data.get("video_title"),
#             final_video_url=job_data.get("final_video_url"),
#             error_message=job_data.get("error_message"),
#             created_at=job_data.get("created_at"),
#             learning_objective=job_data.get("learning_objective")
#         )
        
#         return JSONResponse(content=response.dict())
        
#     except Exception as e:
#         print(f"Status check error: {e}")
#         return JSONResponse(
#             content={"error": "Failed to get job status"},
#             status_code=500
#         )

# @app.get("/video-history")
# async def get_video_history(limit: int = 10):
#     """
#     Get recent video generation history
#     """
#     try:
#         videos = list(
#             video_service.video_jobs_collection
#             .find()
#             .sort("created_at", -1)
#             .limit(limit)
#         )
        
#         # Convert ObjectId and datetime to string
#         for video in videos:
#             video["_id"] = str(video["_id"])
#             if "created_at" in video:
#                 video["created_at"] = video["created_at"].isoformat()
#             if "updated_at" in video:
#                 video["updated_at"] = video["updated_at"].isoformat()
        
#         return JSONResponse(content={
#             "videos": videos,
#             "count": len(videos)
#         })
        
#     except Exception as e:
#         print(f"Video history error: {e}")
#         return JSONResponse(
#             content={"error": "Failed to retrieve video history"},
#             status_code=500
#         )

# @app.delete("/video-job/{job_id}")
# async def delete_video_job(job_id: str):
#     """
#     Delete a video generation job and its assets
#     """
#     try:
#         # Get job data first
#         job_data = video_service.get_video_job(job_id)
        
#         if not job_data:
#             return JSONResponse(
#                 content={"error": "Job not found"},
#                 status_code=404
#             )
        
#         # Delete from database
#         result = video_service.video_jobs_collection.delete_one({"job_id": job_id})
        
#         # TODO: Also delete associated files from Supabase storage
#         # This would require additional cleanup logic
        
#         if result.deleted_count > 0:
#             return JSONResponse(content={
#                 "message": f"Video job {job_id} deleted successfully",
#                 "timestamp": datetime.utcnow().isoformat()
#             })
#         else:
#             return JSONResponse(
#                 content={"error": "Job not found"},
#                 status_code=404
#             )
        
#     except Exception as e:
#         print(f"Video deletion error: {e}")
#         return JSONResponse(
#             content={"error": "Failed to delete video job"},
#             status_code=500
#         )

# @app.get("/video-stats")
# async def get_video_stats():
#     """
#     Get video generation statistics
#     """
#     try:
#         total_jobs = video_service.video_jobs_collection.count_documents({})
#         completed_jobs = video_service.video_jobs_collection.count_documents({"status": "completed"})
#         failed_jobs = video_service.video_jobs_collection.count_documents({"status": "failed"})
#         processing_jobs = video_service.video_jobs_collection.count_documents({
#             "status": {"$in": ["processing", "initializing", "generating_assets", "creating_video"]}
#         })
        
#         # Get recent activity
#         recent_completed = list(
#             video_service.video_jobs_collection
#             .find({"status": "completed"})
#             .sort("created_at", -1)
#             .limit(5)
#         )
        
#         for job in recent_completed:
#             job["_id"] = str(job["_id"])
#             if "created_at" in job:
#                 job["created_at"] = job["created_at"].isoformat()
        
#         return JSONResponse(content={
#             "total_video_jobs": total_jobs,
#             "completed_videos": completed_jobs,
#             "failed_videos": failed_jobs,
#             "processing_videos": processing_jobs,
#             "success_rate": round((completed_jobs / total_jobs * 100) if total_jobs > 0 else 0, 2),
#             "recent_completed": recent_completed,
#             "timestamp": datetime.utcnow().isoformat()
#         })
        
#     except Exception as e:
#         print(f"Video stats error: {e}")
#         return JSONResponse(
#             content={"error": "Failed to retrieve video statistics"},
#             status_code=500
#         )

# # Add this route for testing the video generation system
# @app.post("/test-video-pipeline")
# async def test_video_pipeline():
#     """
#     Test endpoint to verify video generation pipeline
#     """
#     try:
#         test_topic = "photosynthesis"
#         test_context = """
#         Photosynthesis is the process by which plants make their own food using sunlight, water, and carbon dioxide.
#         The process occurs in the chloroplasts of plant cells, specifically in the chlorophyll.
#         The equation is: 6CO2 + 6H2O + sunlight → C6H12O6 + 6O2
#         This process is essential for life on Earth as it produces oxygen and glucose.
#         """
        
#         # Test script generation
#         script = await video_service.generate_video_script(test_topic, test_context)
        
#         if script:
#             return JSONResponse(content={
#                 "status": "success",
#                 "message": "Video pipeline test completed successfully",
#                 "script_preview": {
#                     "title": script.get("video_title"),
#                     "segments_count": len(script.get("segments", [])),
#                     "learning_objective": script.get("learning_objective")
#                 },
#                 "services_status": {
#                     "gemini_api": "✅ Working",
#                     "supabase": "✅ Connected" if video_service.supabase else "❌ Not configured",
#                     "tts_engine": "✅ Ready",
#                     "mongodb": "✅ Connected"
#                 }
#             })
#         else:
#             return JSONResponse(
#                 content={"error": "Script generation failed"},
#                 status_code=500
#             )
            
#     except Exception as e:
#         return JSONResponse(
#             content={
#                 "error": f"Pipeline test failed: {str(e)}",
#                 "services_status": {
#                     "gemini_api": "❌ Error",
#                     "supabase": "❓ Unknown",
#                     "tts_engine": "❓ Unknown",
#                     "mongodb": "❓ Unknown"
#                 }
#             },
#             status_code=500
#         )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)

