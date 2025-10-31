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
from typing import Optional
import asyncio
import uuid

# Load environment variables first
load_dotenv()

app = FastAPI(title="Sahayak API", description="AI-Powered Educational Platform", version="1.0.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://sahayak.me",
        "https://front-eight-murex.vercel.app",
        "https://sahayak-cizr.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Environment Variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# Validate required environment variables
if not PINECONE_API_KEY:
    print("⚠️ PINECONE_API_KEY not found in environment")
if not GROQ_API_KEY:
    print("⚠️ GROQ_API_KEY not found in environment")
if not GEMINI_API_KEY:
    print("⚠️ GEMINI_API_KEY not found in environment")
if not MONGO_URI:
    print("⚠️ MONGO_URI not found in environment")

# Set environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY or ""
os.environ["GROQ_API_KEY"] = GROQ_API_KEY or ""

# Configure AI services
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini configured")
else:
    print("❌ Gemini not configured - GEMINI_API_KEY missing")

# Initialize video generation service with error handling
try:
    from src.video_generator import video_service
    from src.video_tasks import create_video_task_manager
    
    if video_service:
        video_task_manager = create_video_task_manager(video_service)
        VIDEO_GENERATION_AVAILABLE = True
        print("✅ Video generation service loaded")
    else:
        VIDEO_GENERATION_AVAILABLE = False
        video_task_manager = None
        print("⚠️ Video service is None")
except ImportError as e:
    VIDEO_GENERATION_AVAILABLE = False
    video_task_manager = None
    print(f"⚠️ Video generation not available: {e}")
except Exception as e:
    VIDEO_GENERATION_AVAILABLE = False
    video_task_manager = None
    print(f"❌ Video service error: {e}")

class GeminiEmbedding:
    def __init__(self, api_key):
        if api_key:
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

# Initialize services with error handling
try:
    if GEMINI_API_KEY and PINECONE_API_KEY:
        embedding = GeminiEmbedding(GEMINI_API_KEY)
        docsearch = PineconeVectorStore.from_existing_index(index_name="bot", embedding=embedding)
        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        print("✅ Pinecone and embeddings configured")
    else:
        retriever = None
        print("❌ Pinecone/Embeddings not configured - missing API keys")
except Exception as e:
    retriever = None
    print(f"❌ Pinecone setup error: {e}")

try:
    if GROQ_API_KEY:
        client = Groq(api_key=GROQ_API_KEY)
        print("✅ Groq client configured")
    else:
        client = None
        print("❌ Groq not configured")
except Exception as e:
    client = None
    print(f"❌ Groq setup error: {e}")

try:
    if MONGO_URI and MONGO_DB_NAME and COLLECTION_NAME:
        mongo_client = MongoClient(MONGO_URI)
        db = mongo_client[MONGO_DB_NAME]
        collection = db[COLLECTION_NAME]
        # Test connection
        collection.count_documents({})
        print("✅ MongoDB configured")
    else:
        collection = None
        print("❌ MongoDB not configured - missing connection details")
except Exception as e:
    collection = None
    print(f"❌ MongoDB setup error: {e}")

# Pydantic Models
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

class QARequest(BaseModel):
    question: str

# Utility Functions
def generate_completion(prompt_template, user_message: str):
    """Generate completion using Groq API"""
    if not client:
        return "Groq API not configured. Please check your GROQ_API_KEY."
    
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
    if not retriever:
        return "Context retrieval not available. Please configure Pinecone and Gemini."
    
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
    if not collection:
        print(f"Database not available - would save: {entry_type}")
        return
    
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

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Sahayak API is running", 
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with service status"""
    services = {
        "gemini": GEMINI_API_KEY is not None,
        "groq": client is not None,
        "mongodb": collection is not None,
        "pinecone": retriever is not None,
        "video_generation": VIDEO_GENERATION_AVAILABLE
    }
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": services,
        "all_services_ready": all(services.values())
    }

@app.get("/check-video-dependencies")
async def check_video_dependencies():
    """Check video generation dependencies with enhanced error handling"""
    try:
        if not VIDEO_GENERATION_AVAILABLE or not video_service:
            return JSONResponse(content={
                "video_generation_available": False,
                "error": "Video generation service not loaded",
                "status": "service_unavailable",
                "install_commands": [
                    "sudo apt-get update && sudo apt-get install -y ffmpeg",
                    "pip install Pillow>=9.0.0", 
                    "pip install azure-cognitiveservices-speech>=1.35.0"
                ],
                "next_steps": [
                    "1. Run the install commands above",
                    "2. Restart the application",
                    "3. Test again with this endpoint"
                ]
            })
        
        try:
            dependencies = video_service.check_dependencies()
            
            # Get installation help
            installation_help = {}
            try:
                installation_help = video_service.get_installation_help()
            except Exception as help_error:
                print(f"Could not get installation help: {help_error}")
            
            missing_deps = []
            install_commands = []
            
            # Check each dependency with better error handling
            deps_to_check = {
                "ffmpeg": "FFmpeg for video processing",
                "pillow": "Pillow for image generation", 
                "tts": "Azure Text-to-Speech",
                "supabase": "Supabase for cloud storage",
                "mongodb": "MongoDB for job tracking",
                "gemini": "Google Gemini for AI features"
            }
            
            for dep_key, dep_name in deps_to_check.items():
                if not dependencies.get(dep_key, False):
                    missing_deps.append(dep_name)
                    
                    # Add specific install commands
                    if dep_key == "ffmpeg":
                        install_commands.extend([
                            "# Ubuntu/Debian:",
                            "sudo apt-get update && sudo apt-get install -y ffmpeg",
                            "# CentOS/RHEL:",
                            "sudo dnf install -y ffmpeg"
                        ])
                    elif dep_key == "pillow":
                        install_commands.append("pip install Pillow>=9.0.0")
                    elif dep_key == "tts":
                        install_commands.extend([
                            "pip install azure-cognitiveservices-speech>=1.35.0",
                            "# Set environment variables:",
                            "# AZURE_SPEECH_KEY=your_key",
                            "# AZURE_SPEECH_REGION=your_region"
                        ])
                    elif dep_key == "supabase":
                        install_commands.extend([
                            "pip install supabase",
                            "# Set environment variables:",
                            "# SUPABASE_URL=your_url", 
                            "# SUPABASE_KEY=your_key"
                        ])
            
            # Calculate readiness score
            total_deps = len(deps_to_check)
            ready_deps = sum(1 for dep in deps_to_check.keys() if dependencies.get(dep, False))
            readiness_score = (ready_deps / total_deps) * 100 if total_deps > 0 else 0
            
            return JSONResponse(content={
                "video_generation_available": VIDEO_GENERATION_AVAILABLE,
                "dependencies": dependencies,
                "missing_dependencies": missing_deps,
                "install_commands": install_commands,
                "installation_help": installation_help,
                "readiness_score": round(readiness_score, 1),
                "ready_for_video_generation": dependencies.get("ffmpeg", False) and dependencies.get("pillow", False),
                "system_info": {
                    "video_engine": "FFmpeg" if dependencies.get("ffmpeg") else "Not Available",
                    "can_generate_scripts": dependencies.get("gemini", False),
                    "can_create_images": dependencies.get("pillow", False),
                    "can_generate_audio": dependencies.get("tts", False),
                    "can_assemble_video": dependencies.get("ffmpeg", False),
                    "has_cloud_storage": dependencies.get("supabase", False),
                    "has_database": dependencies.get("mongodb", False)
                },
                "status": "ready" if readiness_score >= 80 else "needs_setup" if readiness_score >= 40 else "not_ready",
                "message": f"System is {readiness_score:.1f}% ready for video generation"
            })
            
        except Exception as dep_error:
            return JSONResponse(
                content={
                    "error": f"Dependency check failed: {str(dep_error)}",
                    "status": "check_failed",
                    "fix": "Check video service configuration"
                },
                status_code=500
            )
            
    except Exception as e:
        return JSONResponse(
            content={
                "error": f"Critical dependency check failure: {str(e)}",
                "status": "critical_error",
                "install_commands": [
                    "sudo apt-get update && sudo apt-get install -y ffmpeg",
                    "pip install Pillow>=9.0.0",
                    "pip install azure-cognitiveservices-speech>=1.35.0"
                ]
            },
            status_code=500
        )

@app.post("/test-video-pipeline")
async def test_video_pipeline():
    """Test video generation pipeline with comprehensive error handling"""
    try:
        # Check if video generation is available
        if not VIDEO_GENERATION_AVAILABLE:
            return JSONResponse(content={
                "status": "service_unavailable",
                "message": "Video generation service not loaded",
                "fix": "Install missing dependencies and restart",
                "install_commands": [
                    "sudo apt-get update && sudo apt-get install -y ffmpeg", 
                    "pip install Pillow>=9.0.0", 
                    "pip install azure-cognitiveservices-speech>=1.35.0"
                ]
            })
        
        # Check if video service exists
        if not video_service:
            return JSONResponse(content={
                "status": "service_error",
                "message": "Video service instance not available",
                "fix": "Restart the application after installing dependencies"
            })
        
        # Check if video task manager exists
        if not video_task_manager:
            return JSONResponse(content={
                "status": "task_manager_error", 
                "message": "Video task manager not initialized",
                "fix": "Check video service initialization"
            })
        
        # Test the pipeline with proper error handling
        try:
            # Check if the test method exists
            if not hasattr(video_task_manager, 'test_enhanced_pipeline'):
                # Fallback to basic test
                if hasattr(video_task_manager, 'test_video_pipeline'):
                    result = await video_task_manager.test_video_pipeline("photosynthesis")
                else:
                    # Create a basic test result
                    dependencies = video_service.check_dependencies()
                    result = {
                        "status": "basic_test",
                        "message": "Running basic dependency check",
                        "dependencies": dependencies,
                        "recommendations": []
                    }
                    
                    if not dependencies.get("ffmpeg", False):
                        result["recommendations"].append("Install FFmpeg: sudo apt-get install -y ffmpeg")
                    if not dependencies.get("pillow", False):
                        result["recommendations"].append("Install Pillow: pip install Pillow>=9.0.0")
                    if not dependencies.get("tts", False):
                        result["recommendations"].append("Configure Azure TTS")
            else:
                # Use the enhanced test method
                result = await video_task_manager.test_enhanced_pipeline("photosynthesis")
            
            return JSONResponse(content={
                "status": "success",
                "pipeline_test": result,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except AttributeError as e:
            return JSONResponse(content={
                "status": "method_error",
                "message": f"Test method not found: {str(e)}",
                "available_methods": [method for method in dir(video_task_manager) if not method.startswith('_')],
                "fix": "Check video task manager implementation"
            })
            
        except asyncio.TimeoutError:
            return JSONResponse(content={
                "status": "timeout_error",
                "message": "Pipeline test timed out",
                "fix": "Check system resources and try again"
            })
            
        except Exception as e:
            return JSONResponse(content={
                "status": "test_error",
                "message": f"Pipeline test failed: {str(e)}",
                "error_type": type(e).__name__,
                "fix": "Check video service configuration and dependencies"
            })
        
    except Exception as e:
        return JSONResponse(
            content={
                "status": "critical_error",
                "message": f"Critical pipeline test failure: {str(e)}",
                "error_type": type(e).__name__,
                "fix": "Restart the service and check all configurations",
                "install_commands": [
                    "sudo apt-get update && sudo apt-get install -y ffmpeg",
                    "pip install Pillow>=9.0.0",
                    "pip install azure-cognitiveservices-speech>=1.35.0"
                ]
            },
            status_code=500
        )

@app.post("/generate-video")
async def generate_video(request: VideoGenerationRequest, background_tasks: BackgroundTasks):
    """Generate educational video with improved error handling"""
    if not VIDEO_GENERATION_AVAILABLE or not video_task_manager:
        return JSONResponse(
            content={
                "error": "Video generation not available",
                "status": "service_unavailable",
                "install_commands": [
                    "sudo apt-get update && sudo apt-get install -y ffmpeg",
                    "pip install Pillow>=9.0.0",
                    "pip install azure-cognitiveservices-speech>=1.35.0"
                ],
                "fix": "Install dependencies and restart the service"
            },
            status_code=503
        )
    
    print(f"🎬 Video generation request: {request.topic}")
    
    try:
        # Retrieve context from RAG
        context = retrieve_context(request.topic)
        
        if not context:
            return JSONResponse(
                content={
                    "error": "No relevant educational content found for this topic",
                    "status": "no_content",
                    "suggestion": "Try a different topic or add more specific terms"
                },
                status_code=404
            )
        
        # Create job ID for immediate response
        job_id = str(uuid.uuid4())
        
        # Improved background task function
        def safe_background_video_generation():
            try:
                print(f"🎬 Starting background video generation for: {request.topic}")
                
                # Create new event loop for background task
                import asyncio
                import logging
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    # Run the video generation with timeout
                    actual_job_id = loop.run_until_complete(
                        asyncio.wait_for(
                            video_task_manager.generate_full_video(request.topic, context),
                            timeout=600  # 10 minute timeout
                        )
                    )
                    print(f"✅ Background video generation completed: {actual_job_id}")
                    
                    # Update the original job_id with the actual job_id
                    if video_service and video_service.video_jobs_collection:
                        try:
                            video_service.video_jobs_collection.update_one(
                                {"job_id": actual_job_id},
                                {"$set": {"original_request_id": job_id}}
                            )
                        except Exception as db_error:
                            print(f"Database update error: {db_error}")
                    
                    return actual_job_id
                    
                except asyncio.TimeoutError:
                    print(f"❌ Video generation timed out for: {request.topic}")
                    if video_service:
                        video_service.update_job_status(
                            job_id, "failed", 0,
                            error_message="Video generation timed out"
                        )
                    return None
                    
                except Exception as generation_error:
                    print(f"❌ Video generation error: {generation_error}")
                    logging.error(f"Video generation failed for topic '{request.topic}': {generation_error}")
                    if video_service:
                        video_service.update_job_status(
                            job_id, "failed", 0,
                            error_message=str(generation_error)
                        )
                    return None
                    
                finally:
                    try:
                        loop.close()
                    except Exception as loop_error:
                        print(f"Loop close error: {loop_error}")
                        
            except Exception as critical_error:
                print(f"❌ Critical background task error: {critical_error}")
                logging.error(f"Critical error in background video generation: {critical_error}")
                return None
        
        # Start video generation in background
        background_tasks.add_task(safe_background_video_generation)
        
        # Initialize job status in database
        if video_service:
            try:
                video_service.update_job_status(
                    job_id, "queued", 0,
                    topic=request.topic,
                    video_title=f"{request.topic} - Educational Video",
                    created_at=datetime.utcnow().isoformat()
                )
            except Exception as db_error:
                print(f"Database initialization error: {db_error}")
        
        # Save to database
        save_to_database("video_generation", request.topic, f"Job started with ID: {job_id}")
        
        return JSONResponse(content={
            "job_id": job_id,
            "status": "processing",
            "message": "Video generation started successfully",
            "topic": request.topic,
            "estimated_time": "3-5 minutes",
            "timestamp": datetime.utcnow().isoformat(),
            "pipeline": "Enhanced with FFmpeg-based video assembly",
            "note": "Check /video-status/{job_id} for progress updates"
        })
        
    except Exception as e:
        print(f"Video generation error: {e}")
        return JSONResponse(
            content={
                "error": f"Video generation failed: {str(e)}",
                "status": "generation_error",
                "error_type": type(e).__name__,
                "fix": "Check logs and try again"
            },
            status_code=500
        )


@app.get("/video-status/{job_id}")
async def get_video_status(job_id: str):
    """Get video generation job status"""
    if not VIDEO_GENERATION_AVAILABLE or not video_service:
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
            "available_features": dependencies,
            "pipeline": "FFmpeg-based"
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        print(f"Status check error: {e}")
        return JSONResponse(
            content={"error": "Failed to get job status"},
            status_code=500
        )

@app.post("/worksheet")
async def generate_worksheet(msg: str = Form(...), difficulty: str = Form("Medium")):
    """Generate educational worksheet"""
    print(f"Worksheet Topic: {msg}, Difficulty: {difficulty}")
    
    try:
        context = retrieve_context(msg)
        
        if not context:
            return JSONResponse(
                content={"error": "No relevant content found for this topic"},
                status_code=404
            )
        
        user_prompt = f"""
Topic: {msg}
Difficulty Level: {difficulty}
Context: {context}
"""
        
        response = generate_completion(system_prompt, user_prompt)
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
    """Generate video script"""
    print(f"Video Script Topic: {msg}")
    
    try:
        context = retrieve_context(msg)
        
        if not context:
            return JSONResponse(
                content={"error": "No relevant content found for this topic"},
                status_code=404
            )
        
        user_prompt = f"""
Topic: {msg}
Context: {context}
"""
        
        response = generate_completion(video_script_prompt, user_prompt)
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

@app.post("/qa")
async def generate_answer(data: QARequest):
    """Generate answer using RAG"""
    print(f"Q/A Question: {data.question}")
    
    try:
        context = retrieve_context(data.question)
        
        user_prompt = f"""
Question: {data.question}
Context: {context}
"""
        
        response = generate_completion(qa_prompt, user_prompt)
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

@app.get("/history")
async def chat_history(limit: int = 10):
    """Get chat history"""
    if not collection:
        return JSONResponse(content={
            "history": [],
            "count": 0,
            "message": "Database not configured"
        })
    
    try:
        history = list(collection.find().sort("timestamp", -1).limit(limit))
        
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

@app.delete("/history")
async def clear_history():
    """Clear chat history"""
    if not collection:
        return JSONResponse(content={
            "message": "Database not configured",
            "timestamp": datetime.utcnow().isoformat()
        })
    
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
    """Get statistics"""
    if not collection:
        return JSONResponse(content={
            "total_interactions": 0,
            "worksheets_generated": 0,
            "questions_answered": 0,
            "video_scripts_generated": 0,
            "message": "Database not configured",
            "timestamp": datetime.utcnow().isoformat()
        })
    
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

@app.get("/download-video/{job_id}")
async def download_video(job_id: str):
    """Download generated video file"""
    try:
        video_output_dir = "/workspaces/Sahayak_backend/videos"
        video_path = os.path.join(video_output_dir, f"final_video_{job_id}.mp4")
        
        if not os.path.exists(video_path):
            return JSONResponse(
                content={"error": "Video file not found"},
                status_code=404
            )
        
        file_size = os.path.getsize(video_path)
        if file_size == 0:
            return JSONResponse(
                content={"error": "Video file is empty"},
                status_code=404
            )
        
        from fastapi.responses import FileResponse
        return FileResponse(
            path=video_path,
            filename=f"video_{job_id}.mp4",
            media_type="video/mp4"
        )
        
    except Exception as e:
        return JSONResponse(
            content={"error": f"Download failed: {str(e)}"},
            status_code=500
        )

@app.get("/list-videos")
async def list_videos():
    """List all generated videos"""
    try:
        video_output_dir = "/workspaces/Sahayak_backend/videos"
        
        if not os.path.exists(video_output_dir):
            return JSONResponse(content={"videos": []})
        
        videos = []
        for filename in os.listdir(video_output_dir):
            if filename.endswith('.mp4'):
                filepath = os.path.join(video_output_dir, filename)
                file_size = os.path.getsize(filepath)
                videos.append({
                    "filename": filename,
                    "size_bytes": file_size,
                    "size_mb": round(file_size / (1024 * 1024), 2),
                    "download_url": f"/download-video/{filename.replace('final_video_', '').replace('.mp4', '')}"
                })
        
        return JSONResponse(content={
            "videos": videos,
            "total_count": len(videos)
        })
        
    except Exception as e:
        return JSONResponse(
            content={"error": f"Failed to list videos: {str(e)}"},
            status_code=500
        )       

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting Sahayak API on port {port}")
    print(f"📍 Access at: http://localhost:{port}")
    print(f"📋 Health check: http://localhost:{port}/health")
    print(f"📖 API docs: http://localhost:{port}/docs")
    
    try:
        uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        print("💡 Possible solutions:")
        print(f"   - Check if port {port} is already in use")
        print("   - Try a different port: PORT=8001 python app.py")
        print("   - Check your environment variables")
        