# src/video_generator.py - Fixed with better error handling

import os
import io
import base64
import asyncio
import uuid
import json
import tempfile
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

# Basic imports that should always work
import requests
from pymongo import MongoClient
import google.generativeai as genai
from groq import Groq
from dotenv import load_dotenv

# Optional imports with error handling
try:
    import moviepy.editor as mp
    MOVIEPY_AVAILABLE = True
    print("✅ MoviePy available")
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("⚠️ MoviePy not available - video assembly disabled")

try:
    from PIL import Image
    PIL_AVAILABLE = True
    print("✅ PIL available")
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL not available - image processing disabled")

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
    print("✅ Supabase library available")
except ImportError:
    SUPABASE_AVAILABLE = False
    print("⚠️ Supabase not available - using local storage")

try:
    import pyttsx3
    TTS_AVAILABLE = True
    print("✅ TTS available")
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️ TTS not available - audio generation disabled")

load_dotenv()

@dataclass
class VideoSegment:
    segment_number: int
    duration: float
    audio_script: str
    image_prompt: str
    educational_focus: str
    image_url: Optional[str] = None
    audio_url: Optional[str] = None
    image_path: Optional[str] = None
    audio_path: Optional[str] = None

@dataclass
class VideoJob:
    job_id: str
    topic: str
    status: str
    created_at: datetime
    video_title: str
    total_duration: int
    segments: List[VideoSegment]
    final_video_url: Optional[str] = None
    learning_objective: Optional[str] = None
    error_message: Optional[str] = None
    progress_percentage: int = 0

class VideoGenerationService:
    def __init__(self):
        print("🚀 Initializing Video Generation Service...")
        
        # Initialize API clients
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.mongo_uri = os.getenv("MONGO_URI")
        
        # Initialize Gemini
        if self.gemini_api_key:
            try:
                genai.configure(api_key=self.gemini_api_key)
                print("✅ Gemini configured")
            except Exception as e:
                print(f"⚠️ Gemini configuration failed: {e}")
        else:
            print("⚠️ GEMINI_API_KEY not found in environment")
        
        # Initialize Groq
        if self.groq_api_key:
            try:
                self.groq_client = Groq(api_key=self.groq_api_key)
                print("✅ Groq configured")
            except Exception as e:
                print(f"⚠️ Groq configuration failed: {e}")
                self.groq_client = None
        else:
            print("⚠️ GROQ_API_KEY not found in environment")
            self.groq_client = None
        
        # Initialize Supabase with better error handling
        self.supabase = None
        if SUPABASE_AVAILABLE and self.supabase_url and self.supabase_key:
            try:
                # Validate URL format
                if not self.supabase_url.startswith('https://'):
                    print(f"⚠️ Invalid Supabase URL format: {self.supabase_url}")
                    print("💡 URL should be like: https://your-project.supabase.co")
                elif len(self.supabase_key) < 100:  # Supabase keys are typically long
                    print(f"⚠️ Supabase key seems too short: {len(self.supabase_key)} characters")
                    print("💡 Check your SUPABASE_KEY in .env file")
                else:
                    self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
                    print("✅ Supabase configured")
            except Exception as e:
                print(f"⚠️ Supabase configuration failed: {e}")
                print("💡 Video generation will use local storage instead")
                self.supabase = None
        else:
            missing = []
            if not SUPABASE_AVAILABLE:
                missing.append("supabase library")
            if not self.supabase_url:
                missing.append("SUPABASE_URL")
            if not self.supabase_key:
                missing.append("SUPABASE_KEY")
            print(f"⚠️ Supabase not configured - missing: {', '.join(missing)}")
            print("💡 Add to .env: SUPABASE_URL=https://your-project.supabase.co")
            print("💡 Add to .env: SUPABASE_KEY=your-anon-key")
        
        # Initialize MongoDB
        if self.mongo_uri:
            try:
                self.mongo_client = MongoClient(self.mongo_uri)
                self.db = self.mongo_client[os.getenv("MONGO_DB_NAME")]
                self.video_jobs_collection = self.db["video_jobs"]
                # Test connection
                self.video_jobs_collection.count_documents({})
                print("✅ MongoDB configured")
            except Exception as e:
                print(f"⚠️ MongoDB configuration failed: {e}")
                self.video_jobs_collection = None
        else:
            print("⚠️ MONGO_URI not found in environment")
            self.video_jobs_collection = None
        
        # Initialize TTS if available
        if TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self._configure_tts()
                print("✅ TTS engine configured")
            except Exception as e:
                print(f"⚠️ TTS configuration failed: {e}")
                self.tts_engine = None
        else:
            self.tts_engine = None
        
        print("🎯 Video Generation Service initialization complete!")

    def _configure_tts(self):
        """Configure text-to-speech settings"""
        if not self.tts_engine:
            return
            
        try:
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Try to find a female voice
                for voice in voices:
                    if any(keyword in voice.name.lower() for keyword in ['female', 'zira', 'sophia', 'samantha']):
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
            self.tts_engine.setProperty('rate', 160)
            self.tts_engine.setProperty('volume', 0.9)
        except Exception as e:
            print(f"TTS configuration warning: {e}")

    def check_dependencies(self) -> Dict[str, bool]:
        """Check which dependencies are available"""
        return {
            "moviepy": MOVIEPY_AVAILABLE,
            "pillow": PIL_AVAILABLE,
            "supabase": SUPABASE_AVAILABLE and self.supabase is not None,
            "tts": TTS_AVAILABLE and self.tts_engine is not None,
            "gemini": self.gemini_api_key is not None,
            "groq": self.groq_client is not None,
            "mongodb": hasattr(self, 'video_jobs_collection') and self.video_jobs_collection is not None
        }

    def get_service_status(self) -> Dict[str, str]:
        """Get detailed status of all services"""
        status = {}
        
        # Gemini status
        if self.gemini_api_key:
            status["gemini"] = "✅ Configured"
        else:
            status["gemini"] = "❌ Missing GEMINI_API_KEY"
        
        # Groq status
        if self.groq_client:
            status["groq"] = "✅ Configured"
        else:
            status["groq"] = "❌ Missing GROQ_API_KEY"
        
        # Supabase status
        if self.supabase:
            status["supabase"] = "✅ Connected"
        elif not SUPABASE_AVAILABLE:
            status["supabase"] = "❌ Library not installed"
        elif not self.supabase_url:
            status["supabase"] = "❌ Missing SUPABASE_URL"
        elif not self.supabase_key:
            status["supabase"] = "❌ Missing SUPABASE_KEY"
        else:
            status["supabase"] = "❌ Configuration failed"
        
        # MongoDB status
        if hasattr(self, 'video_jobs_collection') and self.video_jobs_collection:
            status["mongodb"] = "✅ Connected"
        else:
            status["mongodb"] = "❌ Missing MONGO_URI or connection failed"
        
        # TTS status
        if self.tts_engine:
            status["tts"] = "✅ Ready"
        elif not TTS_AVAILABLE:
            status["tts"] = "❌ Install: pip install pyttsx3"
        else:
            status["tts"] = "❌ Initialization failed"
        
        # MoviePy status
        if MOVIEPY_AVAILABLE:
            status["moviepy"] = "✅ Ready"
        else:
            status["moviepy"] = "❌ Install: pip install moviepy"
        
        # PIL status
        if PIL_AVAILABLE:
            status["pillow"] = "✅ Ready"
        else:
            status["pillow"] = "❌ Install: pip install Pillow"
        
        return status

    def _create_video_script_prompt(self, context: str, topic: str) -> str:
        """Create enhanced prompt for video script generation"""
        return f"""You are an expert educational video creator. Create a detailed 30-second educational video script for the topic: "{topic}"

EDUCATIONAL CONTEXT:
{context}

Create exactly 5 segments of 6 seconds each. Return ONLY a valid JSON object:

{{
  "video_title": "{topic} - Educational Explanation",
  "total_duration": 30,
  "learning_objective": "Clear learning goal for students",
  "segments": [
    {{
      "segment_number": 1,
      "duration": 6,
      "audio_script": "Engaging hook (15-18 words maximum)",
      "image_prompt": "Detailed visual description for image generation",
      "educational_focus": "Introduction"
    }},
    {{
      "segment_number": 2,
      "duration": 6,
      "audio_script": "Build on concept (15-18 words maximum)",
      "image_prompt": "Visual showing process beginning",
      "educational_focus": "Concept Introduction"
    }},
    {{
      "segment_number": 3,
      "duration": 6,
      "audio_script": "Core explanation (15-18 words maximum)",
      "image_prompt": "Clear diagram of main process",
      "educational_focus": "Core Process"
    }},
    {{
      "segment_number": 4,
      "duration": 6,
      "audio_script": "Real-world application (15-18 words maximum)",
      "image_prompt": "Real-world example visual",
      "educational_focus": "Application"
    }},
    {{
      "segment_number": 5,
      "duration": 6,
      "audio_script": "Summary and conclusion (15-18 words maximum)",
      "image_prompt": "Memorable closing visual",
      "educational_focus": "Conclusion"
    }}
  ]
}}

Return ONLY the JSON object."""

    async def generate_video_script(self, topic: str, context: str) -> Optional[Dict]:
        """Generate structured video script using Gemini"""
        try:
            if not self.gemini_api_key:
                print("⚠️ Gemini not available, creating fallback script")
                return self._create_fallback_script(topic)
            
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            prompt = self._create_video_script_prompt(context, topic)
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=2000,
                )
            )
            
            # Extract JSON from response
            response_text = response.text.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            script_data = json.loads(response_text)
            print(f"✅ Generated script for: {script_data.get('video_title')}")
            return script_data
            
        except Exception as e:
            print(f"Script generation error: {e}")
            return self._create_fallback_script(topic)

    def _create_fallback_script(self, topic: str) -> Dict:
        """Create a basic fallback script when Gemini is not available"""
        return {
            "video_title": f"{topic} - Educational Video",
            "total_duration": 30,
            "learning_objective": f"Understand the basics of {topic}",
            "segments": [
                {
                    "segment_number": 1,
                    "duration": 6,
                    "audio_script": f"Welcome! Today we'll explore the fascinating world of {topic}.",
                    "image_prompt": f"Introduction slide with {topic} title and colorful background",
                    "educational_focus": "Introduction"
                },
                {
                    "segment_number": 2,
                    "duration": 6,
                    "audio_script": f"Let's start by understanding what {topic} really means.",
                    "image_prompt": f"Simple diagram showing basic concept of {topic}",
                    "educational_focus": "Definition"
                },
                {
                    "segment_number": 3,
                    "duration": 6,
                    "audio_script": f"The key process in {topic} involves several important steps.",
                    "image_prompt": f"Step-by-step visual breakdown of {topic} process",
                    "educational_focus": "Process"
                },
                {
                    "segment_number": 4,
                    "duration": 6,
                    "audio_script": f"We can see {topic} happening all around us every day.",
                    "image_prompt": f"Real-world examples of {topic} in nature or daily life",
                    "educational_focus": "Examples"
                },
                {
                    "segment_number": 5,
                    "duration": 6,
                    "audio_script": f"Remember, {topic} is essential for understanding our world better.",
                    "image_prompt": f"Summary visual with key points about {topic}",
                    "educational_focus": "Conclusion"
                }
            ]
        }

    async def generate_image(self, prompt: str, segment_number: int) -> Optional[str]:
        """Generate or create placeholder image"""
        if PIL_AVAILABLE:
            return await self._create_educational_placeholder(prompt, segment_number)
        else:
            # Return a text description if no image processing available
            return f"Image description: {prompt}"

    async def _create_educational_placeholder(self, prompt: str, segment_number: int) -> str:
        """Create educational placeholder image"""
        try:
            width, height = 800, 600
            colors = [(70, 130, 180), (34, 139, 34), (220, 20, 60), (255, 140, 0), (138, 43, 226)]
            bg_color = colors[segment_number % len(colors)]
            
            img = Image.new('RGB', (width, height), color=bg_color)
            
            # Save to temporary file
            temp_path = os.path.join(tempfile.gettempdir(), f"placeholder_{segment_number}_{uuid.uuid4().hex[:8]}.png")
            img.save(temp_path)
            
            # Upload to storage or return local path
            if self.supabase:
                try:
                    with open(temp_path, 'rb') as f:
                        image_data = f.read()
                    url = await self._upload_image_to_supabase(image_data, f"segment_{segment_number}")
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                    return url
                except Exception as e:
                    print(f"Upload failed, using local path: {e}")
                    return temp_path
            else:
                print(f"📁 Using local storage: {temp_path}")
                return temp_path
                
        except Exception as e:
            print(f"Image creation error: {e}")
            return None

    async def _upload_image_to_supabase(self, image_data: bytes, filename: str) -> str:
        """Upload image to Supabase storage"""
        try:
            if not self.supabase:
                return None
                
            file_path = f"video_images/{filename}_{uuid.uuid4().hex[:8]}.png"
            
            result = self.supabase.storage.from_("video-assets").upload(
                path=file_path,
                file=image_data,
                file_options={"content-type": "image/png"}
            )
            
            if result:
                url = self.supabase.storage.from_("video-assets").get_public_url(file_path)
                return url
            
            return None
            
        except Exception as e:
            print(f"Image upload error: {e}")
            return None

    def generate_audio(self, text: str, segment_number: int) -> Optional[str]:
        """Generate audio using available TTS"""
        if not self.tts_engine:
            print(f"⚠️ TTS not available, returning text for segment {segment_number}")
            return f"Audio script: {text}"
        
        try:
            temp_dir = tempfile.gettempdir()
            audio_filename = f"audio_segment_{segment_number}_{uuid.uuid4().hex[:8]}.wav"
            audio_path = os.path.join(temp_dir, audio_filename)
            
            self.tts_engine.save_to_file(text, audio_path)
            self.tts_engine.runAndWait()
            
            time.sleep(0.5)  # Wait for file creation
            
            if os.path.exists(audio_path):
                if self.supabase:
                    try:
                        with open(audio_path, 'rb') as audio_file:
                            audio_data = audio_file.read()
                        
                        audio_url = self._upload_audio_to_supabase(audio_data, f"segment_{segment_number}")
                        
                        try:
                            os.remove(audio_path)
                        except:
                            pass
                        
                        return audio_url
                    except Exception as e:
                        print(f"Audio upload failed, using local path: {e}")
                        return audio_path
                else:
                    print(f"📁 Using local audio storage: {audio_path}")
                    return audio_path
            
            return None
            
        except Exception as e:
            print(f"Audio generation error: {e}")
            return None

    def _upload_audio_to_supabase(self, audio_data: bytes, filename: str) -> str:
        """Upload audio to Supabase storage"""
        try:
            if not self.supabase:
                return None
                
            file_path = f"video_audio/{filename}_{uuid.uuid4().hex[:8]}.wav"
            
            result = self.supabase.storage.from_("video-assets").upload(
                path=file_path,
                file=audio_data,
                file_options={"content-type": "audio/wav"}
            )
            
            if result:
                url = self.supabase.storage.from_("video-assets").get_public_url(file_path)
                return url
            
            return None
            
        except Exception as e:
            print(f"Audio upload error: {e}")
            return None

    async def create_video_from_segments(self, segments: List[VideoSegment], job_id: str) -> Optional[str]:
        """Create video or return segment information"""
        if not MOVIEPY_AVAILABLE:
            print("⚠️ MoviePy not available - returning segment information instead of video")
            return self._create_video_info_json(segments, job_id)
        
        # MoviePy video creation would go here
        print("🎬 MoviePy available but video assembly not implemented yet")
        return self._create_video_info_json(segments, job_id)

    def _create_video_info_json(self, segments: List[VideoSegment], job_id: str) -> str:
        """Create JSON with video information when MoviePy is not available"""
        video_info = {
            "job_id": job_id,
            "status": "segments_ready",
            "message": "Video segments generated successfully",
            "segments": [
                {
                    "segment": seg.segment_number,
                    "duration": seg.duration,
                    "audio_script": seg.audio_script,
                    "image_prompt": seg.image_prompt,
                    "image_url": seg.image_url,
                    "audio_url": seg.audio_url
                }
                for seg in segments
            ],
            "next_steps": "Install MoviePy for video assembly: pip install moviepy>=1.0.3"
        }
        
        # Save as JSON file
        temp_path = os.path.join(tempfile.gettempdir(), f"video_info_{job_id}.json")
        with open(temp_path, 'w') as f:
            json.dump(video_info, f, indent=2)
        
        return temp_path

    def update_job_status(self, job_id: str, status: str, progress: int = 0, **kwargs):
        """Update job status in database"""
        try:
            if not self.video_jobs_collection:
                print(f"Status update: {job_id} - {status} ({progress}%)")
                return
                
            update_data = {
                "status": status,
                "progress_percentage": progress,
                "updated_at": datetime.utcnow()
            }
            update_data.update(kwargs)
            
            self.video_jobs_collection.update_one(
                {"job_id": job_id},
                {"$set": update_data}
            )
            print(f"Updated job {job_id}: {status} ({progress}%)")
            
        except Exception as e:
            print(f"Database update error: {e}")

    def save_video_job(self, job: VideoJob):
        """Save video job to database"""
        try:
            if not self.video_jobs_collection:
                print(f"Job saved locally: {job.job_id}")
                return
                
            job_data = {
                "job_id": job.job_id,
                "topic": job.topic,
                "status": job.status,
                "created_at": job.created_at,
                "video_title": job.video_title,
                "total_duration": job.total_duration,
                "learning_objective": job.learning_objective,
                "segments": [
                    {
                        "segment_number": seg.segment_number,
                        "duration": seg.duration,
                        "audio_script": seg.audio_script,
                        "image_prompt": seg.image_prompt,
                        "educational_focus": seg.educational_focus,
                        "image_url": seg.image_url,
                        "audio_url": seg.audio_url
                    }
                    for seg in job.segments
                ],
                "final_video_url": job.final_video_url,
                "error_message": job.error_message,
                "progress_percentage": job.progress_percentage
            }
            
            self.video_jobs_collection.insert_one(job_data)
            print(f"Saved job {job.job_id} to database")
            
        except Exception as e:
            print(f"Job save error: {e}")

    def get_video_job(self, job_id: str) -> Optional[Dict]:
        """Get video job from database"""
        try:
            if not self.video_jobs_collection:
                return {"job_id": job_id, "status": "no_database", "message": "Database not configured"}
                
            job_data = self.video_jobs_collection.find_one({"job_id": job_id})
            if job_data:
                job_data["_id"] = str(job_data["_id"])
                if "created_at" in job_data:
                    job_data["created_at"] = job_data["created_at"].isoformat()
                if "updated_at" in job_data:
                    job_data["updated_at"] = job_data["updated_at"].isoformat()
            return job_data
        except Exception as e:
            print(f"Job retrieval error: {e}")
            return None

# Create global instance with error handling
try:
    video_service = VideoGenerationService()
    print("✅ Video service created successfully")
except Exception as e:
    print(f"❌ Video service creation failed: {e}")
    video_service = None