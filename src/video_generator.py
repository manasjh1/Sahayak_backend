import os
import asyncio
import uuid
import json
import tempfile
import subprocess
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass
import math

import requests
from pymongo import MongoClient
import google.generativeai as genai
from groq import Groq
from dotenv import load_dotenv

# Check FFmpeg availability
def check_ffmpeg():
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False

# Optional imports with error handling
try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    PIL_AVAILABLE = True
    print("✅ PIL available")
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL not available")

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
    print("✅ Supabase available")
except ImportError:
    SUPABASE_AVAILABLE = False
    print("⚠️ Supabase not available")

try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_TTS_AVAILABLE = True
    print("✅ Azure TTS available")
except ImportError:
    AZURE_TTS_AVAILABLE = False
    print("⚠️ Azure TTS not available")

FFMPEG_AVAILABLE = check_ffmpeg()
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
        print("🚀 Initializing Enhanced Video Service with Dynamic Images...")
        
        # API configuration
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.mongo_uri = os.getenv("MONGO_URI")
        
        # Initialize services
        self._init_gemini()
        self._init_supabase()
        self._init_mongodb()
        self._init_azure_tts()
        
        print("✅ Enhanced video service initialized")

    def _init_gemini(self):
        if self.gemini_api_key:
            try:
                genai.configure(api_key=self.gemini_api_key)
                print("✅ Gemini configured")
            except Exception as e:
                print(f"⚠️ Gemini error: {e}")

    def _init_supabase(self):
        self.supabase = None
        if SUPABASE_AVAILABLE and self.supabase_url and self.supabase_key:
            try:
                self.supabase = create_client(self.supabase_url, self.supabase_key)
                print("✅ Supabase configured")
            except Exception as e:
                print(f"⚠️ Supabase error: {e}")

    def _init_mongodb(self):
        self.video_jobs_collection = None
        if self.mongo_uri:
            try:
                self.mongo_client = MongoClient(self.mongo_uri)
                self.db = self.mongo_client[os.getenv("MONGO_DB_NAME", "sahayak")]
                self.video_jobs_collection = self.db["video_jobs"]
                self.video_jobs_collection.count_documents({})
                print("✅ MongoDB configured")
            except Exception as e:
                print(f"⚠️ MongoDB error: {e}")

    def _init_azure_tts(self):
        self.tts_engine = None
        if AZURE_TTS_AVAILABLE:
            speech_key = os.getenv("AZURE_SPEECH_KEY")
            service_region = os.getenv("AZURE_SPEECH_REGION")
            if speech_key and service_region:
                try:
                    self.tts_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
                    self.tts_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm)
                    self.tts_engine = "azure"
                    print("✅ Azure TTS configured")
                except Exception as e:
                    print(f"⚠️ Azure TTS error: {e}")

    def check_dependencies(self) -> Dict[str, bool]:
        """Check available dependencies"""
        return {
            "ffmpeg": FFMPEG_AVAILABLE,
            "pillow": PIL_AVAILABLE,
            "supabase": self.supabase is not None,
            "tts": self.tts_engine == "azure",
            "gemini": self.gemini_api_key is not None,
            "groq": True,
            "mongodb": self.video_jobs_collection is not None
        }

    async def generate_video_script(self, topic: str, context: str) -> Optional[Dict]:
        """Generate detailed video script with specific visual descriptions"""
        if not self.gemini_api_key:
            return self._create_fallback_script(topic)
        
        try:
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            prompt = f"""Create a detailed 30-second educational video script for "{topic}".

Educational Context: {context}

Create exactly 5 segments of 6 seconds each. For each segment, provide:
1. Educational narration (exactly 15 words)
2. Detailed visual description for dynamic image generation
3. Specific visual elements that relate to the topic

Return ONLY valid JSON:
{{
  "video_title": "{topic} - Educational Explanation",
  "total_duration": 30,
  "learning_objective": "Students will understand {topic} and its key concepts",
  "segments": [
    {{
      "segment_number": 1,
      "duration": 6,
      "audio_script": "Welcome to learning about {topic}, an amazing scientific process we will explore today.",
      "image_prompt": "Create a vibrant educational illustration showing {topic} with clear labels, bright colors, scientific accuracy, and engaging visual elements that immediately capture what {topic} is about",
      "educational_focus": "Introduction"
    }},
    {{
      "segment_number": 2,
      "duration": 6,
      "audio_script": "The key components of {topic} work together in a fascinating and essential process.",
      "image_prompt": "Detailed diagram showing the main components and elements involved in {topic}, with arrows indicating flow or process, labeled parts, and scientific accuracy",
      "educational_focus": "Components"
    }},
    {{
      "segment_number": 3,
      "duration": 6,
      "audio_script": "Here is exactly how the {topic} process works step by step in nature.",
      "image_prompt": "Step-by-step visual breakdown of the {topic} process, showing each stage clearly with numbers, arrows, and detailed scientific illustrations",
      "educational_focus": "Process"
    }},
    {{
      "segment_number": 4,
      "duration": 6,
      "audio_script": "We can observe {topic} happening all around us in these real world examples.",
      "image_prompt": "Real-world examples of {topic} in nature or daily life, showing practical applications and where students can observe this concept",
      "educational_focus": "Examples"
    }},
    {{
      "segment_number": 5,
      "duration": 6,
      "audio_script": "Remember that {topic} is crucial for life and understanding our natural world completely.",
      "image_prompt": "Summary illustration showing the importance of {topic} with key points, benefits, and why it matters for life on Earth",
      "educational_focus": "Conclusion"
    }}
  ]
}}

Make the image prompts very specific to {topic} and scientifically accurate."""
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=3000,
                )
            )
            
            response_text = response.text.strip()
            
            # Clean JSON extraction
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            script_data = json.loads(response_text)
            print(f"✅ Generated detailed script: {script_data.get('video_title')}")
            return script_data
            
        except Exception as e:
            print(f"Script generation error: {e}")
            return self._create_fallback_script(topic)

    def _create_fallback_script(self, topic: str) -> Dict:
        """Create topic-specific fallback script"""
        return {
            "video_title": f"{topic} - Educational Video",
            "total_duration": 30,
            "learning_objective": f"Understand the key concepts of {topic}",
            "segments": [
                {
                    "segment_number": 1,
                    "duration": 6,
                    "audio_script": f"Welcome! Today we will explore the fascinating world of {topic} and its importance.",
                    "image_prompt": f"Educational title card for {topic} with vibrant colors, scientific symbols, and clear topic name displayed prominently",
                    "educational_focus": "Introduction"
                },
                {
                    "segment_number": 2,
                    "duration": 6,
                    "audio_script": f"Let's understand what {topic} really means and why it matters in science.",
                    "image_prompt": f"Detailed scientific diagram explaining the basic concept of {topic} with labels, arrows, and clear visual elements",
                    "educational_focus": "Definition"
                },
                {
                    "segment_number": 3,
                    "duration": 6,
                    "audio_script": f"Here's how the {topic} process works through these important scientific steps.",
                    "image_prompt": f"Step-by-step process diagram of {topic} showing each stage with numbered steps and scientific accuracy",
                    "educational_focus": "Process"
                },
                {
                    "segment_number": 4,
                    "duration": 6,
                    "audio_script": f"We can see {topic} happening in nature and everyday life around us.",
                    "image_prompt": f"Real-world examples of {topic} in nature showing practical applications and observable instances",
                    "educational_focus": "Examples"
                },
                {
                    "segment_number": 5,
                    "duration": 6,
                    "audio_script": f"Remember, {topic} is essential for understanding our world and scientific processes.",
                    "image_prompt": f"Summary graphic showing the importance and benefits of {topic} with key takeaway points",
                    "educational_focus": "Conclusion"
                }
            ]
        }

    async def generate_dynamic_educational_image(self, prompt: str, topic: str, segment_number: int) -> Optional[str]:
        """Generate dynamic, topic-specific educational images"""
        if not PIL_AVAILABLE:
            return f"Image: {prompt}"
        
        try:
            # Create high-quality educational image
            width, height = 1920, 1080  # Full HD
            
            # Dynamic color schemes based on topic
            topic_colors = self._get_topic_color_scheme(topic, segment_number)
            
            # Create base image with gradient
            img = Image.new('RGB', (width, height), color=topic_colors["primary"])
            draw = ImageDraw.Draw(img)
            
            # Add sophisticated gradient background
            self._add_educational_gradient(img, draw, topic_colors)
            
            # Add topic-specific visual elements
            self._add_topic_visuals(img, draw, topic, segment_number, topic_colors)
            
            # Add educational text and labels
            self._add_educational_text(img, draw, topic, segment_number, topic_colors)
            
            # Add scientific elements and details
            self._add_scientific_elements(img, draw, topic, segment_number, topic_colors)
            
            # Save high-quality image
            temp_path = os.path.join(tempfile.gettempdir(), 
                                   f"dynamic_{topic}_{segment_number}_{uuid.uuid4().hex[:8]}.png")
            img.save(temp_path, "PNG", quality=100, optimize=True)
            
            print(f"✅ Generated dynamic image for {topic} segment {segment_number}")
            return temp_path
            
        except Exception as e:
            print(f"Dynamic image creation error: {e}")
            return None

    def _get_topic_color_scheme(self, topic: str, segment_number: int) -> Dict:
        """Get dynamic color scheme based on topic"""
        topic_lower = topic.lower()
        
        # Science topic color schemes
        if any(word in topic_lower for word in ['photosynthesis', 'plant', 'chlorophyll']):
            schemes = [
                {"primary": (34, 139, 34), "secondary": (144, 238, 144), "accent": (255, 255, 255), "text": (255, 255, 255)},
                {"primary": (0, 100, 0), "secondary": (50, 205, 50), "accent": (255, 255, 0), "text": (255, 255, 255)},
                {"primary": (46, 125, 50), "secondary": (129, 199, 132), "accent": (255, 235, 59), "text": (255, 255, 255)},
                {"primary": (27, 94, 32), "secondary": (102, 187, 106), "accent": (255, 193, 7), "text": (255, 255, 255)},
                {"primary": (56, 142, 60), "secondary": (165, 214, 167), "accent": (255, 152, 0), "text": (255, 255, 255)}
            ]
        elif any(word in topic_lower for word in ['water', 'ocean', 'cycle', 'evaporation']):
            schemes = [
                {"primary": (25, 118, 210), "secondary": (144, 202, 249), "accent": (255, 255, 255), "text": (255, 255, 255)},
                {"primary": (13, 71, 161), "secondary": (100, 181, 246), "accent": (129, 212, 250), "text": (255, 255, 255)},
                {"primary": (21, 101, 192), "secondary": (121, 185, 255), "accent": (179, 229, 252), "text": (255, 255, 255)},
                {"primary": (30, 136, 229), "secondary": (144, 202, 249), "accent": (187, 222, 251), "text": (255, 255, 255)},
                {"primary": (33, 150, 243), "secondary": (166, 212, 250), "accent": (225, 245, 254), "text": (255, 255, 255)}
            ]
        elif any(word in topic_lower for word in ['space', 'solar', 'planet', 'astronomy']):
            schemes = [
                {"primary": (26, 35, 126), "secondary": (92, 107, 192), "accent": (255, 193, 7), "text": (255, 255, 255)},
                {"primary": (40, 53, 147), "secondary": (121, 134, 203), "accent": (255, 235, 59), "text": (255, 255, 255)},
                {"primary": (48, 63, 159), "secondary": (159, 168, 218), "accent": (255, 213, 79), "text": (255, 255, 255)},
                {"primary": (57, 73, 171), "secondary": (197, 202, 233), "accent": (255, 183, 77), "text": (255, 255, 255)},
                {"primary": (63, 81, 181), "secondary": (159, 168, 218), "accent": (255, 202, 40), "text": (255, 255, 255)}
            ]
        else:  # Default science colors
            schemes = [
                {"primary": (103, 58, 183), "secondary": (179, 157, 219), "accent": (255, 193, 7), "text": (255, 255, 255)},
                {"primary": (156, 39, 176), "secondary": (206, 147, 216), "accent": (255, 235, 59), "text": (255, 255, 255)},
                {"primary": (233, 30, 99), "secondary": (248, 187, 208), "accent": (255, 152, 0), "text": (255, 255, 255)},
                {"primary": (255, 87, 34), "secondary": (255, 204, 188), "accent": (76, 175, 80), "text": (255, 255, 255)},
                {"primary": (96, 125, 139), "secondary": (176, 190, 197), "accent": (255, 193, 7), "text": (255, 255, 255)}
            ]
        
        return schemes[(segment_number - 1) % len(schemes)]

    def _add_educational_gradient(self, img: Image.Image, draw: ImageDraw.Draw, colors: Dict):
        """Add sophisticated gradient background"""
        width, height = img.size
        
        # Create vertical gradient
        for y in range(height):
            alpha = y / height
            r = int(colors["primary"][0] * (1 - alpha) + colors["secondary"][0] * alpha)
            g = int(colors["primary"][1] * (1 - alpha) + colors["secondary"][1] * alpha)
            b = int(colors["primary"][2] * (1 - alpha) + colors["secondary"][2] * alpha)
            draw.line([(0, y), (width, y)], fill=(r, g, b))

    def _add_topic_visuals(self, img: Image.Image, draw: ImageDraw.Draw, topic: str, segment_number: int, colors: Dict):
        """Add topic-specific visual elements"""
        width, height = img.size
        accent_color = colors["accent"]
        
        topic_lower = topic.lower()
        
        if 'photosynthesis' in topic_lower or 'plant' in topic_lower:
            if segment_number == 1:  # Introduction - sun and plant
                # Draw sun
                sun_x, sun_y = width // 4, height // 3
                for radius in range(60, 80, 4):
                    draw.ellipse([sun_x - radius, sun_y - radius, sun_x + radius, sun_y + radius], 
                               outline=(255, 255, 0), width=3)
                # Sun rays
                for angle in range(0, 360, 45):
                    end_x = sun_x + 100 * math.cos(math.radians(angle))
                    end_y = sun_y + 100 * math.sin(math.radians(angle))
                    draw.line([sun_x, sun_y, end_x, end_y], fill=(255, 255, 0), width=4)
                
                # Draw plant
                plant_x = 3 * width // 4
                # Stem
                draw.line([plant_x, height - 100, plant_x, height - 300], fill=(0, 128, 0), width=8)
                # Leaves
                for i, y_offset in enumerate([250, 200, 150]):
                    leaf_y = height - y_offset
                    draw.ellipse([plant_x - 40, leaf_y - 20, plant_x + 40, leaf_y + 20], 
                               fill=(0, 150, 0), outline=(0, 100, 0), width=2)
            
            elif segment_number == 2:  # Components - CO2, H2O, light
                # CO2 molecules
                for i in range(3):
                    x = width // 4 + i * 100
                    y = height // 3
                    draw.text((x, y), "CO₂", fill=accent_color, font=self._get_font(48))
                
                # H2O molecules
                for i in range(3):
                    x = width // 4 + i * 100
                    y = 2 * height // 3
                    draw.text((x, y), "H₂O", fill=accent_color, font=self._get_font(48))
        
        elif 'water' in topic_lower or 'cycle' in topic_lower:
            if segment_number == 1:  # Water cycle overview
                # Draw clouds
                for i in range(3):
                    cloud_x = width // 4 + i * 200
                    cloud_y = height // 4
                    self._draw_cloud(draw, cloud_x, cloud_y, accent_color)
                
                # Draw ocean
                draw.rectangle([0, 3 * height // 4, width, height], fill=(0, 119, 190))
                
                # Draw evaporation arrows
                for i in range(5):
                    x = width // 6 + i * 150
                    start_y = 3 * height // 4
                    end_y = height // 2
                    self._draw_arrow(draw, x, start_y, x, end_y, accent_color)

    def _add_educational_text(self, img: Image.Image, draw: ImageDraw.Draw, topic: str, segment_number: int, colors: Dict):
        """Add educational text and labels"""
        width, height = img.size
        text_color = colors["text"]
        
        # Main title
        segment_titles = {
            1: "Introduction",
            2: "Key Components", 
            3: "The Process",
            4: "Real Examples",
            5: "Why It Matters"
        }
        
        title = f"{topic}: {segment_titles.get(segment_number, 'Learning')}"
        
        try:
            title_font = self._get_font(72)
            subtitle_font = self._get_font(48)
        except:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
        
        # Add title with shadow
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (width - title_width) // 2
        
        # Shadow
        draw.text((title_x + 3, 53), title, font=title_font, fill=(0, 0, 0))
        # Main text
        draw.text((title_x, 50), title, font=title_font, fill=text_color)
        
        # Segment number
        segment_text = f"Part {segment_number} of 5"
        segment_bbox = draw.textbbox((0, 0), segment_text, font=subtitle_font)
        segment_width = segment_bbox[2] - segment_bbox[0]
        segment_x = (width - segment_width) // 2
        
        draw.text((segment_x + 2, height - 82), segment_text, font=subtitle_font, fill=(0, 0, 0))
        draw.text((segment_x, height - 80), segment_text, font=subtitle_font, fill=text_color)

    def _add_scientific_elements(self, img: Image.Image, draw: ImageDraw.Draw, topic: str, segment_number: int, colors: Dict):
        """Add scientific diagrams and elements"""
        width, height = img.size
        accent_color = colors["accent"]
        
        # Add scientific symbols or equations based on topic
        topic_lower = topic.lower()
        
        if 'photosynthesis' in topic_lower and segment_number == 3:
            # Add the photosynthesis equation
            equation = "6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂"
            eq_font = self._get_font(36)
            
            eq_bbox = draw.textbbox((0, 0), equation, font=eq_font)
            eq_width = eq_bbox[2] - eq_bbox[0]
            eq_x = (width - eq_width) // 2
            eq_y = height // 2
            
            # Background box for equation
            draw.rectangle([eq_x - 20, eq_y - 10, eq_x + eq_width + 20, eq_y + 50], 
                         fill=(0, 0, 0, 128), outline=accent_color, width=3)
            draw.text((eq_x, eq_y), equation, font=eq_font, fill=accent_color)

    def _draw_cloud(self, draw: ImageDraw.Draw, x: int, y: int, color):
        """Draw a cloud shape"""
        # Multiple circles to form cloud
        circles = [(x, y, 40), (x+30, y-10, 35), (x-30, y-10, 35), (x+15, y+15, 30), (x-15, y+15, 30)]
        for cx, cy, radius in circles:
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color)

    def _draw_arrow(self, draw: ImageDraw.Draw, x1: int, y1: int, x2: int, y2: int, color):
        """Draw an arrow"""
        # Arrow line
        draw.line([x1, y1, x2, y2], fill=color, width=4)
        # Arrow head
        if y2 < y1:  # Upward arrow
            draw.polygon([(x2, y2), (x2-10, y2+20), (x2+10, y2+20)], fill=color)

    def _get_font(self, size: int):
        """Get font with fallback"""
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        except:
            return ImageFont.load_default()

    async def generate_image(self, prompt: str, segment_number: int) -> Optional[str]:
        """Generate image - wrapper for backward compatibility"""
        # Extract topic from prompt (simple approach)
        topic = "science"  # Default
        if "photosynthesis" in prompt.lower():
            topic = "photosynthesis"
        elif "water" in prompt.lower():
            topic = "water cycle"
        
        return await self.generate_dynamic_educational_image(prompt, topic, segment_number)

    def generate_audio(self, text: str, segment_number: int) -> Optional[str]:
        """Generate high-quality audio with proper duration"""
        if self.tts_engine != "azure":
            print(f"⚠️ Azure TTS not available for segment {segment_number}")
            return None
        
        try:
            temp_path = os.path.join(tempfile.gettempdir(), 
                                   f"audio_{segment_number}_{uuid.uuid4().hex[:8]}.wav")
            
            # Configure audio for better quality
            audio_config = speechsdk.audio.AudioOutputConfig(filename=temp_path)
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self.tts_config, 
                audio_config=audio_config
            )
            
            # Use SSML for better control
            ssml_text = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
                <voice name="en-US-AriaNeural">
                    <prosody rate="medium" pitch="medium" volume="loud">
                        {text}
                    </prosody>
                </voice>
            </speak>
            """
            
            result = synthesizer.speak_ssml_async(ssml_text).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                print(f"✅ High-quality audio generated for segment {segment_number}")
                return temp_path
            else:
                print(f"❌ TTS failed for segment {segment_number}: {result.reason}")
                return None
                
        except Exception as e:
            print(f"Audio generation error: {e}")
            return None

    async def create_video_from_segments(self, segments: List[VideoSegment], job_id: str) -> Optional[str]:
        """Create synced video with audio using FFmpeg"""
        if not FFMPEG_AVAILABLE:
            print("❌ FFmpeg not available for video creation")
            return None
        
        try:
            temp_dir = tempfile.gettempdir()
            segment_videos = []
            
            print(f"🎬 Creating synced video from {len(segments)} segments...")
            
            # Create individual video segments with proper sync
            for segment in segments:
                if not segment.image_path or not segment.audio_path:
                    print(f"⚠️ Segment {segment.segment_number} missing assets")
                    continue
                
                segment_video = os.path.join(temp_dir, f"segment_{segment.segment_number}_{job_id}.mp4")
                
                # Get audio duration to ensure proper sync
                audio_duration = self._get_audio_duration(segment.audio_path)
                if audio_duration is None:
                    audio_duration = segment.duration
                
                # FFmpeg command with proper audio-video sync
                cmd = [
                    'ffmpeg', '-y',  # Overwrite output files
                    '-loop', '1',    # Loop the image
                    '-i', segment.image_path,  # Input image
                    '-i', segment.audio_path,  # Input audio
                    '-c:v', 'libx264',  # Video codec
                    '-c:a', 'aac',      # Audio codec
                    '-b:a', '128k',     # Audio bitrate
                    '-ar', '44100',     # Audio sample rate
                    '-t', str(audio_duration),  # Duration matches audio
                    '-pix_fmt', 'yuv420p',  # Pixel format
                    '-vf', 'scale=1920:1080',  # Scale to 1080p
                    '-r', '30',  # Frame rate
                    '-shortest',  # Stop when shortest input ends
                    segment_video
                ]
                
                print(f"  Creating segment {segment.segment_number} with {audio_duration}s duration...")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    segment_videos.append(segment_video)
                    print(f"  ✅ Segment {segment.segment_number} created successfully")
                else:
                    print(f"  ❌ Segment {segment.segment_number} failed: {result.stderr[:100]}")
            
            if not segment_videos:
                print("❌ No segment videos created successfully")
                return None
            
            # Create concat file for FFmpeg
            concat_file = os.path.join(temp_dir, f"concat_{job_id}.txt")
            with open(concat_file, 'w') as f:
                for video in segment_videos:
                    f.write(f"file '{video}'\n")
            
            # Combine all segments into final video with proper encoding
            final_video = os.path.join(temp_dir, f"final_video_{job_id}.mp4")
            concat_cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c:v', 'libx264',  # Re-encode for consistency
                '-c:a', 'aac',      # Re-encode audio
                '-b:v', '2M',       # Video bitrate
                '-b:a', '128k',     # Audio bitrate
                '-r', '30',         # Frame rate
                '-movflags', '+faststart',  # Optimize for web
                final_video
            ]
            
            print("🎬 Combining segments into final synced video...")
            result = subprocess.run(concat_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Final synced video created: {final_video}")
                
                # Verify the final video has audio
                if self._verify_video_has_audio(final_video):
                    print("✅ Video has audio track")
                else:
                    print("⚠️ Video may not have audio track")
                
                # Clean up temporary segment videos
                for video in segment_videos:
                    try:
                        os.remove(video)
                    except:
                        pass
                try:
                    os.remove(concat_file)
                except:
                    pass
                
                return final_video
            else:
                print(f"❌ Video combination failed: {result.stderr[:100]}")
                return None
                
        except Exception as e:
            print(f"Video creation error: {e}")
            return None

    def _get_audio_duration(self, audio_path: str) -> Optional[float]:
        """Get audio file duration using FFprobe"""
        try:
            cmd = [
                'ffprobe', 
                '-v', 'quiet', 
                '-show_entries', 'format=duration', 
                '-of', 'default=noprint_wrappers=1:nokey=1', 
                audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                duration = float(result.stdout.strip())
                print(f"📏 Audio duration: {duration:.2f}s")
                return duration
        except Exception as e:
            print(f"Error getting audio duration: {e}")
        return None

    def _verify_video_has_audio(self, video_path: str) -> bool:
        """Verify video file has audio track"""
        try:
            cmd = [
                'ffprobe', 
                '-v', 'quiet', 
                '-select_streams', 'a:0', 
                '-show_entries', 'stream=codec_type', 
                '-of', 'csv=p=0', 
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0 and 'audio' in result.stdout
        except:
            return False

    def update_job_status(self, job_id: str, status: str, progress: int = 0, **kwargs):
        """Update job status in database"""
        if self.video_jobs_collection:
            try:
                update_data = {
                    "status": status,
                    "progress_percentage": progress,
                    "updated_at": datetime.utcnow(),
                    **kwargs
                }
                self.video_jobs_collection.update_one(
                    {"job_id": job_id},
                    {"$set": update_data},
                    upsert=True
                )
                print(f"📊 Job {job_id}: {status} ({progress}%)")
            except Exception as e:
                print(f"Database update error: {e}")
        else:
            print(f"📊 Job {job_id}: {status} ({progress}%) [No DB]")

    def save_video_job(self, job: VideoJob):
        """Save video job to database"""
        if self.video_jobs_collection:
            try:
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
                            "segment_number": s.segment_number,
                            "duration": s.duration,
                            "audio_script": s.audio_script,
                            "image_prompt": s.image_prompt,
                            "educational_focus": s.educational_focus,
                            "image_url": s.image_url,
                            "audio_url": s.audio_url
                        }
                        for s in job.segments
                    ],
                    "final_video_url": job.final_video_url,
                    "error_message": job.error_message,
                    "progress_percentage": job.progress_percentage
                }
                self.video_jobs_collection.insert_one(job_data)
                print(f"💾 Saved job {job.job_id}")
            except Exception as e:
                print(f"Job save error: {e}")

    def get_video_job(self, job_id: str) -> Optional[Dict]:
        """Get video job from database"""
        if self.video_jobs_collection:
            try:
                job = self.video_jobs_collection.find_one({"job_id": job_id})
                if job:
                    job["_id"] = str(job["_id"])
                    if "created_at" in job:
                        job["created_at"] = job["created_at"].isoformat()
                    if "updated_at" in job:
                        job["updated_at"] = job["updated_at"].isoformat()
                return job
            except Exception as e:
                print(f"Job retrieval error: {e}")
        return {"job_id": job_id, "status": "not_found", "message": "Database not available"}

# Create service instance
try:
    video_service = VideoGenerationService()
    print("✅ Enhanced video service ready with dynamic images and audio sync")
except Exception as e:
    print(f"❌ Enhanced video service creation failed: {e}")
    video_service = None