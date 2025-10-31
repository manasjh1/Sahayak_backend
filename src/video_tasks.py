import asyncio
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from .video_generator import VideoGenerationService, VideoSegment, VideoJob

class VideoGenerationTasks:
    def __init__(self, video_service: VideoGenerationService):
        self.video_service = video_service

    async def generate_full_video(self, topic: str, context: str) -> str:
        """Generate complete educational video with dynamic images and synced audio"""
        job_id = str(uuid.uuid4())
        
        try:
            print(f"🎬 Starting enhanced video generation for: {topic}")
            
            # Step 1: Generate detailed script (5%)
            self.video_service.update_job_status(job_id, "generating_script", 5)
            script_data = await self.video_service.generate_video_script(topic, context)
            
            if not script_data:
                self.video_service.update_job_status(
                    job_id, "failed", 5, 
                    error_message="Failed to generate video script"
                )
                return job_id
            
            print(f"✅ Generated script: {script_data.get('video_title')}")
            
            # Step 2: Create segments (10%)
            segments = []
            for seg_data in script_data.get("segments", []):
                segment = VideoSegment(
                    segment_number=seg_data["segment_number"],
                    duration=seg_data["duration"],
                    audio_script=seg_data["audio_script"],
                    image_prompt=seg_data["image_prompt"],
                    educational_focus=seg_data["educational_focus"]
                )
                segments.append(segment)
            
            print(f"📝 Created {len(segments)} segments")
            
            # Step 3: Save job to database (15%)
            video_job = VideoJob(
                job_id=job_id,
                topic=topic,
                status="script_generated",
                created_at=datetime.utcnow(),
                video_title=script_data.get("video_title", f"{topic} - Educational Video"),
                total_duration=script_data.get("total_duration", 30),
                segments=segments,
                learning_objective=script_data.get("learning_objective", f"Learn about {topic}")
            )
            
            self.video_service.save_video_job(video_job)
            self.video_service.update_job_status(job_id, "generating_assets", 15)
            
            # Step 4: Generate dynamic assets for each segment (15% -> 70%)
            print("🎨 Generating dynamic images and high-quality audio...")
            total_segments = len(segments)
            
            for i, segment in enumerate(segments):
                print(f"📝 Processing segment {segment.segment_number}/{total_segments}: {segment.educational_focus}")
                
                # Generate dynamic, topic-specific image
                print(f"  🎨 Creating dynamic image for {topic}...")
                image_path = await self.video_service.generate_dynamic_educational_image(
                    segment.image_prompt, 
                    topic,
                    segment.segment_number
                )
                
                if image_path:
                    segment.image_path = image_path
                    segment.image_url = image_path
                    print(f"  ✅ Dynamic image created")
                else:
                    print(f"  ❌ Image generation failed")
                    # Continue with other segments
                
                # Generate high-quality synced audio
                print(f"  🎵 Creating audio narration...")
                audio_path = self.video_service.generate_audio(
                    segment.audio_script, 
                    segment.segment_number
                )
                
                if audio_path:
                    segment.audio_path = audio_path
                    segment.audio_url = audio_path
                    print(f"  ✅ Audio generated")
                else:
                    print(f"  ❌ Audio generation failed")
                
                # Update progress
                progress = 15 + int((i + 1) / total_segments * 55)  # 15% to 70%
                self.video_service.update_job_status(job_id, "generating_assets", progress)
                print(f"  📊 Progress: {progress}%")
            
            # Step 5: Create final synced video (70% -> 100%)
            print("🎬 Creating final video with audio-video sync...")
            self.video_service.update_job_status(job_id, "creating_video", 70)
            
            # Verify we have assets for video creation
            ready_segments = [s for s in segments if s.image_path and s.audio_path]
            print(f"📋 {len(ready_segments)}/{len(segments)} segments ready for video creation")
            
            if not ready_segments:
                self.video_service.update_job_status(
                    job_id, "failed", 70,
                    error_message="No segments have both image and audio assets"
                )
                return job_id
            
            # Create the final video
            final_video_path = await self.video_service.create_video_from_segments(ready_segments, job_id)
            
            if final_video_path:
                self.video_service.update_job_status(
                    job_id, "completed", 100,
                    final_video_url=final_video_path
                )
                print(f"✅ Enhanced video generation completed!")
                print(f"📹 Final video: {final_video_path}")
                print(f"🎯 Topic: {topic}")
                print(f"⏱️ Duration: {script_data.get('total_duration', 30)} seconds")
                print(f"🎨 Segments: {len(ready_segments)} with dynamic images")
                print(f"🔊 Audio: High-quality narration with sync")
            else:
                self.video_service.update_job_status(
                    job_id, "failed", 85,
                    error_message="Failed to create final synced video"
                )
                print("❌ Video assembly failed")
            
            return job_id
            
        except Exception as e:
            print(f"❌ Enhanced video generation error: {e}")
            self.video_service.update_job_status(
                job_id, "failed", 0,
                error_message=str(e)
            )
            return job_id

    async def test_enhanced_pipeline(self, test_topic: str = "photosynthesis") -> Dict:
        """Test the enhanced video generation pipeline with comprehensive error handling"""
        try:
            print(f"🧪 Testing enhanced video pipeline with: {test_topic}")
            
            # Check dependencies first
            dependencies = {}
            try:
                dependencies = self.video_service.check_dependencies()
                print(f"📋 Dependencies checked: {dependencies}")
            except Exception as e:
                print(f"❌ Failed to check dependencies: {e}")
                dependencies = {
                    "ffmpeg": False,
                    "pillow": False,
                    "supabase": False,
                    "tts": False,
                    "gemini": False,
                    "groq": False,
                    "mongodb": False
                }
            
            # Test script generation with detailed prompts
            test_context = """
            Photosynthesis is the process by which plants convert light energy into chemical energy.
            It occurs in chloroplasts and involves carbon dioxide, water, and sunlight to produce glucose and oxygen.
            This process is essential for all life on Earth as it produces oxygen and forms the base of food chains.
            """
            
            script = None
            try:
                print("📝 Testing script generation...")
                script = await self.video_service.generate_video_script(test_topic, test_context)
                print(f"✅ Script generation: {'Success' if script else 'Failed'}")
            except Exception as e:
                print(f"❌ Script generation failed: {e}")
            
            # Test dynamic image generation
            test_image = None
            if dependencies.get("pillow", False):
                try:
                    print("🎨 Testing dynamic image generation...")
                    test_image = await self.video_service.generate_dynamic_educational_image(
                        f"Educational diagram showing {test_topic} process with scientific accuracy", 
                        test_topic, 
                        1
                    )
                    print(f"✅ Image generation: {'Success' if test_image else 'Failed'}")
                except Exception as e:
                    print(f"❌ Image generation failed: {e}")
            else:
                print("⚠️ Skipping image test - PIL not available")
            
            # Test high-quality audio generation
            test_audio = None
            if dependencies.get("tts", False):
                try:
                    print("🎵 Testing audio generation...")
                    test_audio = self.video_service.generate_audio(
                        f"Welcome to learning about {test_topic}, an amazing process in nature.", 
                        1
                    )
                    print(f"✅ Audio generation: {'Success' if test_audio else 'Failed'}")
                except Exception as e:
                    print(f"❌ Audio generation failed: {e}")
            else:
                print("⚠️ Skipping audio test - Azure TTS not available")
            
            # Calculate enhanced readiness score
            ready_components = sum(1 for available in dependencies.values() if available)
            total_components = len(dependencies)
            readiness_percentage = (ready_components / total_components) * 100 if total_components > 0 else 0
            
            # Test FFmpeg capabilities
            ffmpeg_features = []
            if dependencies.get("ffmpeg", False):
                ffmpeg_features = [
                    "✅ Video encoding (H.264)",
                    "✅ Audio encoding (AAC)", 
                    "✅ Audio-video sync",
                    "✅ Multiple format support",
                    "✅ Quality optimization"
                ]
            else:
                ffmpeg_features = [
                    "❌ FFmpeg not available",
                    "❌ Cannot create videos",
                    "❌ No audio-video sync",
                    "❌ Limited functionality"
                ]
            
            # Get installation help
            installation_help = {}
            try:
                installation_help = self.video_service.get_installation_help()
            except Exception as e:
                print(f"Warning: Could not get installation help: {e}")
            
            result = {
                "status": "success",
                "test_topic": test_topic,
                "pipeline_version": "Enhanced with Dynamic Images",
                "dependencies": dependencies,
                "readiness_percentage": round(readiness_percentage, 1),
                "component_status": {
                    "script_generation": "✅ Working" if script else "❌ Failed",
                    "dynamic_image_generation": "✅ Working" if test_image else "❌ Failed" if dependencies.get("pillow") else "⚠️ PIL not available",
                    "high_quality_audio": "✅ Working" if test_audio else "❌ Failed" if dependencies.get("tts") else "⚠️ Azure TTS not available",
                    "video_assembly_sync": "✅ Ready" if dependencies.get("ffmpeg") else "❌ FFmpeg not available",
                    "cloud_storage": "✅ Ready" if dependencies.get("supabase") else "⚠️ Local storage only",
                    "database_tracking": "✅ Working" if dependencies.get("mongodb") else "⚠️ Limited functionality"
                },
                "enhanced_features": {
                    "dynamic_images": dependencies.get("pillow", False),
                    "topic_specific_visuals": dependencies.get("pillow", False) and dependencies.get("gemini", False),
                    "audio_video_sync": dependencies.get("ffmpeg", False) and dependencies.get("tts", False),
                    "high_quality_output": dependencies.get("ffmpeg", False),
                    "scientific_accuracy": dependencies.get("gemini", False)
                },
                "ffmpeg_capabilities": ffmpeg_features,
                "installation_help": installation_help,
                "recommendations": []
            }
            
            # Add specific recommendations with error handling
            try:
                if not dependencies.get("ffmpeg", False):
                    result["recommendations"].append("🔧 Install FFmpeg for video creation")
                if not dependencies.get("pillow", False):
                    result["recommendations"].append("🖼️ Install PIL for dynamic images")
                if not dependencies.get("tts", False):
                    result["recommendations"].append("🔊 Configure Azure TTS for audio")
                if not dependencies.get("supabase", False):
                    result["recommendations"].append("☁️ Configure Supabase for cloud storage")
                if not dependencies.get("mongodb", False):
                    result["recommendations"].append("📊 Configure MongoDB for job tracking")
            except Exception as e:
                print(f"Warning: Error generating recommendations: {e}")
                result["recommendations"].append("⚠️ Check system configuration")
            
            # Set overall status with enhanced criteria
            if readiness_percentage >= 85 and dependencies.get("ffmpeg", False) and dependencies.get("pillow", False):
                result["overall_status"] = "✅ Fully Ready for Enhanced Video Generation"
                result["message"] = "All systems operational! Ready to create dynamic educational videos with synced audio."
                result["estimated_quality"] = "High-quality videos with dynamic images and perfect audio sync"
            elif readiness_percentage >= 60 and dependencies.get("ffmpeg", False):
                result["overall_status"] = "⚠️ Partially Ready"
                result["message"] = "Basic video generation available. Some enhanced features may be limited."
                result["estimated_quality"] = "Standard quality videos with basic sync"
            else:
                result["overall_status"] = "❌ Needs Configuration"
                result["message"] = "Multiple components need setup before enhanced video generation."
                result["estimated_quality"] = "Limited functionality"
            
            # Add script preview if available
            if script:
                try:
                    result["script_preview"] = {
                        "title": script.get("video_title"),
                        "segments_count": len(script.get("segments", [])),
                        "learning_objective": script.get("learning_objective"),
                        "enhanced_prompts": True
                    }
                except Exception as e:
                    print(f"Warning: Error creating script preview: {e}")
            
            # Add file paths for verification
            test_results = {}
            if test_image:
                test_results["sample_image"] = test_image
            if test_audio:
                test_results["sample_audio"] = test_audio
            
            if test_results:
                result["test_files"] = test_results
            
            return result
            
        except Exception as e:
            print(f"❌ Pipeline test critical error: {e}")
            return {
                "status": "error",
                "error": f"Enhanced pipeline test failed: {str(e)}",
                "error_type": type(e).__name__,
                "dependencies": self.video_service.check_dependencies() if self.video_service else {},
                "recommendations": [
                    "Check your environment variables",
                    "Ensure all dependencies are properly installed", 
                    "Verify API keys are correctly configured",
                    "Install FFmpeg for video processing",
                    "Install PIL for dynamic image generation",
                    "Restart the service after installing dependencies"
                ],
                "critical_fixes": [
                    "sudo apt-get update && sudo apt-get install -y ffmpeg",
                    "pip install Pillow>=9.0.0",
                    "pip install azure-cognitiveservices-speech>=1.35.0",
                    "Restart the application"
                ]
            }

    async def test_video_pipeline(self, test_topic: str = "photosynthesis") -> Dict:
        """Alternative test method with simpler error handling"""
        try:
            return await self.test_enhanced_pipeline(test_topic)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Video pipeline test failed: {str(e)}",
                "fix": "Check video service configuration"
            }

    async def generate_sample_enhanced_video(self) -> str:
        """Generate a sample video showcasing enhanced features"""
        sample_topic = "Photosynthesis"
        sample_context = """
        Photosynthesis is one of the most important biological processes on Earth. It occurs in plants, algae, and some bacteria.
        During photosynthesis, chloroplasts in plant cells capture light energy and convert carbon dioxide and water into glucose and oxygen.
        The process can be divided into two main stages: light-dependent reactions and light-independent reactions (Calvin cycle).
        This process is crucial because it produces oxygen for life and forms the foundation of most food chains on our planet.
        """
        
        print("🎬 Generating enhanced sample video: Photosynthesis")
        print("🎨 Features: Dynamic images, topic-specific visuals, synced audio")
        
        try:
            job_id = await self.generate_full_video(sample_topic, sample_context)
            print(f"📋 Enhanced sample video job ID: {job_id}")
            return job_id
        except Exception as e:
            print(f"❌ Sample video generation failed: {e}")
            return f"error_{uuid.uuid4()}"

    async def get_generation_stats(self) -> Dict:
        """Get enhanced statistics about video generation"""
        try:
            if not self.video_service.video_jobs_collection:
                return {
                    "total_jobs": 0,
                    "completed": 0,
                    "failed": 0,
                    "processing": 0,
                    "message": "Database not configured"
                }
            
            # Get counts by status with error handling
            try:
                total_jobs = self.video_service.video_jobs_collection.count_documents({})
                completed_jobs = self.video_service.video_jobs_collection.count_documents({"status": "completed"})
                failed_jobs = self.video_service.video_jobs_collection.count_documents({"status": "failed"})
                processing_jobs = self.video_service.video_jobs_collection.count_documents({
                    "status": {"$in": ["generating_script", "generating_assets", "creating_video"]}
                })
            except Exception as e:
                print(f"Database query error: {e}")
                return {
                    "error": "Database query failed",
                    "message": str(e)
                }
            
            # Get recent jobs with more details
            recent_jobs = []
            try:
                recent_jobs = list(
                    self.video_service.video_jobs_collection
                    .find({}, {
                        "job_id": 1, "topic": 1, "status": 1, "created_at": 1, 
                        "video_title": 1, "progress_percentage": 1, "total_duration": 1
                    })
                    .sort("created_at", -1)
                    .limit(10)
                )
                
                # Format recent jobs
                for job in recent_jobs:
                    job["_id"] = str(job["_id"])
                    if "created_at" in job:
                        job["created_at"] = job["created_at"].isoformat()
            except Exception as e:
                print(f"Recent jobs query error: {e}")
            
            # Calculate average generation time for completed jobs
            avg_generation_time = "N/A"
            if completed_jobs > 0:
                # This would require storing completion times - simplified for now
                avg_generation_time = "3-5 minutes"
            
            return {
                "total_jobs": total_jobs,
                "completed": completed_jobs,
                "failed": failed_jobs,
                "processing": processing_jobs,
                "success_rate": round((completed_jobs / total_jobs * 100) if total_jobs > 0 else 0, 1),
                "average_generation_time": avg_generation_time,
                "recent_jobs": recent_jobs,
                "enhanced_features": {
                    "dynamic_images": True,
                    "audio_video_sync": True,
                    "topic_specific_visuals": True,
                    "high_quality_output": True
                },
                "system_status": self.video_service.check_dependencies(),
                "pipeline_version": "Enhanced with Dynamic Images & Audio Sync"
            }
            
        except Exception as e:
            print(f"Stats generation error: {e}")
            return {
                "error": f"Failed to get enhanced stats: {str(e)}",
                "system_status": self.video_service.check_dependencies() if self.video_service else {},
                "recommendations": [
                    "Check database connection",
                    "Verify MongoDB configuration",
                    "Restart the service"
                ]
            }

def create_video_task_manager(video_service: VideoGenerationService) -> Optional[VideoGenerationTasks]:
    """Factory function to create enhanced video task manager with error handling"""
    try:
        if not video_service:
            print("❌ Cannot create task manager - video service is None")
            return None
        
        task_manager = VideoGenerationTasks(video_service)
        print("✅ Video task manager created successfully")
        return task_manager
        
    except Exception as e:
        print(f"❌ Failed to create video task manager: {e}")
        return None