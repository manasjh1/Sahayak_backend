# src/video_tasks.py

import asyncio
from typing import Dict, List
from datetime import datetime
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from .video_generator import VideoGenerationService, VideoSegment, VideoJob

class VideoGenerationTasks:
    def __init__(self, video_service: VideoGenerationService):
        self.video_service = video_service

    async def generate_full_video(self, topic: str, context: str) -> str:
        """
        Main async function to generate complete video
        Returns job_id for tracking
        """
        # Create unique job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job
        try:
            self.video_service.update_job_status(job_id, "initializing", 0)
            
            # Step 1: Generate video script (10%)
            print(f"🎬 Generating script for: {topic}")
            script_data = await self.video_service.generate_video_script(topic, context)
            
            if not script_data:
                self.video_service.update_job_status(
                    job_id, "failed", 0, 
                    error_message="Failed to generate video script"
                )
                return job_id
            
            # Create segments from script
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
            
            # Create and save job
            video_job = VideoJob(
                job_id=job_id,
                topic=topic,
                status="script_generated",
                created_at=datetime.utcnow(),
                video_title=script_data.get("video_title", f"{topic} - Educational Video"),
                total_duration=script_data.get("total_duration", 30),
                segments=segments,
                learning_objective=script_data.get("learning_objective", "")
            )
            
            self.video_service.save_video_job(video_job)
            self.video_service.update_job_status(job_id, "generating_assets", 10)
            
            # Step 2: Generate images and audio in parallel (10% -> 80%)
            await self._generate_all_assets(job_id, segments)
            
            # Step 3: Create final video (80% -> 100%)
            self.video_service.update_job_status(job_id, "creating_video", 80)
            
            final_video_url = await self.video_service.create_video_from_segments(segments, job_id)
            
            if final_video_url:
                self.video_service.update_job_status(
                    job_id, "completed", 100,
                    final_video_url=final_video_url
                )
                print(f"✅ Video generation completed: {final_video_url}")
            else:
                self.video_service.update_job_status(
                    job_id, "failed", 80,
                    error_message="Failed to create final video"
                )
            
            return job_id
            
        except Exception as e:
            print(f"❌ Video generation error: {e}")
            self.video_service.update_job_status(
                job_id, "failed", 0,
                error_message=str(e)
            )
            return job_id

    async def _generate_all_assets(self, job_id: str, segments: List[VideoSegment]):
        """Generate all images and audio files in parallel"""
        try:
            total_segments = len(segments)
            completed_segments = 0
            
            # Create tasks for parallel execution
            tasks = []
            
            # Generate images and audio for each segment
            for segment in segments:
                # Add image generation task
                image_task = self._generate_segment_image(segment)
                tasks.append(('image', segment.segment_number, image_task))
                
                # Add audio generation task (run in thread since it's blocking)
                audio_task = self._generate_segment_audio_async(segment)
                tasks.append(('audio', segment.segment_number, audio_task))
            
            # Execute all tasks and update progress
            for task_type, segment_num, task in tasks:
                try:
                    result = await task
                    
                    # Update the segment with the generated asset
                    for segment in segments:
                        if segment.segment_number == segment_num:
                            if task_type == 'image':
                                segment.image_url = result
                                print(f"✅ Image generated for segment {segment_num}")
                            elif task_type == 'audio':
                                segment.audio_url = result
                                print(f"✅ Audio generated for segment {segment_num}")
                            break
                    
                    completed_segments += 1
                    progress = 10 + int((completed_segments / (total_segments * 2)) * 70)  # 10% to 80%
                    self.video_service.update_job_status(job_id, "generating_assets", progress)
                    
                except Exception as e:
                    print(f"❌ Asset generation error for segment {segment_num} ({task_type}): {e}")
                    # Continue with other assets even if one fails
            
            print(f"🎯 Asset generation completed. Generated assets for {len(segments)} segments")
            
        except Exception as e:
            print(f"❌ Asset generation batch error: {e}")
            raise e

    async def _generate_segment_image(self, segment: VideoSegment) -> str:
        """Generate image for a single segment"""
        return await self.video_service.generate_image(
            segment.image_prompt, 
            segment.segment_number
        )

    async def _generate_segment_audio_async(self, segment: VideoSegment) -> str:
        """Generate audio for a single segment (async wrapper)"""
        loop = asyncio.get_event_loop()
        
        # Run the blocking TTS operation in a thread pool
        return await loop.run_in_executor(
            None,
            self.video_service.generate_audio,
            segment.audio_script,
            segment.segment_number
        )

# Create global task manager
def create_video_task_manager(video_service: VideoGenerationService) -> VideoGenerationTasks:
    return VideoGenerationTasks(video_service)