#!/usr/bin/env python3
"""
Quick Fix for Sahayak Video Audio Issues
"""
import os
import subprocess
import tempfile
import glob
import shutil

def fix_ffmpeg_command(video_generator_path="src/video_generator.py"):
    """Fix the FFmpeg commands in video_generator.py"""
    try:
        # Create backup
        backup_path = f"{video_generator_path}.bak"
        shutil.copy2(video_generator_path, backup_path)
        print(f"✅ Created backup at {backup_path}")
        
        # Read file
        with open(video_generator_path, 'r') as f:
            content = f.read()
        
        # Fix segment creation command
        old_segment = """                cmd = [
                    'ffmpeg', '-y',
                    '-loop', '1', '-i', segment.image_path,
                    '-i', segment.audio_path,"""
        
        new_segment = """                cmd = [
                    'ffmpeg', '-y',
                    '-loop', '1', '-i', segment.image_path,
                    '-i', segment.audio_path,"""
        
        # Fix audio encoding section
        old_audio = """                    # Audio encoding
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-ar', '44100',
                    '-ac', '2',"""
        
        new_audio = """                    # Audio encoding - improved for better inclusion
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-ar', '44100',
                    '-ac', '2',
                    '-af', 'volume=4.0',  # Increase volume"""
        
        # Fix audio conversion
        old_conversion = """            # 2. Convert MP3 to WAV (Simple, correct conversion)
            conversion_cmd = [
                'ffmpeg', '-y',
                '-i', temp_mp3,
                '-acodec', 'pcm_s16le',  # Standard 
                '-ar', '44100',
                '-ac', '2',             # Stereo audio
                temp_path
            ]"""
        
        new_conversion = """            # 2. Convert MP3 to WAV with enhanced audio
            conversion_cmd = [
                'ffmpeg', '-y',
                '-i', temp_mp3,
                '-acodec', 'pcm_s16le',  # Standard 
                '-ar', '44100',
                '-ac', '2',             # Stereo audio
                '-af', 'volume=4.0,loudnorm=I=-16:TP=-1.5:LRA=11',  # Normalize and boost volume
                temp_path
            ]"""
        
        # Apply fixes
        content = content.replace(old_audio, new_audio)
        content = content.replace(old_conversion, new_conversion)
        
        # Write fixed file
        with open(video_generator_path, 'w') as f:
            f.write(content)
        
        print(f"✅ Fixed FFmpeg commands in {video_generator_path}")
        return True
    except Exception as e:
        print(f"❌ Error fixing FFmpeg commands: {e}")
        return False

def fix_existing_videos(video_dir="/workspaces/Sahayak_backend/videos"):
    """Fix existing videos that have no audio"""
    if not os.path.exists(video_dir):
        os.makedirs(video_dir, exist_ok=True)
        print(f"Created videos directory: {video_dir}")
        return False
    
    videos = glob.glob(os.path.join(video_dir, "final_video_*.mp4"))
    if not videos:
        print("No existing videos found to fix")
        return False
    
    print(f"Found {len(videos)} videos to check")
    
    fixed_count = 0
    for video_path in videos:
        # Check if video has audio
        job_id = os.path.basename(video_path).replace("final_video_", "").replace(".mp4", "")
        
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_type', '-of', 'csv=p=0',
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if 'audio' in result.stdout:
                print(f"✅ Video already has audio: {os.path.basename(video_path)}")
                continue
            
            print(f"🔧 Fixing video: {os.path.basename(video_path)}")
            
            # Look for audio files in temp directory
            temp_dir = tempfile.gettempdir()
            audio_files = glob.glob(os.path.join(temp_dir, f"audio_segment_*_{job_id[-8:]}.wav"))
            if not audio_files:
                audio_files = glob.glob(os.path.join(temp_dir, "audio_segment_*.wav"))
            
            if not audio_files:
                print(f"❌ No audio files found for {os.path.basename(video_path)}")
                continue
            
            # Extract audio from first segment to add to video
            audio_path = audio_files[0]
            if not os.path.exists(audio_path):
                print(f"❌ Audio file not found: {audio_path}")
                continue
            
            # Normalize audio
            normalized_audio = os.path.join(temp_dir, f"normalized_{job_id}.wav")
            norm_cmd = [
                'ffmpeg', '-y', '-i', audio_path,
                '-af', 'volume=4.0,loudnorm=I=-16:TP=-1.5:LRA=11',
                '-ar', '44100', '-ac', '2',
                normalized_audio
            ]
            subprocess.run(norm_cmd, capture_output=True, timeout=60)
            
            if not os.path.exists(normalized_audio):
                print(f"❌ Failed to normalize audio for {os.path.basename(video_path)}")
                continue
            
            # Add audio to video
            fixed_path = os.path.join(video_dir, f"fixed_{os.path.basename(video_path)}")
            fix_cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', normalized_audio,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-shortest',
                '-map', '0:v:0',
                '-map', '1:a:0',
                fixed_path
            ]
            
            result = subprocess.run(fix_cmd, capture_output=True, timeout=300)
            
            if result.returncode == 0 and os.path.exists(fixed_path):
                # Replace original with fixed version
                os.replace(fixed_path, video_path)
                print(f"✅ Fixed and replaced: {os.path.basename(video_path)}")
                fixed_count += 1
            else:
                print(f"❌ Failed to fix: {os.path.basename(video_path)}")
                
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(video_path)}: {e}")
    
    print(f"Fixed {fixed_count} out of {len(videos)} videos")
    return fixed_count > 0

def main():
    print("🔧 SAHAYAK VIDEO AUDIO QUICK FIX 🔧")
    print("=" * 50)
    
    # 1. Fix FFmpeg commands in video_generator.py
    print("\n1️⃣ Fixing FFmpeg commands...")
    fix_ffmpeg_command()
    
    # 2. Fix existing videos
    print("\n2️⃣ Fixing existing videos...")
    fix_existing_videos()
    
    print("\n" + "=" * 50)
    print("✅ FIXES APPLIED")
    print("=" * 50)
    print("\nTo ensure audio works in new videos:")
    print("1. Restart your server to apply code changes")
    print("2. Generate a new test video")
    print("3. If still no audio, check if FFmpeg is installed:")
    print("   sudo apt-get update && sudo apt-get install -y ffmpeg")

if __name__ == "__main__":
    main()