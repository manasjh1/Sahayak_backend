#!/usr/bin/env python3
"""
COMPLETE STANDALONE FIX for Sahayak Video Audio Issues
Just run this file and it will fix everything in one go
"""
import os
import subprocess
import glob
import shutil
import tempfile
from gtts import gTTS
import uuid

def fix_audio_in_video_generator():
    """Fix all audio-related code in video_generator.py"""
    file_path = "src/video_generator.py"
    if not os.path.exists(file_path):
        print(f"❌ Cannot find {file_path}")
        return False
    
    # Create backup
    backup_path = f"{file_path}.bak"
    shutil.copy2(file_path, backup_path)
    print(f"✅ Created backup at {backup_path}")
    
    # Read original file
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # Initialize flags for finding sections
    in_generate_audio = False
    in_create_video = False
    in_segment_cmd = False
    in_concat_cmd = False
    generate_audio_start = 0
    create_video_start = 0
    segment_cmd_start = 0
    concat_cmd_start = 0
    
    # Find important sections
    for i, line in enumerate(lines):
        if "def generate_audio(" in line:
            in_generate_audio = True
            generate_audio_start = i
        elif "def create_video_from_segments(" in line:
            in_create_video = True
            create_video_start = i
        elif in_create_video and "cmd = [" in line and "'ffmpeg'" in line and not in_segment_cmd:
            in_segment_cmd = True
            segment_cmd_start = i
        elif in_create_video and "concat_cmd = [" in line:
            in_concat_cmd = True
            concat_cmd_start = i
    
    # Replace generate_audio function with improved version
    if generate_audio_start > 0:
        # Find the end of the function
        function_end = generate_audio_start
        indent_level = 0
        found_def = False
        for i in range(generate_audio_start, len(lines)):
            if "def " in lines[i] and i > generate_audio_start + 1:
                function_end = i
                break
            if i == len(lines) - 1:
                function_end = i + 1
        
        # Replace the function with our improved version
        improved_generate_audio = """    def generate_audio(self, text: str, segment_number: int) -> Optional[str]:
        \"\"\"Generate high-quality audio using working gTTS\"\"\"
    
        temp_path = os.path.join(tempfile.gettempdir(), 
                            f"audio_{segment_number}_{uuid.uuid4().hex[:8]}.wav")
        
        try:
            print(f"🎵 Generating audio for segment {segment_number} using gTTS...")
            from gtts import gTTS
            
            # 1. Save to temporary MP3
            temp_mp3 = temp_path.replace('.wav', '_temp.mp3')
            tts = gTTS(text=text, lang='en', tld='com', slow=False)
            tts.save(temp_mp3)
            
            if not os.path.exists(temp_mp3) or os.path.getsize(temp_mp3) == 0:
                print("❌ gTTS failed to generate MP3")
                return None
            
            print(f"✅ Generated MP3 (gTTS): {os.path.getsize(temp_mp3)} bytes")

            # 2. Convert MP3 to WAV with IMPROVED audio settings
            conversion_cmd = [
                'ffmpeg', '-y',
                '-i', temp_mp3,
                '-acodec', 'pcm_s16le',  # Standard format
                '-ar', '44100',          # Standard sample rate
                '-ac', '2',              # Stereo audio
                '-af', 'volume=4.0,loudnorm=I=-16:TP=-1.5:LRA=11',  # Normalize audio and increase volume
                temp_path
            ]
            
            result = subprocess.run(conversion_cmd, capture_output=True, text=True, timeout=60)
            
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)
            
            if result.returncode == 0 and os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                print(f"✅ Audio ready (gTTS): {os.path.getsize(temp_path)} bytes")
                return temp_path
            else:
                print(f"❌ gTTS audio conversion failed: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"❌ gTTS generation failed: {e}")
            return None

"""
        
        # Insert the improved function
        lines = lines[:generate_audio_start] + [improved_generate_audio] + lines[function_end:]
    
    # Create new content for fixing FFmpeg commands
    modified_content = ""
    
    # Process the file line by line for other fixes
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Fix FFmpeg segment creation command
        if i >= segment_cmd_start and "cmd = [" in line and "'ffmpeg'" in line:
            segment_cmd_block = []
            # Collect the entire command block
            while i < len(lines) and "]" not in lines[i]:
                segment_cmd_block.append(lines[i])
                i += 1
            segment_cmd_block.append(lines[i])  # Add the closing bracket line
            
            # Convert to string to make replacements
            cmd_str = "".join(segment_cmd_block)
            
            # Fix 1: Add audio filter for volume boost
            if "'-af'" not in cmd_str:
                cmd_str = cmd_str.replace(
                    "'-ac', '2',", 
                    "'-ac', '2',\n                    '-af', 'volume=4.0,loudnorm',  # Increase volume"
                )
            
            # Fix 2: Add audio tune for stillimage
            if "'-tune'" not in cmd_str:
                cmd_str = cmd_str.replace(
                    "'-c:v', 'libx264',", 
                    "'-c:v', 'libx264',\n                    '-tune', 'stillimage',  # Optimize for still images"
                )
            
            # Fix 3: Add avoid_negative_ts
            if "'-avoid_negative_ts'" not in cmd_str:
                cmd_str = cmd_str.replace(
                    "'-shortest',", 
                    "'-shortest',\n                    '-avoid_negative_ts', '1',  # Avoid timing issues"
                )
            
            modified_content += cmd_str
            i += 1  # Already incremented in the loop
            continue
        
        # Fix FFmpeg concat command
        if i >= concat_cmd_start and "concat_cmd = [" in line:
            concat_cmd_block = []
            # Collect the entire command block
            while i < len(lines) and "]" not in lines[i]:
                concat_cmd_block.append(lines[i])
                i += 1
            concat_cmd_block.append(lines[i])  # Add the closing bracket line
            
            # Convert to string to make replacements
            cmd_str = "".join(concat_cmd_block)
            
            # Fix 1: Increase audio bitrate
            cmd_str = cmd_str.replace("'-b:a', '128k',", "'-b:a', '192k',  # Higher bitrate for better audio")
            
            # Fix 2: Add stereo channels if missing
            if "'-ac'" not in cmd_str:
                cmd_str = cmd_str.replace(
                    "'-ar', '44100',", 
                    "'-ar', '44100',\n                '-ac', '2',  # Stereo audio"
                )
            
            # Fix 3: Add audio filter for volume boost
            if "'-af'" not in cmd_str:
                cmd_str = cmd_str.replace(
                    "'-ar', '44100',", 
                    "'-ar', '44100',\n                '-af', 'volume=4.0',  # Increase volume"
                )
            
            modified_content += cmd_str
            i += 1  # Already incremented in the loop
            continue
        
        modified_content += line
        i += 1
    
    # Add fix_video_audio method at the end of the class
    fix_method = """
    def fix_video_audio(self, video_path: str, output_path: Optional[str] = None) -> Optional[str]:
        \"\"\"Fix video by boosting audio volume\"\"\"
        if output_path is None:
            output_path = video_path.replace('.mp4', '_fixed.mp4')
        
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-c:v', 'copy',           # Copy video stream without re-encoding
            '-c:a', 'aac',            # AAC audio codec
            '-b:a', '192k',           # Higher bitrate
            '-af', 'volume=4.0',      # Increase volume
            output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, timeout=300)
            if os.path.exists(output_path):
                return output_path
        except Exception as e:
            print(f"Error fixing video: {e}")
        
        return None
"""
    
    # Find the end of the class to add the new method
    last_method_end = 0
    for i in range(len(modified_content) - 1, 0, -1):
        if "def " in modified_content[i:i+5] and "    " in modified_content[i:i+5]:
            # Find the end of this method
            for j in range(i + 1, len(modified_content)):
                if j == len(modified_content) - 1 or ("def " in modified_content[j:j+5] and "    " in modified_content[j:j+5]):
                    last_method_end = j
                    break
            break
    
    # Insert the new method
    if last_method_end > 0:
        modified_content = modified_content[:last_method_end] + fix_method + modified_content[last_method_end:]
    else:
        # Fallback - just append to the end
        modified_content += fix_method
    
    # Write the modified file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print(f"✅ Fixed audio issues in {file_path}")
    return True

def fix_existing_videos():
    """Fix all existing videos by boosting audio"""
    # Detect video directory
    video_dirs = [
        "/workspaces/Sahayak_backend/videos",
        "./videos",
        "../videos",
        os.path.join(os.getcwd(), "videos")
    ]
    
    video_dir = None
    for dir_path in video_dirs:
        if os.path.exists(dir_path):
            video_dir = dir_path
            break
    
    if not video_dir:
        print("❌ Cannot find videos directory")
        # Create a videos directory
        video_dir = os.path.join(os.getcwd(), "videos")
        os.makedirs(video_dir, exist_ok=True)
        print(f"✅ Created videos directory: {video_dir}")
    
    # Find all videos
    videos = glob.glob(os.path.join(video_dir, "final_video_*.mp4"))
    print(f"Found {len(videos)} videos to process")
    
    # Process each video
    fixed_count = 0
    for video_path in videos:
        print(f"Processing: {os.path.basename(video_path)}")
        
        # Create a fixed version
        fixed_path = video_path.replace('.mp4', '_fixed.mp4')
        
        # Use FFmpeg to fix the audio
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-c:v', 'copy',           # Copy video stream without re-encoding
            '-c:a', 'aac',            # AAC audio codec
            '-b:a', '192k',           # Higher bitrate
            '-af', 'volume=4.0',      # Increase volume
            fixed_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, timeout=300)
            
            if os.path.exists(fixed_path) and os.path.getsize(fixed_path) > 0:
                # Backup original
                backup_path = video_path.replace('.mp4', '_original.mp4')
                shutil.copy2(video_path, backup_path)
                
                # Replace with fixed version
                shutil.copy2(fixed_path, video_path)
                print(f"✅ Fixed: {os.path.basename(video_path)}")
                fixed_count += 1
            else:
                print(f"❌ Failed to fix: {os.path.basename(video_path)}")
                
        except Exception as e:
            print(f"❌ Error fixing {os.path.basename(video_path)}: {e}")
    
    print(f"Fixed {fixed_count} out of {len(videos)} videos")
    return fixed_count > 0

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE)
        return result.returncode == 0
    except:
        return False

def check_gtts():
    """Check if gTTS is installed"""
    try:
        from gtts import gTTS
        return True
    except:
        return False

def install_gtts():
    """Install gTTS package"""
    try:
        subprocess.run(['pip', 'install', 'gtts'], check=True)
        return True
    except:
        return False

def generate_test_audio():
    """Generate a test audio file to verify TTS is working"""
    try:
        temp_dir = tempfile.gettempdir()
        test_file = os.path.join(temp_dir, f"test_audio_{uuid.uuid4().hex[:8]}.mp3")
        
        tts = gTTS("This is a test audio file for the Sahayak system", lang='en')
        tts.save(test_file)
        
        if os.path.exists(test_file) and os.path.getsize(test_file) > 0:
            print(f"✅ Test audio generation successful: {test_file}")
            os.remove(test_file)
            return True
        else:
            print("❌ Test audio file empty or not created")
            return False
    except Exception as e:
        print(f"❌ Test audio generation failed: {e}")
        return False

def main():
    print("\n" + "=" * 50)
    print("🔧 SAHAYAK VIDEO AUDIO COMPLETE FIXER 🔧")
    print("=" * 50)
    
    # 1. Check dependencies
    print("\n1️⃣ Checking dependencies...")
    
    ffmpeg_installed = check_ffmpeg()
    if ffmpeg_installed:
        print("✅ FFmpeg is installed")
    else:
        print("❌ FFmpeg is missing! Audio won't work without it.")
        print("   Install FFmpeg from: https://ffmpeg.org/download.html")
    
    gtts_installed = check_gtts()
    if gtts_installed:
        print("✅ gTTS is installed")
    else:
        print("❌ gTTS is missing, attempting to install...")
        if install_gtts():
            print("✅ gTTS installed successfully")
            gtts_installed = True
        else:
            print("❌ gTTS installation failed, please install manually:")
            print("   pip install gtts")
    
    # 2. Test audio generation
    if gtts_installed:
        print("\n2️⃣ Testing TTS system...")
        if generate_test_audio():
            print("✅ TTS system is working")
        else:
            print("❌ TTS system test failed")
    
    # 3. Fix video generator code
    print("\n3️⃣ Fixing video generator code...")
    fix_audio_in_video_generator()
    
    # 4. Fix existing videos
    print("\n4️⃣ Fixing existing videos...")
    fix_existing_videos()
    
    print("\n" + "=" * 50)
    print("✅ FIXES COMPLETED")
    print("=" * 50)
    
    print("\nNext steps:")
    print("1. Restart your server:")
    print("   kill -9 $(pgrep -f 'python app.py')  # Stop current server")
    print("   python app.py                        # Start server again")
    print("\n2. Generate a new test video:")
    print("   curl -X POST \"http://localhost:8000/generate-video\" -H \"Content-Type: application/json\" -d '{\"topic\":\"photosynthesis\",\"duration\":30,\"style\":\"educational\"}'")
    
    if not ffmpeg_installed:
        print("\n⚠️ WARNING: FFmpeg is not installed! Audio won't work without it.")
        print("   Install FFmpeg immediately to fix audio issues!")

if __name__ == "__main__":
    main()