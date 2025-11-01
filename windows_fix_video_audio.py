#!/usr/bin/env python3
"""
Windows-Compatible Quick Fix for Sahayak Video Audio Issues
"""
import os
import subprocess
import shutil
import glob

def fix_ffmpeg_command(video_generator_path="src/video_generator.py"):
    """Fix the FFmpeg commands in video_generator.py - Windows compatible"""
    try:
        # Create backup
        backup_path = f"{video_generator_path}.bak"
        shutil.copy2(video_generator_path, backup_path)
        print(f"Created backup at {backup_path}")
        
        # Read file with UTF-8 encoding
        with open(video_generator_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.readlines()
        
        # Process line by line to avoid encoding issues
        new_content = []
        in_audio_section = False
        in_conversion_section = False
        
        for i, line in enumerate(content):
            # Audio encoding in segment creation
            if '# Audio encoding' in line:
                in_audio_section = True
                new_content.append(line)
                continue
            
            if in_audio_section and "'-ac', '2'," in line:
                new_content.append(line)
                # Add volume boost line after channels
                new_content.append("                    '-af', 'volume=4.0',  # Increase volume\n")
                in_audio_section = False
                continue
            
            # Audio conversion in generate_audio method
            if "# 2. Convert MP3 to WAV" in line:
                in_conversion_section = True
                new_content.append("            # 2. Convert MP3 to WAV with enhanced audio\n")
                continue
            
            if in_conversion_section and "'-ac', '2'," in line:
                new_content.append(line)
                # Add normalization after channels
                new_content.append("                '-af', 'volume=4.0,loudnorm=I=-16:TP=-1.5:LRA=11',  # Normalize and boost volume\n")
                in_conversion_section = False
                continue
                
            # Keep the original line
            new_content.append(line)
        
        # Write fixed file
        with open(video_generator_path, 'w', encoding='utf-8') as f:
            f.writelines(new_content)
        
        print(f"Fixed FFmpeg commands in {video_generator_path}")
        return True
    except Exception as e:
        print(f"Error fixing FFmpeg commands: {e}")
        return False

def find_ffmpeg():
    """Check if FFmpeg is installed and available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              text=True)
        if result.returncode == 0:
            print("FFmpeg is installed and working")
            return True
        else:
            print("FFmpeg command returned an error")
            return False
    except FileNotFoundError:
        print("FFmpeg not found - please install it")
        return False
    except Exception as e:
        print(f"Error checking FFmpeg: {e}")
        return False

def main():
    print("SAHAYAK VIDEO AUDIO QUICK FIX - WINDOWS VERSION")
    print("=" * 50)
    
    # Check FFmpeg
    print("\nChecking FFmpeg installation...")
    ffmpeg_available = find_ffmpeg()
    
    if not ffmpeg_available:
        print("\nFFmpeg is required for video generation with audio.")
        print("Please install FFmpeg:")
        print("1. Download from: https://ffmpeg.org/download.html")
        print("2. Add it to your PATH environment variable")
        print("3. Restart your command prompt")
        
    # Fix FFmpeg commands in video_generator.py
    print("\nFixing FFmpeg commands...")
    fix_ffmpeg_command()
    
    print("\n" + "=" * 50)
    print("FIXES APPLIED")
    print("=" * 50)
    print("\nTo ensure audio works in new videos:")
    print("1. Make sure FFmpeg is installed and in your PATH")
    print("2. Restart your server to apply code changes")
    print("3. Generate a new test video")
    
    if not ffmpeg_available:
        print("\nNOTE: No FFmpeg detected! You MUST install FFmpeg for audio to work!")

if __name__ == "__main__":
    main()