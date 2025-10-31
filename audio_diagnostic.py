#!/usr/bin/env python3
"""
Audio Diagnostic Script - Check why videos have no sound
"""

import subprocess
import os
import json

def check_video_audio_detailed(video_path):
    """Detailed audio stream analysis"""
    print(f"🔍 Analyzing video: {video_path}")
    
    if not os.path.exists(video_path):
        print("❌ Video file not found")
        return False
    
    file_size = os.path.getsize(video_path)
    print(f"📁 File size: {file_size} bytes ({file_size/1024/1024:.2f} MB)")
    
    try:
        # Get detailed stream information
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', '-show_format', video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"❌ FFprobe failed: {result.stderr}")
            return False
        
        data = json.loads(result.stdout)
        
        print("\n📊 STREAM ANALYSIS:")
        print("=" * 40)
        
        video_streams = []
        audio_streams = []
        
        for i, stream in enumerate(data.get('streams', [])):
            codec_type = stream.get('codec_type', 'unknown')
            codec_name = stream.get('codec_name', 'unknown')
            
            if codec_type == 'video':
                video_streams.append(stream)
                print(f"🎥 Video Stream {i}:")
                print(f"   Codec: {codec_name}")
                print(f"   Resolution: {stream.get('width', '?')}x{stream.get('height', '?')}")
                print(f"   Frame Rate: {stream.get('r_frame_rate', '?')}")
                print(f"   Duration: {stream.get('duration', '?')} seconds")
                
            elif codec_type == 'audio':
                audio_streams.append(stream)
                print(f"🔊 Audio Stream {i}:")
                print(f"   Codec: {codec_name}")
                print(f"   Sample Rate: {stream.get('sample_rate', '?')} Hz")
                print(f"   Channels: {stream.get('channels', '?')}")
                print(f"   Bit Rate: {stream.get('bit_rate', '?')} bps")
                print(f"   Duration: {stream.get('duration', '?')} seconds")
                
                # Check if audio stream has actual data
                if 'duration' in stream:
                    duration = float(stream['duration'])
                    if duration > 0:
                        print(f"   ✅ Audio duration: {duration:.2f}s")
                    else:
                        print(f"   ❌ Audio duration is 0")
                else:
                    print(f"   ⚠️ No duration information")
        
        print(f"\n📈 SUMMARY:")
        print(f"Video streams: {len(video_streams)}")
        print(f"Audio streams: {len(audio_streams)}")
        
        if len(audio_streams) == 0:
            print("❌ NO AUDIO STREAMS FOUND")
            return False
        elif len(audio_streams) > 0:
            print("✅ Audio streams detected")
            
            # Check if audio has content
            for i, stream in enumerate(audio_streams):
                if 'duration' in stream and float(stream['duration']) > 0:
                    print(f"✅ Audio stream {i} has content")
                else:
                    print(f"❌ Audio stream {i} appears empty")
        
        return len(audio_streams) > 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False

def test_audio_extraction(video_path):
    """Try to extract audio to see if it exists"""
    print(f"\n🎵 Testing audio extraction...")
    
    try:
        audio_output = video_path.replace('.mp4', '_extracted_audio.wav')
        
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM audio
            '-ar', '44100',  # Sample rate
            '-ac', '2',  # Stereo
            audio_output
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and os.path.exists(audio_output):
            audio_size = os.path.getsize(audio_output)
            print(f"✅ Audio extracted: {audio_size} bytes")
            
            if audio_size > 1000:  # Should have some content
                print("✅ Audio appears to have content")
                
                # Get audio duration
                dur_cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 
                          'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', 
                          audio_output]
                dur_result = subprocess.run(dur_cmd, capture_output=True, text=True)
                
                if dur_result.returncode == 0:
                    duration = float(dur_result.stdout.strip())
                    print(f"🕐 Extracted audio duration: {duration:.2f}s")
                
                # Clean up
                os.remove(audio_output)
                return True
            else:
                print("❌ Extracted audio is too small (likely empty)")
                os.remove(audio_output)
                return False
        else:
            print(f"❌ Audio extraction failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Audio extraction test failed: {e}")
        return False

def suggest_fixes():
    """Suggest audio fixes based on common issues"""
    print(f"\n🔧 COMMON AUDIO FIXES:")
    print("=" * 40)
    
    fixes = [
        "1. Audio Codec Issue:",
        "   Try different audio codec: -c:a libmp3lame",
        "   Or use: -c:a pcm_s16le for uncompressed audio",
        "",
        "2. Sample Rate Issue:", 
        "   Force standard sample rate: -ar 44100",
        "   Or try: -ar 48000",
        "",
        "3. Channel Layout Issue:",
        "   Force stereo: -ac 2",
        "   Or mono: -ac 1",
        "",
        "4. Audio Sync Issue:",
        "   Add audio delay: -itsoffset 0.1 -i audio.wav",
        "   Or force sync: -async 1",
        "",
        "5. Container Issue:",
        "   Try different container: output.mkv instead of .mp4",
        "   Or remux: ffmpeg -i input.mp4 -c copy output_remux.mp4"
    ]
    
    for fix in fixes:
        print(fix)

def run_audio_diagnostic():
    """Run complete audio diagnostic"""
    print("🔍 AUDIO DIAGNOSTIC STARTING")
    print("=" * 50)
    
    # Find the most recent video
    video_dir = "/workspaces/Sahayak_backend/videos"
    
    if not os.path.exists(video_dir):
        print(f"❌ Video directory not found: {video_dir}")
        return
    
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    
    if not video_files:
        print(f"❌ No video files found in {video_dir}")
        return
    
    # Sort by modification time (newest first)
    video_files.sort(key=lambda x: os.path.getmtime(os.path.join(video_dir, x)), reverse=True)
    
    latest_video = os.path.join(video_dir, video_files[0])
    print(f"📹 Testing latest video: {video_files[0]}")
    
    # Run diagnostics
    has_audio_streams = check_video_audio_detailed(latest_video)
    
    if has_audio_streams:
        print("\n" + "="*50)
        audio_extractable = test_audio_extraction(latest_video)
        
        if not audio_extractable:
            print("❌ Audio streams exist but appear empty")
        else:
            print("✅ Audio streams have content")
    
    # Always show fixes
    suggest_fixes()
    
    print("\n" + "="*50)
    print("🏁 DIAGNOSTIC COMPLETE")

if __name__ == "__main__":
    run_audio_diagnostic()