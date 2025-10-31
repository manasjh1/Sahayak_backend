import os
import tempfile
import subprocess
import uuid
from typing import Optional
import requests
import json

class GoogleTTSService:
    """Google TTS Service with multiple options"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('GOOGLE_CLOUD_TTS_API_KEY')
        self.cloud_tts_available = bool(self.api_key)
        
        # Google Cloud TTS settings
        self.cloud_tts_url = "https://texttospeech.googleapis.com/v1/text:synthesize"
        
        # Voice configurations
        self.voices = {
            'female_us': {
                'cloud': {'languageCode': 'en-US', 'name': 'en-US-Neural2-F', 'ssmlGender': 'FEMALE'},
                'gtts': {'lang': 'en', 'tld': 'com', 'slow': False}
            },
            'male_us': {
                'cloud': {'languageCode': 'en-US', 'name': 'en-US-Neural2-D', 'ssmlGender': 'MALE'},
                'gtts': {'lang': 'en', 'tld': 'com', 'slow': False}
            },
            'female_uk': {
                'cloud': {'languageCode': 'en-GB', 'name': 'en-GB-Neural2-A', 'ssmlGender': 'FEMALE'},
                'gtts': {'lang': 'en', 'tld': 'co.uk', 'slow': False}
            }
        }
        
        self.default_voice = 'female_us'
    
    def generate_audio_cloud_tts(self, text: str, output_path: str, voice: str = None) -> bool:
        """Generate audio using Google Cloud TTS API"""
        if not self.cloud_tts_available:
            print("❌ Google Cloud TTS API key not available")
            return False
        
        try:
            if voice is None:
                voice = self.default_voice
            
            voice_config = self.voices.get(voice, self.voices[self.default_voice])['cloud']
            
            print(f"🎵 Generating audio with Google Cloud TTS: {voice_config['name']}")
            
            # Prepare request payload
            payload = {
                "input": {"text": text},
                "voice": voice_config,
                "audioConfig": {
                    "audioEncoding": "MP3",
                    "speakingRate": 1.0,
                    "pitch": 0.0,
                    "volumeGainDb": 6.0,  # Increase volume
                    "sampleRateHertz": 44100
                }
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Make API request
            response = requests.post(
                f"{self.cloud_tts_url}?key={self.api_key}",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"❌ Google Cloud TTS API error: {response.status_code} - {response.text}")
                return False
            
            # Decode base64 audio content
            import base64
            audio_content = base64.b64decode(response.json()['audioContent'])
            
            # Save as temporary MP3
            temp_mp3 = output_path.replace('.wav', '_temp.mp3')
            with open(temp_mp3, 'wb') as f:
                f.write(audio_content)
            
            print(f"✅ Generated MP3: {os.path.getsize(temp_mp3)} bytes")
            
            # Convert MP3 to WAV with enhanced audio
            conversion_cmd = [
                'ffmpeg', '-y',
                '-i', temp_mp3,
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                '-ac', '2',
                '-af', 'volume=3.0,loudnorm',  # Loud and normalized
                output_path
            ]
            
            result = subprocess.run(conversion_cmd, capture_output=True, text=True, timeout=60)
            
            # Clean up temp MP3
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)
            
            if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"✅ Converted to WAV: {os.path.getsize(output_path)} bytes")
                return True
            else:
                print(f"❌ MP3 to WAV conversion failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Google Cloud TTS error: {e}")
            return False
    
    def generate_audio_gtts(self, text: str, output_path: str, voice: str = None) -> bool:
        """Generate audio using free gTTS (Google Translate TTS)"""
        try:
            from gtts import gTTS
            
            if voice is None:
                voice = self.default_voice
            
            voice_config = self.voices.get(voice, self.voices[self.default_voice])['gtts']
            
            print(f"🎵 Generating audio with gTTS: {voice_config}")
            
            # Create gTTS object
            tts = gTTS(
                text=text,
                lang=voice_config['lang'],
                tld=voice_config['tld'],
                slow=voice_config['slow']
            )
            
            # Save to temporary MP3
            temp_mp3 = output_path.replace('.wav', '_temp.mp3')
            tts.save(temp_mp3)
            
            if not os.path.exists(temp_mp3) or os.path.getsize(temp_mp3) == 0:
                print("❌ gTTS failed to generate MP3")
                return False
            
            print(f"✅ Generated MP3: {os.path.getsize(temp_mp3)} bytes")
            
            # Convert MP3 to WAV with loud volume
            conversion_cmd = [
                'ffmpeg', '-y',
                '-i', temp_mp3,
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                '-ac', '2',
                '-af', 'volume=4.0,loudnorm',  # Very loud and normalized
                output_path
            ]
            
            result = subprocess.run(conversion_cmd, capture_output=True, text=True, timeout=60)
            
            # Clean up temp MP3
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)
            
            if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"✅ Converted to WAV: {os.path.getsize(output_path)} bytes")
                return True
            else:
                print(f"❌ MP3 to WAV conversion failed: {result.stderr}")
                return False
                
        except ImportError:
            print("❌ gTTS not installed. Run: pip install gtts")
            return False
        except Exception as e:
            print(f"❌ gTTS error: {e}")
            return False
    
    def generate_audio(self, text: str, output_path: str, voice: str = None) -> bool:
        """Generate audio with automatic fallback"""
        
        # Try Google Cloud TTS first (if API key available)
        if self.cloud_tts_available:
            print("🎵 Trying Google Cloud TTS...")
            if self.generate_audio_cloud_tts(text, output_path, voice):
                return True
            else:
                print("⚠️ Google Cloud TTS failed, trying gTTS...")
        
        # Fallback to free gTTS
        if self.generate_audio_gtts(text, output_path, voice):
            return True
        
        # Last resort: eSpeak
        print("⚠️ gTTS failed, trying eSpeak fallback...")
        try:
            cmd = [
                'espeak', '-s', '150', '-p', '50', '-a', '200',
                '-v', 'en+f3', '-w', output_path, text
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"✅ eSpeak fallback successful: {os.path.getsize(output_path)} bytes")
                return True
            else:
                print(f"❌ eSpeak failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ eSpeak error: {e}")
            return False
    
    def test_all_methods(self, test_text="Hello, this is a test of Google TTS service."):
        """Test all available TTS methods"""
        print("🧪 Testing Google TTS Service...")
        
        temp_dir = tempfile.gettempdir()
        results = {}
        
        # Test Cloud TTS
        if self.cloud_tts_available:
            print("\n--- Testing Google Cloud TTS ---")
            test_file = os.path.join(temp_dir, f"test_cloud_{uuid.uuid4().hex[:8]}.wav")
            
            if self.generate_audio_cloud_tts(test_text, test_file):
                print("✅ Google Cloud TTS: SUCCESS")
                results['cloud_tts'] = True
                os.remove(test_file)
            else:
                print("❌ Google Cloud TTS: FAILED")
                results['cloud_tts'] = False
        else:
            print("⚠️ Google Cloud TTS: API key not provided")
            results['cloud_tts'] = False
        
        # Test gTTS
        print("\n--- Testing gTTS ---")
        test_file = os.path.join(temp_dir, f"test_gtts_{uuid.uuid4().hex[:8]}.wav")
        
        if self.generate_audio_gtts(test_text, test_file):
            print("✅ gTTS: SUCCESS")
            results['gtts'] = True
            os.remove(test_file)
        else:
            print("❌ gTTS: FAILED")
            results['gtts'] = False
        
        # Summary
        print("\n" + "="*40)
        print("🏁 GOOGLE TTS TEST RESULTS:")
        print("="*40)
        
        working_methods = []
        for method, success in results.items():
            status = "✅ WORKING" if success else "❌ FAILED"
            print(f"{method}: {status}")
            if success:
                working_methods.append(method)
        
        if working_methods:
            print(f"\n🎉 Available methods: {', '.join(working_methods)}")
        else:
            print("\n❌ No Google TTS methods working!")
        
        return working_methods

# Integration class for your video generator
class GeminiTTSService:
    """TTS Service using Google technologies (renamed for your preference)"""
    
    def __init__(self, api_key=None):
        self.google_tts = GoogleTTSService(api_key)
        self.voice_options = ['female_us', 'male_us', 'female_uk']
        self.default_voice = 'female_us'
    
    def generate_audio(self, text: str, segment_number: int, voice: str = None) -> Optional[str]:
        """Generate audio using Google TTS technologies"""
        temp_path = os.path.join(
            tempfile.gettempdir(), 
            f"audio_segment_{segment_number}_{uuid.uuid4().hex[:8]}.wav"
        )
        
        if voice is None:
            voice = self.default_voice
        
        print(f"🎵 Generating audio for segment {segment_number} using Google TTS...")
        
        if self.google_tts.generate_audio(text, temp_path, voice):
            print(f"✅ Google TTS successful: {os.path.getsize(temp_path)} bytes")
            return temp_path
        else:
            print(f"❌ Google TTS failed for segment {segment_number}")
            return None
    
    def set_voice(self, voice: str):
        """Set the default voice"""
        if voice in self.voice_options:
            self.default_voice = voice
            print(f"🎤 Voice set to: {voice}")
        else:
            print(f"⚠️ Voice '{voice}' not available. Options: {self.voice_options}")

def install_requirements():
    """Install required packages"""
    import subprocess
    
    packages = ['gtts', 'requests']
    
    for package in packages:
        try:
            subprocess.run(['pip', 'install', package], check=True)
            print(f"✅ Installed {package}")
        except:
            print(f"❌ Failed to install {package}")

def test_google_tts():
    """Test Google TTS installation and functionality"""
    print("🧪 Testing Google TTS Service...")
    
    # Install requirements
    install_requirements()
    
    # Test with and without API key
    api_key = os.getenv('GOOGLE_CLOUD_TTS_API_KEY')
    
    tts = GoogleTTSService(api_key)
    working_methods = tts.test_all_methods()
    
    if working_methods:
        print(f"\n✅ Google TTS is working! Available: {working_methods}")
        return True
    else:
        print("\n❌ Google TTS setup failed")
        return False

if __name__ == "__main__":
    test_google_tts()

