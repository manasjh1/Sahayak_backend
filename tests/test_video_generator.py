import unittest
import os
from src.video_generator import VideoGenerator

class TestVideoGenerator(unittest.TestCase):
    
    def setUp(self):
        self.generator = VideoGenerator()
        
    def test_frame_generation(self):
        """Test generating individual frames"""
        frame = self.generator.generate_frame("photosynthesis", 1, 30)
        self.assertIsNotNone(frame)
        
    def test_audio_generation(self):
        """Test audio synthesis"""
        audio_path = self.generator.generate_audio("photosynthesis")
        self.assertTrue(os.path.exists(audio_path))
        self.assertTrue(os.path.getsize(audio_path) > 0)
        
    def test_full_pipeline(self):
        """Test end-to-end video generation"""
        video_path = self.generator.create_video("photosynthesis", 5)
        self.assertTrue(os.path.exists(video_path))
        # Verify video has audio stream
        self.assertTrue(self.generator.verify_video_has_audio(video_path))

if __name__ == '__main__':
    unittest.main()