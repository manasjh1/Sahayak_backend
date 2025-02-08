import os
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

class DBHandler:
    def __init__(self):
        # Get MongoDB credentials from .env
        self.mongo_uri = os.getenv("MONGO_URI")
        self.db_name = os.getenv("MONGO_DB_NAME")
        self.collection_name = os.getenv("COLLECTION_NAME")
        
        # Connect to MongoDB
        if not self.mongo_uri or not self.db_name or not self.collection_name:
            raise ValueError("Missing database configuration in .env file")
        
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]

    def save_chat(self, user_message, bot_message):
        """
        Save chat history to the MongoDB collection.
        """
        chat_entry = {
            "user_message": user_message,
            "bot_message": bot_message,
            "timestamp": datetime.utcnow()
        }
        self.collection.insert_one(chat_entry)
        print(f"Chat saved: {chat_entry}")

    def get_chat_history(self, limit=10):
        """
        Retrieve the last `limit` chat messages from the collection.
        """
        return list(self.collection.find().sort("timestamp", -1).limit(limit))
