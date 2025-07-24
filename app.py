from flask import Flask, render_template, jsonify, request

from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_pinecone import PineconeVectorStore

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

import google.generativeai as genai # Added for Gemini

from pymongo import MongoClient

from src.prompt import * # Imports both system_prompt and video_script_prompt

from datetime import datetime

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY') # Added for Gemini
MONGO_URI = os.environ.get("MONGO_URI")
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY # Set for Gemini

embeddings = download_hugging_face_embeddings()

index_name = "bot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize Gemini Client for ALL LLM operations
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro') # Or your chosen Gemini model, e.g., 'gemini-1.5-pro-latest'

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[MONGO_DB_NAME]
collection = db[COLLECTION_NAME]

def handle_query(query):
    retrieved_docs = retriever.invoke(query)
    if not retrieved_docs:
        return "No relevant information available from the documents."
    context = "\n".join(doc.page_content for doc in retrieved_docs)

    # Use the existing system_prompt for general RAG queries with Gemini
    # Combine system instructions, context, and user question into a single prompt for Gemini
    full_prompt_for_rag = system_prompt["content"].format(
        topic="general science", # A generic topic for the system prompt in this context
        difficulty_level="medium", # A generic difficulty for the system prompt in this context
        context=context
    ) + f"\n\nUser Question: {query}"

    try:
        completion = gemini_model.generate_content(
            [{"role": "user", "parts": [full_prompt_for_rag]}],
            generation_config=genai.types.GenerationConfig(
                temperature=0.7, # Adjusted for RAG to be more factual
                max_output_tokens=500,
                top_p=1,
            )
        )
        # Ensure to handle potential multiple parts in Gemini's response
        response_text = ""
        if completion.candidates and completion.candidates[0].content.parts:
            for part in completion.candidates[0].content.parts:
                response_text += part.text
        return response_text.strip()

    except Exception as e:
        print(f"Error with Gemini RAG query: {e}")
        return "Sorry, I couldn't process that query at the moment using Gemini."

def Youtube_chain(query):
    return handle_query(query)

def rag_chain(query):
    return Youtube_chain(query)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User Input (Chat):", msg)
    response = rag_chain(msg)
    print("Response (Chat):", response)
    chat_entry = {
        "user_message": msg,
        "bot_response": response,
        "timestamp": datetime.utcnow(),
        "type": "chat_query" # Added type for differentiation
    }
    collection.insert_one(chat_entry)
    return jsonify({"response": response})

@app.route("/generate_video_script", methods=["POST"])
def generate_video_script():
    # Expects JSON payload: {"topic": "about metal and non metal"}
    data = request.json
    topic = data.get("topic")

    if not topic:
        return jsonify({"error": "Missing 'topic' in request"}), 400

    # Format the video script prompt with the dynamic topic
    formatted_video_prompt_content = video_script_prompt["content"].format(topic=topic)

    try:
        # Call Gemini API for script generation
        response = gemini_model.generate_content(
            [
                {"role": "user", "parts": [formatted_video_prompt_content]},
                # Adding a model response and user follow-up can sometimes guide the LLM better
                {"role": "model", "parts": ["Understood. I will generate a 30-second video script about " + topic + " with audio and image prompts for each 10-second segment."]},
                {"role": "user", "parts": ["Please proceed with generating the script."]}
            ],
            generation_config=genai.types.GenerationConfig(
                temperature=0.9, # Higher temperature for creativity in script generation
                max_output_tokens=1000, # Increased max tokens for potentially longer script
                top_p=1,
            )
        )

        # Extract the text content from the response
        script_content = ""
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                script_content += part.text

        # Save the script generation event to MongoDB
        chat_entry = {
            "user_message": f"Generate video script for topic: {topic}",
            "bot_response": script_content,
            "timestamp": datetime.utcnow(),
            "type": "video_script_generation" # Differentiate this entry
        }
        collection.insert_one(chat_entry)

        return jsonify({"script": script_content})

    except Exception as e:
        print(f"Error generating video script with Gemini: {e}")
        return jsonify({"error": "Failed to generate video script", "details": str(e)}), 500


@app.route("/history", methods=["GET"])
def chat_history():
    history = list(collection.find().sort("timestamp", -1).limit(10))
    # You might want to process 'history' to display it nicely on the frontend,
    # potentially distinguishing between chat_query and video_script_generation.
    return jsonify({"history": history})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)