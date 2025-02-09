from flask import Flask, render_template, jsonify, request
from src.helper import load_pdf_file, text_split
from src.helper import download_hugging_face_embeddings
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from groq import Groq
from pymongo import MongoClient
from src.prompt import *
from datetime import datetime

app = Flask(__name__)

load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
MONGO_URI = os.environ.get("MONGO_URI")
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "bot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

client = Groq(api_key=GROQ_API_KEY)

# Connect to MongoDB
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[MONGO_DB_NAME]
collection = db[COLLECTION_NAME]


def handle_query(query):
    retrieved_docs = retriever.invoke(query)

    if not retrieved_docs:
        return "No relevant information available."

    context = "\n".join(doc.page_content for doc in retrieved_docs)

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            system_prompt,
            {"role": "user", "content": f"{context}\n\nUser question: {query}"}
        ],
        temperature=0.4,
        max_completion_tokens=500,
        top_p=1,
        stream=False,
        stop=None,
    )

    return completion.choices[0].message.content.strip()


def question_answer_chain(query):
    return handle_query(query)


def rag_chain(query):
    return question_answer_chain(query)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User Input:", msg)
    
    # Process user input
    response = rag_chain(msg)
    print("Response:", response)

    # Save chat to MongoDB
    chat_entry = {
        "user_message": msg,
        "bot_response": response,
        "timestamp": datetime.utcnow()
    }
    collection.insert_one(chat_entry)  # Save to the MongoDB collection

    return jsonify({"response": response})


@app.route("/history", methods=["GET"])
def chat_history():
    """
    Fetch and display the last 10 chat history records.
    """
    history = list(collection.find().sort("timestamp", -1).limit(10))
    return jsonify({"history": history})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)