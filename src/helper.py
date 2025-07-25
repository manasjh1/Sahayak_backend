import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai

load_dotenv()

def load_pdf_file(data_path):
    pdf_loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(
        data_path,
        glob="*.txt",
        loader_cls=lambda path: TextLoader(path, encoding="utf-8")
    )
    
    pdf_documents = pdf_loader.load()
    txt_documents = txt_loader.load()
    return pdf_documents + txt_documents

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

class GeminiEmbedding:
    def __init__(self, api_key, max_retries=3, timeout=60):
        genai.configure(api_key=api_key)
        self.max_retries = max_retries
        self.timeout = timeout
        self.retry_delay = 2
    
    def embed_documents(self, texts):
        """For document embedding during indexing"""
        print(f"📄 Embedding {len(texts)} documents...")
        return self._embed_texts_batch(texts, task_type="retrieval_document")
    
    def embed_query(self, text):
        """For query embedding during search"""
        result = self._embed_texts([text], task_type="retrieval_query")
        return result[0] if result else [0.0] * 768
    
    def _embed_texts_batch(self, texts, task_type="retrieval_query", batch_size=10):
        """Process texts in smaller batches"""
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            print(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")
            
            batch_embeddings = self._embed_texts(batch, task_type)
            all_embeddings.extend(batch_embeddings)
            
            if i + batch_size < len(texts):
                time.sleep(1)
        
        print(f"✅ Successfully embedded {len(all_embeddings)} texts")
        return all_embeddings
    
    def _embed_texts(self, texts, task_type="retrieval_query"):
        """Embed texts with retry logic"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        
        for idx, text in enumerate(texts):
            success = False
            
            for attempt in range(self.max_retries):
                try:
                    truncated_text = text[:20000] if len(text) > 20000 else text
                    
                    print(f"  📝 Embedding text {idx+1}/{len(texts)} (attempt {attempt+1})")
                    
                    res = genai.embed_content(
                        model="models/embedding-001",
                        content=truncated_text,
                        task_type=task_type
                    )
                    
                    embeddings.append(res["embedding"])
                    success = True
                    print(f"    ✅ Success!")
                    break
                    
                except Exception as e:
                    print(f"    ❌ Attempt {attempt+1} failed: {str(e)[:100]}...")
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt)
                        print(f"    ⏳ Retrying in {delay} seconds...")
                        time.sleep(delay)
            
            if not success:
                print(f"⚠️ Using fallback embedding for failed text {idx+1}")
                embeddings.append([0.0] * 768)
        
        return embeddings

def get_gemini_embeddings(api_key):
    """Factory function to create Gemini embedding instance"""
    return GeminiEmbedding(api_key, max_retries=3, timeout=120)
