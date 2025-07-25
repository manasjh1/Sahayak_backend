import os
import sys
import math
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from src.helper import load_pdf_file, text_split, get_gemini_embeddings

load_dotenv()

def calculate_optimal_batch_size(text_chunks, target_size_mb=1.5):
    """
    Calculate optimal batch size to stay under Pinecone's 2MB limit
    """
    if not text_chunks:
        return 25
    
    # Estimate average text size from a sample
    sample_size = min(10, len(text_chunks))
    avg_text_size = sum(len(chunk.page_content.encode('utf-8')) for chunk in text_chunks[:sample_size]) / sample_size
    
    # Account for embedding vector size (768 * 4 bytes for float32) + metadata + JSON overhead
    embedding_size = 768 * 4  # 3KB per embedding
    metadata_overhead = 300   # Estimated JSON/metadata overhead per record
    
    estimated_size_per_doc = avg_text_size + embedding_size + metadata_overhead
    
    # Calculate batch size with safety margin
    target_size_bytes = target_size_mb * 1024 * 1024
    batch_size = max(10, min(50, int(target_size_bytes / estimated_size_per_doc)))
    
    print(f"📊 Estimated size per document: {estimated_size_per_doc/1024:.1f}KB")
    print(f"📦 Calculated optimal batch size: {batch_size}")
    
    return batch_size

def upload_documents_in_batches(documents, embeddings, index_name, batch_size=25):
    """
    Upload documents to Pinecone in manageable batches to avoid 2MB limit
    """
    total_docs = len(documents)
    total_batches = math.ceil(total_docs / batch_size)
    
    print(f"🚀 Uploading {total_docs} documents in {total_batches} batches of ~{batch_size}")
    
    # Initialize the vector store
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name, 
        embedding=embeddings
    )
    
    successful_uploads = 0
    failed_uploads = 0
    
    for i in range(0, total_docs, batch_size):
        batch_num = (i // batch_size) + 1
        end_idx = min(i + batch_size, total_docs)
        batch_docs = documents[i:end_idx]
        
        print(f"📤 Uploading batch {batch_num}/{total_batches} ({len(batch_docs)} documents)")
        
        try:
            # Upload this batch
            vectorstore.add_documents(batch_docs)
            successful_uploads += len(batch_docs)
            print(f"    ✅ Batch {batch_num} uploaded successfully")
            
            # Small delay to avoid rate limiting
            if batch_num < total_batches:
                import time
                time.sleep(0.5)
                
        except Exception as e:
            error_msg = str(e)
            print(f"    ❌ Batch {batch_num} failed: {error_msg[:100]}...")
            
            # Try with smaller sub-batches if the batch is large enough
            if len(batch_docs) > 5:
                smaller_batch_size = max(3, len(batch_docs) // 3)
                print(f"    🔄 Retrying with smaller sub-batches of {smaller_batch_size}")
                
                for j in range(0, len(batch_docs), smaller_batch_size):
                    sub_batch = batch_docs[j:j + smaller_batch_size]
                    try:
                        vectorstore.add_documents(sub_batch)
                        successful_uploads += len(sub_batch)
                        print(f"        ✅ Sub-batch uploaded ({len(sub_batch)} docs)")
                        import time
                        time.sleep(0.3)
                    except Exception as sub_e:
                        failed_uploads += len(sub_batch)
                        print(f"        ❌ Sub-batch failed: {str(sub_e)[:60]}...")
            else:
                failed_uploads += len(batch_docs)
                print(f"    ⚠️ Skipping batch {batch_num} - cannot reduce size further")
    
    return successful_uploads, failed_uploads

def main():
    # Load environment variables
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    
    if not PINECONE_API_KEY or not GEMINI_API_KEY:
        print("❌ Missing API keys in .env file")
        print("Required: PINECONE_API_KEY, GEMINI_API_KEY")
        return
    
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
    
    print("🚀 Starting optimized data ingestion process...")
    
    try:
        # Load and process documents
        print("📚 Loading documents from Data/ directory...")
        extracted_data = load_pdf_file(data_path='Data/')
        
        if not extracted_data:
            print("❌ No documents found in Data/ directory")
            print("Make sure you have PDF or TXT files in the Data/ folder")
            return
        
        print(f"📄 Found {len(extracted_data)} documents")
        
        print("✂️ Splitting text into chunks...")
        text_chunks = text_split(extracted_data)
        print(f"📝 Created {len(text_chunks)} text chunks")
        
        # Initialize embeddings
        print("🤖 Initializing Gemini embeddings...")
        embeddings = get_gemini_embeddings(GEMINI_API_KEY)
        
        # Initialize Pinecone
        print("🌲 Connecting to Pinecone...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index_name = "bot"
        
        # Check if index exists, create if not
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing_indexes:
            print(f"🔨 Creating new index '{index_name}'...")
            pc.create_index(
                name=index_name,
                dimension=768,  # Gemini embedding dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print("⏳ Waiting for index to be ready...")
            import time
            time.sleep(20)  # Wait for index to be ready
        else:
            print(f"✅ Index '{index_name}' already exists")
        
        # Calculate optimal batch size
        optimal_batch_size = calculate_optimal_batch_size(text_chunks)
        
        # Upload documents in batches
        print("🔄 Starting batch upload process...")
        print("⚠️  Note: This avoids the 2MB Pinecone batch limit by uploading in smaller chunks")
        
        successful, failed = upload_documents_in_batches(
            text_chunks, 
            embeddings, 
            index_name, 
            optimal_batch_size
        )
        
        # Summary
        total_docs = len(text_chunks)
        success_rate = (successful / total_docs) * 100 if total_docs > 0 else 0
        
        print("\n" + "="*60)
        print("📊 UPLOAD SUMMARY")
        print("="*60)
        print(f"✅ Successfully uploaded: {successful}/{total_docs} documents ({success_rate:.1f}%)")
        if failed > 0:
            print(f"❌ Failed uploads: {failed} documents")
        
        # Verify final index stats
        try:
            import time
            time.sleep(2)  # Wait a moment for index to update
            stats = pc.Index(index_name).describe_index_stats()
            print(f"📈 Final index statistics: {stats.total_vector_count} vectors stored")
            print("="*60)
            
            if successful > 0:
                print("🎉 SUCCESS! Your documents are now indexed and ready for search!")
            else:
                print("⚠️ No documents were successfully uploaded. Check the errors above.")
                
        except Exception as e:
            print(f"⚠️ Could not retrieve final index stats: {e}")
            if successful > 0:
                print("🎉 Upload completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Critical error during ingestion: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
