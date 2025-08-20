"""
Dubai FAQ Data Loader Script
============================
This script loads FAQs from CSV and stores them in Pinecone.
Run this ONCE during initial setup or when updating FAQ data.

Usage:
    python load_faqs.py

Environment Variables Required:
    PINECONE_API_KEY=your_pinecone_api_key
    INDEX_NAME=dubai-faq-index (or your preferred name)
"""

import os
import pandas as pd
import time
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "dubai-faq-index")  # Default name if not set
CSV_FILE_PATH = "dubai_faqs.csv"  # Path to your CSV file
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2 model

print("=== Dubai FAQ Data Loader ===")
print(f"Index Name: {INDEX_NAME}")
print(f"CSV File: {CSV_FILE_PATH}")

def initialize_pinecone():
    """Initialize Pinecone client"""
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in environment variables!")
    
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        print("✅ Pinecone client initialized successfully")
        return pc
    except Exception as e:
        raise Exception(f"Failed to initialize Pinecone: {e}")

def create_or_get_index(pc):
    """Create index if it doesn't exist, or get existing index"""
    try:
        # List existing indexes
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if INDEX_NAME in existing_indexes:
            print(f"📋 Index '{INDEX_NAME}' already exists")
            
            # Ask user if they want to delete and recreate
            response = input("Do you want to delete and recreate the index? (y/N): ").lower()
            if response == 'y':
                print(f"🗑️ Deleting existing index '{INDEX_NAME}'...")
                pc.delete_index(INDEX_NAME)
                
                # Wait for deletion to complete
                print("⏳ Waiting for index deletion...")
                time.sleep(10)
            else:
                print("⚠️ Using existing index. Data will be upserted (updated/inserted)")
        
        # Create index if it doesn't exist or was deleted
        existing_indexes = [index.name for index in pc.list_indexes()]
        if INDEX_NAME not in existing_indexes:
            print(f"🚀 Creating new index '{INDEX_NAME}'...")
            pc.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'  # Change if needed
                )
            )
            
            # Wait for index to be ready
            print("⏳ Waiting for index to be ready...")
            time.sleep(30)  # Give it time to initialize
        
        # Connect to the index
        index = pc.Index(INDEX_NAME)
        print(f"✅ Connected to index '{INDEX_NAME}'")
        return index
        
    except Exception as e:
        raise Exception(f"Failed to create/get index: {e}")

def load_csv_data():
    """Load FAQ data from CSV file"""
    try:
        if not os.path.exists(CSV_FILE_PATH):
            raise FileNotFoundError(f"CSV file not found: {CSV_FILE_PATH}")
        
        df = pd.read_csv(CSV_FILE_PATH)
        
        # Validate required columns
        required_columns = ['question', 'answer']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV")
        
        # Remove any empty rows
        df = df.dropna(subset=['question', 'answer'])
        
        print(f"✅ Loaded {len(df)} FAQs from CSV")
        return df
        
    except Exception as e:
        raise Exception(f"Failed to load CSV data: {e}")

def generate_embeddings(questions):
    """Generate embeddings for questions"""
    try:
        print("🧠 Loading embedding model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("🔄 Generating embeddings...")
        embeddings = model.encode(questions, show_progress_bar=True)
        
        print(f"✅ Generated {len(embeddings)} embeddings")
        return embeddings
        
    except Exception as e:
        raise Exception(f"Failed to generate embeddings: {e}")

def store_in_pinecone(index, df, embeddings):
    """Store FAQs and embeddings in Pinecone"""
    try:
        vectors = []
        
        print("📦 Preparing vectors for Pinecone...")
        for i, (_, row) in enumerate(df.iterrows()):
            vector = {
                'id': f'faq_{i}',
                'values': embeddings[i].tolist(),
                'metadata': {
                    'question': str(row['question']),
                    'answer': str(row['answer'])
                }
            }
            vectors.append(vector)
        
        # Upsert in batches
        batch_size = 100
        total_batches = (len(vectors) + batch_size - 1) // batch_size
        
        print(f"⬆️ Uploading {len(vectors)} vectors in {total_batches} batches...")
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            print(f"📤 Uploading batch {batch_num}/{total_batches} ({len(batch)} vectors)")
            index.upsert(vectors=batch)
            
            # Small delay between batches
            time.sleep(1)
        
        print("✅ All vectors uploaded successfully!")
        
        # Wait a moment and get index stats
        print("📊 Getting index statistics...")
        time.sleep(5)
        stats = index.describe_index_stats()
        print(f"Index Stats: {stats['total_vector_count']} vectors, {stats['dimension']} dimensions")
        
    except Exception as e:
        raise Exception(f"Failed to store data in Pinecone: {e}")

def verify_data(index):
    """Verify that data was stored correctly by doing a test query"""
    try:
        print("🔍 Verifying data with test query...")
        
        # Generate test query embedding
        model = SentenceTransformer('all-MiniLM-L6-v2')
        test_query = "What time is best to visit Dubai?"
        query_embedding = model.encode([test_query])
        
        # Search index
        results = index.query(
            vector=query_embedding[0].tolist(),
            top_k=3,
            include_metadata=True
        )
        
        if results.matches and len(results.matches) > 0:
            print("✅ Data verification successful!")
            print("Top 3 matches:")
            for i, match in enumerate(results.matches[:3], 1):
                print(f"  {i}. Score: {match.score:.3f}")
                print(f"     Q: {match.metadata['question'][:100]}...")
                print(f"     A: {match.metadata['answer'][:100]}...")
                print()
        else:
            print("⚠️ No matches found in verification test")
            
    except Exception as e:
        print(f"⚠️ Verification failed: {e}")

def main():
    """Main execution function"""
    try:
        print("Starting FAQ data loading process...\n")
        
        # Step 1: Initialize Pinecone
        pc = initialize_pinecone()
        
        # Step 2: Create or get index
        index = create_or_get_index(pc)
        
        # Step 3: Load CSV data
        df = load_csv_data()
        
        # Step 4: Generate embeddings
        embeddings = generate_embeddings(df['question'].tolist())
        
        # Step 5: Store in Pinecone
        store_in_pinecone(index, df, embeddings)
        
        # Step 6: Verify data
        verify_data(index)
        
        print("\n🎉 FAQ data loading completed successfully!")
        print(f"Index Name: {INDEX_NAME}")
        print(f"Total FAQs: {len(df)}")
        print("\nYou can now run your main application!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease check your configuration and try again.")
        return False
    
    return True

if __name__ == "__main__":
    # Run the main function
    success = main()
    
    if not success:
        print("\n🔧 Troubleshooting Tips:")
        print("1. Check your PINECONE_API_KEY in .env file")
        print("2. Ensure dubai_faqs.csv exists in the same directory")
        print("3. Verify your Pinecone account has available quota")
        print("4. Check internet connection")
        
    input("\nPress Enter to exit...")