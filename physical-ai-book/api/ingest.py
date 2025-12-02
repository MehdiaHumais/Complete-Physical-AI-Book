import os
import glob
import uuid
import time
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import google.generativeai as genai

# --- CONFIGURATION ---
# 🔴 PASTE YOUR NEW API KEY HERE 🔴
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE" 
genai.configure(api_key=GEMINI_API_KEY)

qdrant = QdrantClient(path="qdrant_db")
COLLECTION_NAME = "textbook"

# 1. Setup Database
# Gemini text-embedding-004 uses 768 dimensions
qdrant.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)

print("🔍 Scanning docs folder...")
files = glob.glob("docs/**/*.md", recursive=True)
points = []

# 2. Process Files
for file_path in files:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split by double newlines to get paragraphs
    chunks = content.split("\n\n")
    
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 30: continue
        
        # RETRY LOOP for Rate Limits
        while True:
            try:
                # 🛑 SAFETY PAUSE: Wait 4s to stay under 15 RPM limit
                time.sleep(4)
                
                print(f"Processing {os.path.basename(file_path)} chunk {i+1}/{len(chunks)}...")
                
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=chunk,
                    task_type="retrieval_document"
                )
                
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=result['embedding'],
                    payload={"text": chunk, "source": file_path}
                ))
                break # Success, move to next chunk
                
            except Exception as e:
                if "429" in str(e):
                    print(f"   ⏳ Rate Limit Hit! Pausing for 60 seconds...")
                    time.sleep(60)
                else:
                    print(f"   ❌ Error: {e}")
                    break

# 3. Save to Database
if points:
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"🎉 Success! Uploaded {len(points)} chunks to the database.")