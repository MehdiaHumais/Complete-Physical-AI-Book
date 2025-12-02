from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI # We use OpenAI client but connect to Local Qwen

app = FastAPI()

# --- CONFIGURATION ---
# Connect to local Qdrant
qdrant = QdrantClient(path="qdrant_db")
encoder = SentenceTransformer('all-MiniLM-L6-v2')
COLLECTION_NAME = "textbook"

# Connect to Local Qwen (Ollama)
client = OpenAI(
    base_url="http://localhost:11434/v1", # Pointing to Ollama
    api_key="ollama" # Required but unused
)

class ChatRequest(BaseModel):
    question: str

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    print(f"User asked: {req.question}")
    
    # 1. Search the Book (RAG)
    query_vector = encoder.encode(req.question).tolist()
    search_results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=3 # Get top 3 most relevant paragraphs
    )
    
    # 2. Build Context
    context_text = "\n\n".join([hit.payload['text'] for hit in search_results])
    
    # 3. Generate Answer with Qwen
    system_prompt = f"""You are an expert Professor in Physical AI. 
    Use the following Textbook Context to answer the student's question. 
    If the answer is not in the context, say 'I cannot find that in the textbook'.
    
    CONTEXT:
    {context_text}
    """
    
    response = client.chat.completions.create(
        model="qwen2.5-coder", # Ensure you have this pulled in Ollama
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req.question}
        ]
    )
    
    return {"reply": response.choices[0].message.content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)