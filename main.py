from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from dotenv import load_dotenv
from typing import List, Optional
import logging
from datetime import date

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
ALLOWED_ORIGINS = [
    "https://samirpatel.in",
    "https://www.samirpatel.in",
    "http://localhost:3000",
]
app = FastAPI(
    title="Samir Patel Portfolio Chatbot",
    description="AI-powered chatbot for portfolio information",
    version="1.0.0"
)

API_KEY = os.environ.get("PORTFOLIO_API_KEY")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Content-Type", "X-API-Key"],
)

async def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="You are not authorized to access this resource."
        )
    
# Global variables for models (loaded once)
embed_model: Optional[SentenceTransformer] = None
chroma_client: Optional[chromadb.ClientAPI] = None
collection: Optional[chromadb.Collection] = None
groq_client: Optional[Groq] = None

@app.on_event("startup")
async def startup_event():
    """Initialize models and database connections on startup"""
    global embed_model, chroma_client, collection, groq_client
    
    try:
        logger.info("Loading embedding model...")
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        logger.info("Connecting to ChromaDB...")
        chroma_client = chromadb.PersistentClient(path="chroma_db")
        collection = chroma_client.get_collection("portfolio_rag")
        
        logger.info("Initializing Groq client...")
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        groq_client = Groq(api_key=api_key)
        
        logger.info("✅ All services initialized successfully")
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down services...")

# Request/Response models
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="User's question about Samir's portfolio")

class ChatResponse(BaseModel):
    answer: str
    confidence: str

class HealthResponse(BaseModel):
    status: str
    services: dict

today = date.today()
# Enhanced system prompt with date context
SYSTEM_PROMPT = """You are Samir Patel's AI assistant, representing him professionally and accurately.

## Current Date Context:
- Today's date: {today}
- All experience and dates should be calculated relative to {today}
- Professional experience started: July 2023

## Your Role:
- Answer questions about Samir's experience, skills, projects, and background
- Provide accurate information based on the context provided
- Be conversational, professional, and confident
- Speak naturally as Samir would speak about his own experience

## Critical Guidelines:
- Use first person ("I have...", "I worked on...", "My experience includes...")
- Be direct and confident - don't add disclaimers like "based on the context" or "to the best of my knowledge"
- If you have the information in the context, state it clearly without hedging
- Only mention missing information if the user asks something truly not covered
- Be specific with numbers, dates, technologies, and achievements when available
- Keep responses concise and well-organized (use bullet points for lists)

## Date Calculations:
- When mentioning "years of experience", calculate from July 2023 to {today} and state the exact duration with months if applicable and dont' say opproximate mention over the exact time
- For specific roles, use the exact dates provided in context
- Current role: Software Engineer at Talent Systems (promoted November 2025)
- Always frame experience relative to {today}

## Response Style:
- Professional yet personable
- Direct and confident about stated facts
- No unnecessary qualifiers or disclaimers
- Natural conversational flow

## What NOT to do:
- ❌ Don't say "based on the context provided"
- ❌ Don't say "to the best of my knowledge" 
- ❌ Don't say "the context doesn't mention"
- ❌ Don't add uncertain language when you have the information
- ❌ Don't say "over XYZ years" when it's actually XYZ years
- ✅ Just answer confidently with the information you have

Alternative friendly responses for off-topic questions:
- "I appreciate the question! While I don't have details on that, I'd love to share about my software engineering journey, the technologies I work with, or the projects I've built."
- "Great question! That's outside my portfolio info, but I'm excited to discuss my experience in full-stack development, my work at Talent Systems, or any of my side projects. What would you like to explore?"
"""


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Samir Patel Portfolio Chatbot API",
        "docs": "/docs",
        "health": "/health",
        "chat": "/chat"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health"""
    services = {
        "embedding_model": embed_model is not None,
        "chromadb": collection is not None,
        "groq_client": groq_client is not None
    }
    
    all_healthy = all(services.values())
    
    return {
        "status": "healthy" if all_healthy else "unhealthy",
        "services": services
    }

@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def chat(req: ChatRequest):
    """
    Chat with Samir's portfolio assistant
    
    - **question**: Your question about Samir's experience, skills, projects, or background
    
    Returns an AI-generated response based on Samir's actual portfolio data.
    """
    try:
        # Validate services are initialized
        if not all([embed_model, collection, groq_client]):
            raise HTTPException(status_code=503, detail="Services not initialized")
        
        question = req.question.strip()
        logger.info(f"Processing query: {question[:100]}...")
        
        # Step 1: Generate embedding for the question
        try:
            q_embed = embed_model.encode([question]).tolist()[0]
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to process question")
        
        # Step 2: Retrieve relevant context from ChromaDB
        try:
            results = collection.query(
                query_embeddings=[q_embed],
                n_results=5
            )
            
            if not results["documents"] or not results["documents"][0]:
                return ChatResponse(
                    answer="I don't have information about that in my portfolio data. Could you ask something else about my experience, skills, or projects?",
                    confidence="low"
                )
            
            chunks = results["documents"][0]
            distances = results.get("distances", [[]])[0]
            
            # Build context
            context = "\n\n".join(chunks)
            
            # Determine confidence based on similarity scores
            avg_distance = sum(distances) / len(distances) if distances else 1.0
            confidence = "high" if avg_distance < 0.5 else "medium" if avg_distance < 0.8 else "low"
            
        except Exception as e:
            logger.error(f"ChromaDB query failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to retrieve context")
        
        # Step 3: Generate response with Groq
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ]
            
            groq_response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                max_tokens=500,
                temperature=0.7,
                top_p=0.9,
            )
            
            answer = groq_response.choices[0].message.content
            
            logger.info(f"✅ Query processed successfully. Confidence: {confidence}")
            
            return ChatResponse(
                answer=answer,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Groq API call failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)