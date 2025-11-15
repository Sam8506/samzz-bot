from sentence_transformers import SentenceTransformer
import chromadb
import os
import re
from typing import List, Dict

DATA_DIR = "data"
CHROMA_DB = "chroma_db"

def smart_chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Smart chunking that respects sentence and paragraph boundaries
    """
    # Split by double newlines (paragraphs) first
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # If adding this paragraph exceeds chunk size
        if len(current_chunk) + len(para) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Add overlap from end of previous chunk
                words = current_chunk.split()
                overlap_text = ' '.join(words[-20:]) if len(words) > 20 else current_chunk
                current_chunk = overlap_text + '\n\n' + para
            else:
                # Paragraph itself is too long, split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                temp_chunk = ""
                
                for sentence in sentences:
                    if len(temp_chunk) + len(sentence) > chunk_size:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                            temp_chunk = sentence
                        else:
                            chunks.append(sentence.strip())
                            temp_chunk = ""
                    else:
                        temp_chunk += ' ' + sentence if temp_chunk else sentence
                
                if temp_chunk:
                    current_chunk = temp_chunk
        else:
            current_chunk += '\n\n' + para if current_chunk else para
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def extract_metadata(filename: str, chunk: str) -> Dict:
    """
    Extract metadata from filename and chunk content
    """
    metadata = {
        "source": filename,
        "file_type": filename.replace('.txt', '')
    }
    
    # Detect section type based on content
    chunk_lower = chunk.lower()
    
    if 'education' in chunk_lower or 'bachelor' in chunk_lower or 'university' in chunk_lower:
        metadata['category'] = 'education'
    elif 'project' in chunk_lower or 'built' in chunk_lower or 'developed' in chunk_lower:
        metadata['category'] = 'projects'
    elif 'skill' in chunk_lower or 'technology' in chunk_lower or 'experience with' in chunk_lower:
        metadata['category'] = 'skills'
    elif 'position:' in chunk_lower or 'duration:' in chunk_lower or 'role:' in chunk_lower:
        metadata['category'] = 'experience'
    elif 'professional summary' in chunk_lower or 'biography' in chunk_lower:
        metadata['category'] = 'about'
    else:
        metadata['category'] = 'general'
    
    # Extract company names if present
    companies = ['talent systems', 'casting networks', 'agevole', 'kintu designs', 'charusat']
    for company in companies:
        if company in chunk_lower:
            metadata['company'] = company.title()
            break
    
    # Extract technologies mentioned
    technologies = ['react', 'nextjs', 'typescript', 'javascript', 'node.js', 'nestjs', 
                   'postgresql', 'mongodb', 'aws', 'docker', 'swift', 'flutter']
    found_techs = [tech for tech in technologies if tech in chunk_lower]
    if found_techs:
        metadata['technologies'] = ', '.join(found_techs[:5])  # Limit to 5
    
    return metadata

def deduplicate_chunks(chunks: List[str]) -> List[str]:
    """
    Remove duplicate or highly similar chunks
    """
    unique_chunks = []
    seen_starts = set()
    
    for chunk in chunks:
        # Use first 100 chars as identifier
        chunk_id = chunk[:100].strip().lower()
        if chunk_id not in seen_starts and len(chunk.strip()) > 50:  # Minimum chunk size
            seen_starts.add(chunk_id)
            unique_chunks.append(chunk)
    
    return unique_chunks

# Init Chroma
client = chromadb.PersistentClient(path=CHROMA_DB)

# Reset old collection
try:
    client.delete_collection("portfolio_rag")
    print("🗑️  Deleted old collection")
except:
    pass

collection = client.create_collection(
    name="portfolio_rag",
    metadata={"description": "Samir Patel's portfolio information"}
)

model = SentenceTransformer("all-MiniLM-L6-v2")

# Process files in priority order
file_priority = [
    "bio.txt",          # Most comprehensive
    "resume.txt",       # Detailed experience
    "tech_stack.txt",   # Skills details
    "projects.txt"      # Project details
]

all_chunks = []
all_metadatas = []
all_ids = []

print("\n📚 Processing portfolio files...\n")

# Process prioritized files first
for filename in file_priority:
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"📄 Processing: {filename}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Smart chunking with overlap
        chunks = smart_chunk_text(text, chunk_size=800, overlap=100)
        chunks = deduplicate_chunks(chunks)
        
        print(f"   ✓ Created {len(chunks)} smart chunks")
        
        for i, chunk in enumerate(chunks):
            metadata = extract_metadata(filename, chunk)
            all_chunks.append(chunk)
            all_metadatas.append(metadata)
            all_ids.append(f"{filename.replace('.txt', '')}_chunk_{i}")

# Process any remaining .txt files
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".txt") and filename not in file_priority:
        filepath = os.path.join(DATA_DIR, filename)
        print(f"📄 Processing: {filename}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        
        chunks = smart_chunk_text(text, chunk_size=800, overlap=100)
        chunks = deduplicate_chunks(chunks)
        
        print(f"   ✓ Created {len(chunks)} smart chunks")
        
        for i, chunk in enumerate(chunks):
            metadata = extract_metadata(filename, chunk)
            all_chunks.append(chunk)
            all_metadatas.append(metadata)
            all_ids.append(f"{filename.replace('.txt', '')}_chunk_{i}")

# Generate embeddings in batches for efficiency
print(f"\n🧠 Generating embeddings for {len(all_chunks)} total chunks...")
embeddings = model.encode(all_chunks, show_progress_bar=True)

# Add to ChromaDB
print("\n💾 Adding to vector database...")
collection.add(
    embeddings=embeddings.tolist(),
    documents=all_chunks,
    metadatas=all_metadatas,
    ids=all_ids
)

# Verify
count = collection.count()
print(f"\n✅ Done! Successfully embedded {count} chunks into ChromaDB")

# Show sample metadata distribution
from collections import Counter
categories = [m['category'] for m in all_metadatas]
print(f"\n📊 Chunk distribution by category:")
for category, count in Counter(categories).most_common():
    print(f"   {category}: {count} chunks")