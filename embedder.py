from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional
import os

# Global model to avoid reloading
_model = None

def get_embedding_model():
    """Get or load the embedding model"""
    global _model
    if _model is None:
        print("🔄 Loading embedding model...")
        _model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded")
    return _model

def create_faiss_index(docs: List[Dict]) -> Tuple[Optional[faiss.Index], List[Dict], List[str]]:
    """
    Create FAISS index from documents
    
    Args:
        docs: List of documents with 'title', 'abstract', 'url' keys
        
    Returns:
        Tuple of (FAISS index, metadata list, text list)
    """
    if not docs:
        print("❌ No documents provided for indexing")
        return None, [], []
    
    print(f"📊 Creating FAISS index for {len(docs)} documents...")
    
    # Extract texts and metadata
    texts = []
    metadatas = []
    
    for doc in docs:
        abstract = doc.get('abstract', '').strip()
        title = doc.get('title', 'No title').strip()
        
        # Use both title and abstract for better context
        if abstract and abstract != "No abstract available":
            text = f"{title}. {abstract}"
        else:
            text = title
            
        texts.append(text)
        metadatas.append({
            "title": title,
            "url": doc.get('url', ''),
            "pmid": doc.get('pmid', '')
        })
    
    if not texts:
        print("❌ No valid texts found for embedding")
        return None, [], []
    
    try:
        # Get embedding model
        model = get_embedding_model()
        
        # Create embeddings
        print("🔄 Generating embeddings...")
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
        
        if len(embeddings) == 0:
            print("❌ No embeddings generated")
            return None, [], []
        
        # Create FAISS index
        print("🔄 Building FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        
        # Normalize embeddings for better similarity search
        faiss.normalize_L2(embeddings.astype(np.float32))
        index.add(embeddings.astype(np.float32))
        
        print(f"✅ FAISS index created with {index.ntotal} documents")
        return index, metadatas, texts
        
    except Exception as e:
        print(f"❌ Error creating FAISS index: {e}")
        return None, [], []

def search_similar_documents(index: faiss.Index, metadatas: List[Dict], texts: List[str], 
                           query: str, k: int = 3) -> List[Dict]:
    """
    Search for similar documents using the FAISS index
    
    Args:
        index: FAISS index
        metadatas: List of document metadata
        texts: List of document texts
        query: Search query
        k: Number of results to return
        
    Returns:
        List of similar documents with metadata and similarity scores
    """
    if index is None or not metadatas:
        return []
    
    try:
        # Get embedding model and encode query
        model = get_embedding_model()
        query_embedding = model.encode([query])
        
        # Normalize query embedding
        faiss.normalize_L2(query_embedding.astype(np.float32))
        
        # Search
        distances, indices = index.search(query_embedding.astype(np.float32), k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(metadatas):  # Valid index
                result = {
                    'rank': i + 1,
                    'similarity_score': float(1 / (1 + distance)),  # Convert distance to similarity
                    'text': texts[idx],
                    'metadata': metadatas[idx]
                }
                results.append(result)
        
        return results
        
    except Exception as e:
        print(f"❌ Error during similarity search: {e}")
        return []