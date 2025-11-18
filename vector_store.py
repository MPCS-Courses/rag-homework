"""
FAISS Vector Store Management
"""
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional


class VectorStore:
    """FAISS-based vector store"""
    
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2', dimension: int = 384):
        """
        Initialize vector store
        
        Args:
            embedding_model_name: sentence-transformers model name
            dimension: Vector dimension (automatically obtained from model)
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.model_name = embedding_model_name
        # Get actual vector dimension
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index (using L2 distance)
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Store metadata (document chunk information)
        self.metadata = []
        
    def add_documents(self, chunks: List[Dict[str, any]], source_info: Optional[Dict] = None):
        """
        Add document chunks to vector store
        
        Args:
            chunks: List of document chunks, each containing 'text' field
            source_info: Source document information (e.g., filename)
        """
        if not chunks:
            return
        
        # Extract text
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings).astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Save metadata
        for i, chunk in enumerate(chunks):
            metadata = {
                'text': chunk['text'],
                'chunk_index': len(self.metadata),
                'source': source_info or {}
            }
            # If start and end information exists, save it too
            if 'start' in chunk:
                metadata['start'] = chunk['start']
            if 'end' in chunk:
                metadata['end'] = chunk['end']
            
            self.metadata.append(metadata)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, any]]:
        """
        Search for similar document chunks
        
        Args:
            query: Query text
            top_k: Return top k most similar results
            
        Returns:
            list: Search results list, each containing 'text', 'score', 'metadata'
        """
        if self.index.ntotal == 0:
            return []
        
        # Convert query text to vector
        query_embedding = self.embedding_model.encode([query], show_progress_bar=False)
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Build results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                result = {
                    'text': self.metadata[idx]['text'],
                    'score': float(distance),  # L2 distance, smaller is more similar
                    'similarity': 1 / (1 + distance),  # Convert to similarity score (0-1)
                    'metadata': self.metadata[idx].get('source', {}),
                    'chunk_index': self.metadata[idx].get('chunk_index', idx)
                }
                results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, any]:
        """Get vector store statistics"""
        return {
            'total_chunks': self.index.ntotal,
            'dimension': self.dimension,
            'model': self.model_name
        }
    
    def clear(self):
        """Clear vector store"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
    
    def save(self, index_path: str, metadata_path: str):
        """Save vector index and metadata to files"""
        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def load(self, index_path: str, metadata_path: str):
        """Load vector index and metadata from files"""
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
