"""
RAG Core Engine
"""
from typing import List, Dict
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


class RAGEngine:
    """RAG Engine: Combining vector retrieval and LLM generation"""
    
    def __init__(self, vector_store, model: str = "gpt-3.5-turbo"):
        """
        Initialize RAG engine
        
        Args:
            vector_store: VectorStore instance
            model: OpenAI model name
        """
        self.vector_store = vector_store
        self.model = model
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY in .env file")
        self.client = OpenAI(api_key=api_key)
    
    def query(self, question: str, top_k: int = 3, temperature: float = 0.7) -> Dict[str, any]:
        """
        Execute RAG query
        
        Args:
            question: User question
            top_k: Number of document chunks to retrieve
            temperature: LLM temperature parameter
            
        Returns:
            dict: Contains answer, retrieval results, etc.
        """
        # 1. Retrieve relevant documents from vector store
        retrieved_docs = self.vector_store.search(question, top_k=top_k)
        
        # 2. Build context
        context = self._build_context(retrieved_docs)
        
        # 3. Build prompt
        prompt = self._build_prompt(question, context)
        
        # 4. Call LLM to generate answer
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an assistant that answers questions based on document content. Please answer questions based on the provided document content. If there is no relevant information in the documents, please state so."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
        
        return {
            'answer': answer,
            'retrieved_docs': retrieved_docs,
            'context': context
        }
    
    def _build_context(self, retrieved_docs: List[Dict]) -> str:
        """Build context text"""
        if not retrieved_docs:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"[Document Chunk {i}]\n{doc['text']}\n")
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Build prompt"""
        prompt = f"""
        Please answer the question based on the following document content.
        Document Content: {context}
        Question: {question}
        Please answer the question based on the document content above. If there is no relevant information in the documents, please state that the answer cannot be found in the documents.
        """
        return prompt
