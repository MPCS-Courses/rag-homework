"""
Document Loader
"""
import os
from typing import List, Dict
from pathlib import Path
import docx


class DocumentLoader:
    """Document loader supporting multiple formats"""
    
    def __init__(self):
        self.supported_extensions = {'.txt', '.md', '.docx'}
    
    def load_document(self, file_path: str) -> Dict[str, str]:
        """
        Load a single document
        
        Args:
            file_path: Document path
            
        Returns:
            dict: {'content': document content, 'filename': filename, 'extension': extension}
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file format: {extension}")
        
        if extension == '.txt':
            content = self._load_txt(file_path)
        elif extension == '.md':
            content = self._load_txt(file_path)  # Markdown is also a text file
        elif extension == '.docx':
            content = self._load_docx(file_path)
        else:
            raise ValueError(f"Unimplemented file format: {extension}")
        
        return {
            'content': content,
            'filename': path.name,
            'extension': extension,
            'file_path': file_path
        }
    
    def load_directory(self, directory: str) -> List[Dict[str, str]]:
        """
        Load all supported format documents from directory
        
        Args:
            directory: Directory path
            
        Returns:
            list: Document list
        """
        documents = []
        directory_path = Path(directory)
        
        for file_path in directory_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    doc = self.load_document(str(file_path))
                    documents.append(doc)
                except Exception as e:
                    print(f"Failed to load document {file_path}: {e}")
        
        return documents
    
    def _load_txt(self, file_path: str) -> str:
        """Load text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _load_docx(self, file_path: str) -> str:
        """Load Word document"""
        doc = docx.Document(file_path)
        paragraphs = [para.text for para in doc.paragraphs]
        return '\n'.join(paragraphs)
    
    def chunk_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, any]]:
        """
        Chunk text into pieces
        
        Args:
            text: Original text
            chunk_size: Size of each chunk (number of characters)
            chunk_overlap: Number of overlapping characters between chunks
            
        Returns:
            list: List of text chunks, each containing {'text': text content, 'start': start position, 'end': end position}
        """
        if len(text) <= chunk_size:
            return [{'text': text, 'start': 0, 'end': len(text)}]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at punctuation marks, newlines, etc.
            if end < len(text):
                # Search backwards for a suitable break point
                for i in range(end, max(start, end - 100), -1):
                    if text[i] in ['\n', '。', '.', '!', '?']:
                        end = i + 1
                        break
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'start': start,
                    'end': end
                })
            
            # Next chunk start position (considering overlap)
            start = end - chunk_overlap if end < len(text) else end
        
        return chunks
