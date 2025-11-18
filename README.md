# MPCS 57200 Generative AI Homework 7-1

## Installation

1. git clone the project

2. Install dependencies: ```pip install -r requirements.txt```

3. Create `.env` file and add your OpenAI API key: ```OPENAI_API_KEY=your_api_key_here```

## Usage

1. Place your documents in the `documents/` directory in project root (supported formats: `.txt`, `.md`, `.docx`)

2. Run the application: ```streamlit run app.py```

3. Usage flow:
   - On side bar, click "Load Documents"
   - Adjust chunking parameters (optional)
   - Click "Build Vector Index"
   - Start asking questions in the main interface

## Key Observations and Result Critique

The RAG chatbot demonstrates effective retrieval-augmented generation capabilities, successfully combining vector search with LLM-based answer generation. Through testing with various document types and query patterns, several key observations and limitations have been identified.

### Chunking Strategy Analysis

The chunking strategy proves to be one of the most critical factors affecting system performance. Through testing the same prompts with various chunk sizes, I observed that smaller chunks (200-300 characters) typically produce more granular retrieval with better precision for specific questions. However, this comes at the cost of context loss, as answers can become fragmented when information spans multiple chunks. Slightly larger chunks around 500~1000 characters provide a better balance between context preservation and retrieval precision, working well for most general questions. When chunks exceed 2000 characters, the system preserves more context which benefits complex questions requiring multiple sentences, but this often leads to retrieving irrelevant information along with relevant content, resulting in lower precision.

The chunk overlap parameter also significantly impacts retrieval quality. With no overlap, context loss occurs at chunk boundaries, especially when sentences are split mid-thought. A moderate overlap of 50-100 characters helps maintain context continuity and is recommended for most use cases. While high overlap above 150 characters may cause redundant retrieval, it ensures no information loss at boundaries. The optimal chunking strategy depends heavily on the document structure and question types. For technical documentation with clear sections, smaller chunks with moderate overlap work best. For narrative content or documents with longer contextual dependencies, larger chunks are preferable.

### Embedding Model and Retrieval Quality

The system employs the `all-MiniLM-L6-v2` model with 384 dimensions for generating embeddings. This model offers fast inference time for local execution and provides a good balance between quality and speed for general semantic similarity tasks. However, the model may struggle with domain-specific terminology and has limited ability to capture nuanced semantic relationships. The 384-dimensional representation may not capture all semantic subtleties compared to larger models like `all-mpnet-base-v2` with 768 dimensions, though such models come with increased computational cost.

### System Limitations and Future Improvements

Several system limitations were identified during testing. The system doesn't expand queries with synonyms or related terms, which could improve retrieval for queries using different terminology than the documents. Once built, the index is static and doesn't update automatically when documents change, requiring manual re-indexing. There's no metadata filtering capability, preventing users from filtering by document type, date, or other metadata. Error handling is limited and doesn't gracefully handle edge cases such as empty documents or encoding issues. Finally, each query is independent with no conversation memory, meaning the system cannot maintain context from previous questions in a multi-turn dialogue.

For future improvements, the chunking strategy could be enhanced with semantic chunking that splits at sentence boundaries while preserving paragraphs, and adding metadata to chunks such as document source, position, and section. Retrieval could be improved by implementing re-ranking using cross-encoders, adding query expansion with synonyms and related terms, and considering hybrid search combining keyword and semantic approaches. Answer generation could benefit from improved prompt engineering with few-shot examples, explicit source citation in answers, and better synthesis of information from multiple chunks. System architecture enhancements could include conversation memory for multi-turn dialogues, incremental indexing for document updates, and metadata filtering capabilities.
