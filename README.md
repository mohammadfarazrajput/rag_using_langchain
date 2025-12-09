# RAG Application with LangChain

A Retrieval-Augmented Generation (RAG) application built with LangChain that allows you to query PDF and text documents using natural language.

## Features

- 📄 Support for PDF and text file processing
- 🔍 Semantic search using HuggingFace embeddings
- 💾 Vector storage with ChromaDB
- 🤖 Powered by Google Gemini AI
- 🎯 MMR (Maximal Marginal Relevance) retrieval for diverse results
- 📝 Academic and professional response tone

## Prerequisites

- Python 3.8+
- Google API key for Gemini

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd langchain
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install langchain langchain-community langchain-core langchain-chroma
pip install langchain-huggingface langchain-text-splitters langchain-google-genai
pip install python-dotenv pypdf chromadb
```

4. Create a `.env` file in the root directory:
```
GOOGLE_API_KEY=your_google_api_key_here
```

## Project Structure
```
langchain/
├── RAG.py              # Main application file
├── unit.txt            # Sample text file
├── unit4.pdf           # Sample PDF file
├── .env                # Environment variables (not tracked)
├── rag_chroma_db/      # Vector database storage (not tracked)
└── README.md           # This file
```

## Usage

1. Place your documents in the project directory:
   - Text files: `.txt` format
   - PDF files: `.pdf` format

2. Run the application:
```bash
python RAG.py
```

3. Enter your query when prompted
4. The system will retrieve relevant context and provide an answer

## How It Works

1. **Document Loading**: Loads text and PDF files using appropriate loaders
2. **Text Splitting**: Chunks documents into smaller pieces (750 chars with 80 char overlap)
3. **Embedding**: Converts text chunks to vectors using `all-MiniLM-L6-v2` model
4. **Storage**: Stores embeddings in ChromaDB for efficient retrieval
5. **Retrieval**: Uses MMR algorithm to find relevant chunks
6. **Generation**: Gemini AI generates answers based only on retrieved context

## Configuration

### Text Splitter Settings
```python
chunk_size = 750        # Size of each text chunk
chunk_overlap = 80      # Overlap between chunks
```

### Retriever Settings
```python
search_type = "mmr"     # Maximal Marginal Relevance
k = 2                   # Number of documents to retrieve
fetch_k = 5             # Number of documents to fetch before MMR
lambda_mult = 0.5       # Diversity parameter (0=diverse, 1=similar)
```

### Model Settings
```python
model = "gemini-2.5-flash-lite"
temperature = 1
```

## Important Notes

- The application only uses information from provided documents (no external knowledge)
- First run creates the vector database which is persisted locally
- Subsequent runs will add documents again - see roadmap for improvements

## Roadmap

- [ ] Add Streamlit UI for better user experience
- [ ] Implement file upload functionality
- [ ] Add support for multiple file formats (.docx, .csv, etc.)
- [ ] Prevent duplicate document ingestion
- [ ] Add chat history
- [ ] Show source citations
- [ ] Multi-document support

## Technologies Used

- **LangChain**: Framework for LLM applications
- **ChromaDB**: Vector database
- **HuggingFace**: Embedding models
- **Google Gemini**: Language model
- **PyPDF**: PDF processing

## Contributing

Feel free to open issues or submit pull requests!

## License

MIT

## Author

Built while learning LangChain and RAG concepts