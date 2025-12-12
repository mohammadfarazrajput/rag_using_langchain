# RAG Application with LangChain

A Retrieval-Augmented Generation (RAG) application built with LangChain that allows you to query PDF and text documents using natural language.

## Features

- 📄 Support for PDF and text file processing
- 🔍 Semantic search using HuggingFace embeddings
- 💾 Vector storage with ChromaDB
- 🤖 Powered by Google Gemini AI (free tier)
- 🎯 MMR (Maximal Marginal Relevance) retrieval for diverse results
- 📝 Academic and professional response tone
- 🔄 Interactive chat loop with document context

## Why These Models?

This project uses **Google Gemini** (free tier) and **HuggingFace embeddings** instead of OpenAI models because:
- ❌ No access to paid OpenAI API
- ✅ Google Gemini offers a generous free tier
- ✅ HuggingFace embeddings are completely free and open-source
- ✅ Results are comparable for learning and development purposes

## Prerequisites

- Python 3.8+
- Google API key for Gemini (free at [Google AI Studio](https://makersuite.google.com/app/apikey))

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
pip install -r requirements.txt
```

Or manually:
```bash
pip install langchain langchain-community langchain-core langchain-chroma
pip install langchain-huggingface langchain-text-splitters langchain-google-genai
pip install python-dotenv pypdf chromadb
```

4. Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_google_api_key_here
```

## Project Structure
```
langchain/
├── RAG.py              # Main application file
├── prompts.py          # Prompt templates
├── unit.txt            # Sample text file
├── unit4.pdf           # Sample PDF file
├── .env                # Environment variables (not tracked)
├── .gitignore          # Git ignore rules
├── requirements.txt    # Python dependencies
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

3. When prompted:
   - Enter the file path/name of your document
   - Ask questions about the document
   - Type 'Q' to quit

4. Example interaction:
```
Enter the file name: unit4.pdf
Chat with uploaded document: What is Software Configuration Management?
[Answer based on document context]
To quit press 'Q': no
Chat with uploaded document: ...
```

## How It Works

1. **Document Loading**: Detects file type and loads using appropriate loader
2. **Text Splitting**: Chunks documents optimally (800-1200 chars based on file type)
3. **Embedding**: Converts text chunks to vectors using `all-MiniLM-L6-v2` model
4. **Storage**: Stores embeddings in ChromaDB with persistence
5. **Retrieval**: Uses MMR algorithm to find relevant, diverse chunks
6. **Generation**: Gemini AI generates answers based only on retrieved context

## Configuration

### Chunk Size Optimization
```python
# Text files
chunk_size = 800
chunk_overlap = 100

# PDF files  
chunk_size = 1200
chunk_overlap = 150
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
model = "gemini-2.5-flash-lite"  # Free tier model
temperature = 1
embedding_model = "all-MiniLM-L6-v2"  # Free HuggingFace model
```

## Important Notes

- ✅ Only uses information from provided documents (no hallucinations)
- ✅ First run creates vector database which persists locally
- ✅ Avoids duplicate document ingestion
- ⚠️ Requires active internet for Gemini API calls
- ⚠️ Free tier has rate limits (60 requests/minute for Gemini)

## Roadmap

- [x] Basic RAG implementation
- [x] Support for PDF and text files
- [x] Prevent duplicate document ingestion
- [x] Interactive chat loop
- [ ] Add Streamlit UI for better user experience
- [ ] Implement file upload functionality
- [ ] Add support for multiple file formats (.docx, .csv, etc.)
- [ ] Add persistent chat history
- [ ] Show source citations with page numbers
- [ ] Multi-document support in single session
- [ ] Conversation memory for follow-up questions

## Technologies Used

- **LangChain**: Framework for LLM applications
- **ChromaDB**: Vector database for embeddings
- **HuggingFace**: Free embedding models (`all-MiniLM-L6-v2`)
- **Google Gemini**: Free-tier language model
- **PyPDF**: PDF processing
- **Python-dotenv**: Environment variable management

## Limitations

- Requires internet connection for API calls
- Free tier rate limits apply
- Single document context per session
- No conversation memory between queries

## Troubleshooting

**"No module named 'prompts.py'"**
- Ensure `prompts.py` exists and use `from prompts import concise_prompt`

**"ModuleNotFoundError"**
- Install all dependencies: `pip install -r requirements.txt`

**"API key not found"**
- Create `.env` file with your Google API key

**"I don't have enough information"**
- Check if documents were properly embedded
- Verify collection name matches
- Try rephrasing your question

## Contributing

Feel free to open issues or submit pull requests! This is a learning project and contributions are welcome.

## License

MIT

## Author

Built while learning LangChain and RAG concepts. Using free-tier models due to no access to paid OpenAI services.

-Mohammad Faraz Rajput
---

⭐ If you find this helpful, please star the repository!