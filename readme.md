# RAG PDF Question Answering System
This project implements a Retrieval-Augmented Generation (RAG) system using Google Gemini, LangChain, and FAISS. It allows users to upload PDF documents and ask questions about their content through a Streamlit web interface.

## Features
- 📤 Upload and process PDF documents
- 🔍 Split text into semantic chunks
- 🧠 Generate embeddings using HuggingFace (local, no API overhead)
- 💾 Store embeddings in FAISS for fast similarity search
- 🤖 Answer questions using Google Gemini AI
- 🔄 Automatic index management per session/file
- ✅ Input validation and error handling
- 🧹 Session cleanup and index management

## Tech Stack
- **Frontend**: Streamlit
- **LLM**: Google Gemini 2.5 Pro
- **Embeddings**: HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
- **Vector DB**: FAISS
- **Framework**: LangChain
- **Language**: Python 3.10+

## How It Works
1. Load environment variables from `.env` file
2. User uploads a PDF via Streamlit interface
3. PDF is validated and sanitized
4. Text is extracted and split into overlapping chunks
5. Chunks are converted into embeddings (locally)
6. Embeddings are stored in a FAISS index (per session/file)
7. When user asks a question, the system:
   - Retrieves the most relevant chunks
   - Sends them to Gemini with the question
   - Returns the AI-generated answer

## Setup Instructions

### 1. Clone & Create Virtual Environment
\`\`\`bash
git clone <repo-url>
cd third
python -m venv ai_agent_env
ai_agent_env\Scripts\activate  # Windows
source ai_agent_env/bin/activate  # macOS/Linux
\`\`\`

### 2. Install Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 3. Configure API Key
Create a `.env` file in the project root:
\`\`\`
GOOGLE_API_KEY=your_google_api_key_here
\`\`\`

**How to get the API key:**
- Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
- Create a free API key for Gemini
- Add it to your `.env` file

### 4. Run the Application
\`\`\`bash
streamlit run app.py
\`\`\`

The app will open in your default browser at `http://localhost:8501`

## Usage
1. **Upload PDF**: Click "Browse files" and select a PDF (max 50MB)
2. **Ask Questions**: Type your question and click "Get Answer"
3. **Manage Sessions**: Use the buttons at the bottom to clear PDFs or indices
4. **Upload New PDF**: Simply upload a new file to start a new session

## Security Notes
- ⚠️ **NEVER commit `.env` to version control** (it's in `.gitignore`)
- 🔐 The API key must be rotated if ever exposed
- 📄 Uploaded PDFs are stored in `uploads/` (add to `.gitignore` if needed)
- 🔍 Filenames are sanitized to prevent security issues

## Project Structure
\`\`\`
├── app.py                 # Streamlit UI
├── backend.py             # RAG system logic
├── requirements.txt       # Python dependencies
├── .env                   # API keys (not in repo)
├── .gitignore            # Git ignore rules
├── README.md             # This file
├── faiss_index/          # Vector database (per session)
└── uploads/              # Temporary PDF storage
\`\`\`

## Troubleshooting

**Q: "GOOGLE_API_KEY not found in environment"**
- A: Create a `.env` file with your API key

**Q: "File too large" error**
- A: Maximum file size is 50MB. Try a smaller PDF

**Q: "Invalid PDF file" error**
- A: Ensure the uploaded file is a valid PDF

**Q: Old answers appearing for new PDF**
- A: Click "Clear Current PDF" or "Clear All Indices" to reset

## Future Improvements
- [ ] Support for multiple PDF analysis simultaneously
- [ ] Chat history/conversation memory
- [ ] Custom chunking strategies
- [ ] Export answers to file
- [ ] Support for other document formats (DOCX, TXT, etc.)
- [ ] Fine-tuning for domain-specific knowledge

## .gitignore
\`\`\`
# Environment variables
.env

# Python
__pycache__/
*.pyc
venv/
ai_agent_env/

# Project
faiss_index/
uploads/
*.pdf
\`\`\`

## Author
Purvesh Gandhi  
B.Tech CSE Student

## License
MIT
