# 🧠 Simple RAG-Based Document Assistant

This is a beginner-friendly **Retrieval-Augmented Generation (RAG)** application that allows you to ask questions based on a PDF document using Streamlit and LangChain.

---

## 📚 Usage

1. Place your PDF file in the `data/` folder (e.g., `BOI.pdf`)
2. Install dependencies:

```bash
git clone https://github.com/yourusername/simple-rag-assistant.git
cd simple-rag-assistant
python -m venv venv
venv\Scripts\activate  # On Windows
# or source venv/bin/activate  # On Unix/Mac
pip install -r requirements.txt
```

3. Start the Streamlit app:

```bash
streamlit run app.py
```

4. Ask questions based on the content of the PDF

---

## 🧰 Tech Stack

- **LangChain** for chaining and retrieval
- **Ollama** for local language models
- **ChromaDB** for storing and querying document embeddings
- **Streamlit** for the web UI

---

That's it! Simple, clean, and perfect for learning how RAG systems work.

