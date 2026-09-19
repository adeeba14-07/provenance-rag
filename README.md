
#  Provenance RAG

### Source-Cited Retrieval-Augmented Generation for Trustworthy Document Analysis

Provenance RAG is a Retrieval-Augmented Generation (RAG) system designed to make AI-generated answers more **transparent, traceable, and evidence-based**.

The system retrieves relevant information from uploaded documents, generates an answer using the retrieved context, and verifies factual claims against the source material.

---

##  Features

-  Upload and analyze documents
-  Hybrid retrieval using **BM25 + Vector Search**
-  Reciprocal Rank Fusion (RRF)
-  Cross-Encoder reranking
-  Retrieval-Augmented Generation
-  Source-cited answers
-  Claim-level verification
-  Trust score for generated answers
-  Numerical calculation handling
-  RAG evaluation metrics
-  Document-based conversations
-  Analytics dashboard

---

##  How It Works

```text
Document Upload
      ↓
Text Extraction & Chunking
      ↓
Embedding Generation
      ↓
ChromaDB Vector Store
      ↓
Hybrid Retrieval
(BM25 + Vector Search)
      ↓
Reciprocal Rank Fusion
      ↓
Cross-Encoder Reranking
      ↓
LLM Answer Generation
      ↓
Claim Extraction
      ↓
Claim Verification
      ↓
Source Evidence + Trust Score
````

---

##  Claim Verification

Generated answers are checked against the retrieved source content.

Each factual claim is classified as:

| Status           | Meaning                          |
| ---------------- | -------------------------------- |
| 🟢 `SUPPORTED`   | Strongly supported by the source |
| 🟡 `PARTIAL`     | Partially supported              |
| 🔴 `UNSUPPORTED` | Not sufficiently supported       |

A trust score is then calculated based on the verification results.

---

##  Tech Stack

**Language:** Python

**Framework:** Streamlit

**RAG & Retrieval:** ChromaDB, BM25, Sentence Transformers, RRF

**Reranking:** Cross-Encoder

**AI:** LLM-based generation through Groq API

**Document Processing:** PyPDF, LangChain

**Data & Visualization:** Pandas, NumPy, Plotly

---

##  Project Structure

```text
provenance-rag/
│
├── app.py
├── retrieval.py
├── verifier.py
├── calculator.py
├── evaluation.py
├── auth.py
├── conversations.py
├── completeness.py
├── tracker.py
├── settings.py
├── theme.py
├── pages/
├── requirements.txt
└── .env.example
```

---

## To Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/adeeba14-07/provenance-rag.git
cd provenance-rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API key

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

### 4. Run the application

```bash
streamlit run app.py
```

---

## 📊 Evaluation

The project includes evaluation for:

* Faithfulness
* Answer Relevancy
* Context Precision
* Context Recall

---

##  Limitations

The trust score indicates how strongly a generated claim is supported by the retrieved source material. It should not be interpreted as a guarantee of absolute factual truth.

Human review is still recommended for high-stakes applications.

---

##  Future Enhancements

* Improved contradiction detection
* Page-level citations
* Multi-document comparison
* Advanced hallucination detection
* Better numerical reasoning
* Additional document formats
* External source verification

---

##  Author

**Adeeba Tasneem**

B.Tech Computer Science & Engineering

GitHub: [https://github.com/adeeba14-07](https://github.com/adeeba14-07)

---

