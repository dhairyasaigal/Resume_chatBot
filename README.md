# Resume Interview Agent

An AI-powered interview assistant that represents Dhairya Saigal in interviews. Interviewers can ask questions about his background, education, skills, projects, internships, and more — and receive grounded answers backed by a personal knowledge base.

## Architecture

```
Interviewer (Streamlit)
        │
        ▼
LangGraph Workflow
  ┌─────────────────────────────────┐
  │  retrieve_context node          │
  │    → Embed query                │
  │    → Qdrant similarity search   │
  │    → Format context string      │
  ├─────────────────────────────────┤
  │  generate_answer node           │
  │    → System prompt + context    │
  │    → Conversation history       │
  │    → Groq LLM (ChatGroq)        │
  └─────────────────────────────────┘
        │
        ▼
PostgreSQL (LangGraph Checkpointer)
  → Persists conversation threads
  → Survives application restarts
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq (llama-3.1-8b-instant) via LangChain |
| Orchestration | LangGraph |
| Vector DB | Qdrant (local or cloud) |
| Embeddings | sentence-transformers/all-mpnet-base-v2 |
| Persistence | PostgreSQL via langgraph-checkpoint-postgres |
| UI | Streamlit |
| Config | pydantic-settings + python-dotenv |

## Project Structure

```
resume-interview-agent/
├── app/
│   ├── config.py              # Centralized settings (pydantic-settings)
│   ├── graph/
│   │   ├── state.py           # InterviewState TypedDict
│   │   ├── nodes.py           # retrieve_context + generate_answer nodes
│   │   └── workflow.py        # LangGraph graph builder
│   ├── rag/
│   │   ├── loader.py          # Recursive .md file loader
│   │   ├── chunker.py         # RecursiveCharacterTextSplitter
│   │   ├── embeddings.py      # HuggingFace embeddings (cached)
│   │   ├── vectorstore.py     # Qdrant collection management
│   │   └── retriever.py       # Similarity search + context formatting
│   ├── llm/
│   │   └── groq_model.py      # ChatGroq singleton
│   ├── memory/
│   │   └── checkpointer.py    # PostgresSaver + connection pool
│   └── prompts/
│       └── system_prompt.py   # AI identity + behavior rules
├── data/                      # Knowledge base (Markdown files)
│   ├── resume.md
│   ├── education.md
│   ├── skills.md
│   ├── internships.md
│   ├── achievements.md
│   ├── research.md
│   └── projects/
│       ├── rakshak.md
│       ├── learnflow.md
│       ├── tourist_safety.md
│       └── sdxl_generator.md
├── scripts/
│   └── ingest.py              # One-time ingestion pipeline
├── tests/
│   ├── test_retrieval.py      # Qdrant retrieval tests
│   ├── test_graph.py          # LangGraph node tests
│   └── test_memory.py         # Persistence tests
├── streamlit_app.py           # Main UI
├── .env.example
└── requirements.txt
```

## Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```env
# Groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant

# Qdrant (leave QDRANT_URL empty for local file-based Qdrant)
QDRANT_URL=
QDRANT_API_KEY=
QDRANT_COLLECTION=resume_knowledge
QDRANT_LOCAL_PATH=./qdrant_storage

# PostgreSQL
POSTGRES_URL=postgresql://user:password@localhost:5432/interview_agent

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2

# RAG
TOP_K=5
CHUNK_SIZE=700
CHUNK_OVERLAP=100
```

## Installation

```bash
cd resume-interview-agent
pip install -r requirements.txt
```

## Starting PostgreSQL

Using Docker (quickest):
```bash
docker run -d \
  --name interview-postgres \
  -e POSTGRES_USER=user \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=interview_agent \
  -p 5432:5432 \
  postgres:15
```

Or use an existing PostgreSQL instance. Make sure the database exists before running the app.

## Starting Qdrant (local)

For the MVP, Qdrant runs as a local file-based store — no separate server needed.
Just leave `QDRANT_URL` empty in `.env` and set `QDRANT_LOCAL_PATH=./qdrant_storage`.

To use Qdrant Cloud instead, set `QDRANT_URL` and optionally `QDRANT_API_KEY`.

## Running Ingestion

Populate your knowledge base files in `data/` first, then:

```bash
python scripts/ingest.py
```

Expected output:
```
=== Resume Knowledge Base Ingestion ===

Loading documents...
Documents loaded: 11
  - resume.md [resume]
  - education.md [education]
  ...

Chunks created: 47
Embedding documents...
Creating Qdrant collection if needed...
Uploading vectors...

=== Ingestion Complete ===
Collection : resume_knowledge
Vectors    : 47
```

## Starting the App

```bash
streamlit run streamlit_app.py
```

## Example Questions

- "Tell me about Dhairya."
- "What did you build at Honda?"
- "Explain Rakshak technically."
- "Why did you choose TensorFlow Lite?"
- "What are your strongest technical skills?"
- "Compare Rakshak and LearnFlow."
- "What is Dhairya's salary?" ← Should respond that it doesn't have this information

## How LangGraph Persistence Works

Each conversation is identified by a `thread_id`. When you invoke the graph with a `thread_id`, LangGraph:

1. Loads the prior state (messages, context) from PostgreSQL
2. Appends the new message and runs the graph
3. Saves the updated state back to PostgreSQL

This means conversations survive application restarts. Reopening a thread resumes exactly where it left off.

## How RAG Works

1. User question → embedding via sentence-transformers
2. Embedding → Qdrant similarity search → top-k chunks
3. Chunks → formatted context string
4. Context + conversation history → Groq LLM
5. LLM generates grounded answer

## Future Roadmap

- P1: Security / Llama Guard jailbreak detection
- P2: LangSmith observability and tracing
- P3: Evaluation pipeline (correctness, faithfulness, retrieval)
- P4: Redis caching for frequent queries
- P5: Docker deployment
