# Projects — Dhairya Saigal

## Projects Overview

Dhairya has built 5 projects spanning AI automation, RAG systems, quality engineering, and deep learning:

1. **NG Mail Router / Email File Sender V2 (HMSI)** — Python automation system that monitors CMM inspection PDFs, detects NG parts, and auto-routes email alerts to responsible departments. Eliminated 100% manual intervention.

2. **Automated OOT Trend Analysis & Cp/Cpk Reporting (HMSI)** — Automated quality pipeline that processes monthly CMM data, computes Cp/Cpk process capability metrics, identifies recurring Out-of-Tolerance characteristics, and generates audit-ready Excel reports.

3. **AI Onboarding Mentor Agent (HMSI)** — Dual-agent RAG system built with LangGraph for autonomous trainee onboarding at Honda. Includes an evaluator agent for end-of-day learning assessments. Reduced dependency on human trainers.

4. **Resume Interview Agent** — AI-powered RAG interview assistant (this project) using LangGraph, Qdrant, Groq LLM, PostgreSQL memory, and Streamlit. Represents Dhairya in interviews by retrieving and answering questions from his personal knowledge base.

5. **LearnFlow — AI Study Companion** — Personalized study app for JEE/NEET/Board students. Trained LSTM Forgetting-Curve model from scratch (76.6% accuracy, AUC 0.79). FastAPI + React 19 frontend, deployed on Hugging Face Spaces.

---

## Project 1: NG Mail Router / Email File Sender V2 (HMSI)

**Tech Stack:** Python, PDF Parsing, File System Monitoring, SMTP Email Automation, Logging

### Problem
HMSI's CMM inspection process required manual PDF checking to identify NG (Out-of-Tolerance) parts and notify responsible departments, causing delays.

### Solution
- Monitors MeasureLink output folder continuously
- Reads and analyses each new inspection PDF
- Detects NG characteristics automatically
- Identifies responsible department via configured part/department mapping
- Sends email alerts to concerned teams automatically
- Maintains full logs of processed reports

### Impact
Reduced manual intervention by 100%. Enabled real-time quality alerts without printing physical reports.

---

## Project 2: Automated OOT Trend Analysis & Cp/Cpk Reporting System (HMSI)

**Tech Stack:** Python, Pandas, NumPy, PDF Parsing, Statistical Analysis, Matplotlib, OpenPyXL, Excel Automation

### Problem
Quality teams manually analysed monthly inspection reports, calculated process-capability metrics, and populated official Excel templates — time-consuming and error-prone.

### Solution
- Processes monthly CMM/MeasureLink inspection reports
- Identifies and ranks recurring Out-of-Tolerance (OOT) characteristics
- Calculates Mean, Standard Deviation, Cp, Cpk, Cpl, Cpu automatically
- Maps results into official Cp/Cpk Excel reporting format
- Applies automated pass/fail validation for audit-ready output

### Impact
Fully automated pipeline: CMM Reports → OOT Trend Analysis → Cp/Cpk Calculation → Official Quality Report. Eliminated manual errors and saved significant engineering time.

---

## Project 3: AI Onboarding Mentor Agent (HMSI)

**Tech Stack:** Python, LangGraph, RAG, Groq

### Overview
Dual-agent RAG system for autonomous trainee onboarding at Honda Motorcycle & Scooter India, covering workflows, SOPs, and technical processes.

### Key Features
- Two-agent architecture: retrieval agent + evaluator agent
- End-of-day automated learning assessments
- Covers manufacturing workflows, policies, and technical processes
- Enterprise security architecture with full traceability

### Impact
Reduced dependency on human trainers. Presented in New Honda Circle (NHC) with Quality Guardians team.

---

## Project 4: Resume Interview Agent

**Tech Stack:** Python, LangGraph, Qdrant, Groq LLM, PostgreSQL, Streamlit, Sentence Transformers

### Overview
AI-powered RAG interview assistant that represents Dhairya during interviews by retrieving information from a structured personal knowledge base and generating grounded responses.

### Architecture
- LangGraph workflow: retrieve_context → generate_answer
- Qdrant vector database for semantic search
- Groq LLM for response generation
- PostgreSQL (Neon) for persistent multi-session memory
- Streamlit frontend with multi-session management

### Key Features
- Chunked markdown knowledge base across 7 categories
- Multi-turn context-aware conversations
- Source citation on every response
- RAG evaluation pipeline (correctness, faithfulness, retrieval precision)

---

## Project 5: LearnFlow — AI Study Companion

**Tech Stack:** Python, FastAPI, React 19, TensorFlow, LSTM, SQLite, Hugging Face Spaces

### Overview
Personalized AI study companion for competitive exam students (JEE, NEET, Board exams).

### Key Features
- Trained LSTM Forgetting-Curve model from scratch: 76.6% accuracy, AUC 0.79
- Synthetic student behavior data generation for training
- 49+ topics across JEE, NEET, and Board exam streams
- FastAPI backend + React 19 frontend + SQLite database
- Deployed on Hugging Face Spaces

### Innovation
Custom deep learning model for personalized learning curves — adapts study schedules based on predicted memory retention.
