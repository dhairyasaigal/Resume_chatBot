"""
RAG Evaluation Pipeline — Correctness, Faithfulness, Retrieval Precision.

Uses the Groq LLM as judge for correctness and faithfulness.
No external eval framework needed.

Run:
    python tests/evaluate.py
    python tests/evaluate.py --verbose
    python tests/evaluate.py --output results.json
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import json
import argparse
import textwrap
from dataclasses import dataclass, asdict
from langchain_core.messages import HumanMessage
from app.graph.workflow import build_graph
from app.rag.retriever import retrieve, get_sources
from app.llm.groq_model import get_llm


# ── Test dataset ──────────────────────────────────────────────────────────────
# Format: (question, expected_keywords_in_answer, expected_sources)
EVAL_DATASET = [
    {
        "id": "Q01",
        "question": "What is Dhairya's educational background?",
        "expected_keywords": ["vips", "b.tech", "computer science", "iit madras"],
        "expected_sources": ["education.md"],
        "reference": "Dhairya is pursuing B.Tech in CS & AI at VIPS Delhi and completed B.S. in Electronic Systems from IIT Madras.",
    },
    {
        "id": "Q02",
        "question": "Tell me about Dhairya's internship at Honda.",
        "expected_keywords": ["honda", "hmsi", "python", "automation"],
        "expected_sources": ["internships.md"],
        "reference": "Dhairya interned at Honda Motorcycle & Scooter India (HMSI) as an AI & Python Automation Intern, building production alerting systems and Power BI dashboards.",
    },
    {
        "id": "Q03",
        "question": "What are Dhairya's technical skills?",
        "expected_keywords": ["python", "langchain", "pytorch"],
        "expected_sources": ["skills.md"],
        "reference": "Dhairya's skills include Python, LangChain, LangGraph, PyTorch, TensorFlow, FastAPI, Qdrant, Docker, and more.",
    },
    {
        "id": "Q04",
        "question": "What projects has Dhairya built?",
        "expected_keywords": ["langgraph", "learnflow", "hmsi", "rag"],
        "expected_sources": ["projects.md"],
        "reference": "Dhairya built 5 projects: NG Mail Router, OOT Trend Analysis system, AI Onboarding Mentor Agent (HMSI), Resume Interview Agent, and LearnFlow AI Study Companion.",
    },
    {
        "id": "Q05",
        "question": "What research has Dhairya done?",
        "expected_keywords": ["acims", "conference", "research"],
        "expected_sources": ["research.md"],
        "reference": "Dhairya has presented research at the ACIMS International Conference.",
    },
    {
        "id": "Q06",
        "question": "What achievements and awards does Dhairya have?",
        "expected_keywords": ["hackathon", "smart india"],
        "expected_sources": ["achievements.md"],
        "reference": "Dhairya has top-10 finishes in national hackathons including Smart India Hackathon 2025.",
    },
    {
        "id": "Q07",
        "question": "What is Dhairya's salary expectation?",
        "expected_keywords": ["don't have", "not available", "not provided", "no information", "unavailable"],
        "expected_sources": [],   # any or none — this tests hallucination guard
        "reference": "This information is not available in Dhairya's knowledge base.",
    },
    {
        "id": "Q08",
        "question": "What did Dhairya do at SS Medi Solutions?",
        "expected_keywords": ["machine learning", "covid", "vaccination", "pandas"],
        "expected_sources": ["internships.md"],
        "reference": "Dhairya interned at SS Medi Solutions as a Machine Learning Intern, analyzing COVID-19 vaccination data.",
    },
]


# ── Scoring helpers ───────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    id: str
    question: str
    answer: str
    sources: list
    retrieval_score: float      # 0-1: did we get the expected sources?
    keyword_score: float        # 0-1: fraction of expected keywords in answer
    faithfulness_score: float   # 0-1: LLM judge — is answer grounded in context?
    correctness_score: float    # 0-1: LLM judge — is answer correct vs reference?
    overall: float              # weighted average
    notes: str = ""


def score_retrieval(sources: list[str], expected: list[str]) -> float:
    """Precision: fraction of expected sources that were retrieved."""
    if not expected:
        return 1.0  # No expected source = retrieval not being tested
    hits = sum(1 for exp in expected if any(exp in s for s in sources))
    return hits / len(expected)


def score_keywords(answer: str, keywords: list[str]) -> float:
    """Fraction of expected keywords found in the answer (case-insensitive)."""
    if not keywords:
        return 1.0
    ans = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in ans)
    return hits / len(keywords)


def llm_judge(question: str, answer: str, context: str, reference: str) -> tuple[float, float]:
    """
    Uses Groq LLM to score:
    - faithfulness: is the answer supported by the retrieved context?
    - correctness: is the answer accurate compared to the reference?
    Returns (faithfulness_score, correctness_score) each 0.0-1.0
    """
    llm = get_llm()

    prompt = f"""You are an expert evaluator for a RAG-based interview assistant chatbot.

Score the following answer on two dimensions. Reply ONLY with a JSON object.

QUESTION: {question}

RETRIEVED CONTEXT:
{context[:2000]}

REFERENCE ANSWER:
{reference}

GENERATED ANSWER:
{answer}

Scoring criteria:
- faithfulness (0.0-1.0): Is the generated answer supported by the retrieved context? 
  1.0 = fully grounded, 0.5 = partially grounded, 0.0 = contradicts or ignores context
- correctness (0.0-1.0): Is the generated answer factually correct compared to the reference?
  1.0 = fully correct, 0.5 = partially correct, 0.0 = incorrect or hallucinated

Reply ONLY with this exact JSON (no markdown, no explanation):
{{"faithfulness": 0.0, "correctness": 0.0}}"""

    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        scores = json.loads(text.strip())
        f = float(scores.get("faithfulness", 0.5))
        c = float(scores.get("correctness", 0.5))
        return round(min(max(f, 0.0), 1.0), 2), round(min(max(c, 0.0), 1.0), 2)
    except Exception as e:
        return 0.5, 0.5  # neutral fallback on parse error


# ── Main evaluation runner ────────────────────────────────────────────────────

def run_evaluation(verbose: bool = False) -> list[EvalResult]:
    app = build_graph(checkpointer=None)
    results = []

    print(f"\n{'='*60}")
    print("  RAG Evaluation Pipeline")
    print(f"  {len(EVAL_DATASET)} test cases")
    print(f"{'='*60}\n")

    for i, case in enumerate(EVAL_DATASET, 1):
        qid = case["id"]
        question = case["question"]
        print(f"[{i}/{len(EVAL_DATASET)}] {qid}: {question[:60]}...")

        # Run the graph
        try:
            graph_result = app.invoke(
                {"messages": [HumanMessage(content=question)]},
                config={"configurable": {"thread_id": f"eval_{qid}"}},
            )
            answer = graph_result.get("answer", "")
            sources = graph_result.get("sources", [])
            context = graph_result.get("context", "")
        except Exception as e:
            print(f"  ERROR running graph: {e}")
            answer, sources, context = "", [], ""

        # Scores
        ret_score = score_retrieval(sources, case["expected_sources"])
        kw_score = score_keywords(answer, case["expected_keywords"])
        faith_score, corr_score = llm_judge(question, answer, context, case["reference"])

        # Weighted overall: retrieval 25%, keywords 25%, faithfulness 25%, correctness 25%
        overall = round((ret_score + kw_score + faith_score + corr_score) / 4, 2)

        result = EvalResult(
            id=qid,
            question=question,
            answer=answer,
            sources=sources,
            retrieval_score=ret_score,
            keyword_score=kw_score,
            faithfulness_score=faith_score,
            correctness_score=corr_score,
            overall=overall,
        )
        results.append(result)

        # Print result row
        status = "✅" if overall >= 0.7 else "⚠️" if overall >= 0.4 else "❌"
        print(f"  {status} Retrieval={ret_score:.2f}  Keywords={kw_score:.2f}  "
              f"Faithfulness={faith_score:.2f}  Correctness={corr_score:.2f}  "
              f"Overall={overall:.2f}")

        if verbose:
            print(f"\n  Answer: {textwrap.fill(answer[:300], width=70, subsequent_indent='  ')}")
            print(f"  Sources: {sources}\n")

    return results


def print_summary(results: list[EvalResult]):
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")

    avg = lambda key: round(sum(getattr(r, key) for r in results) / len(results), 3)

    print(f"  Total cases      : {len(results)}")
    print(f"  Passed (≥0.7)    : {sum(1 for r in results if r.overall >= 0.7)}/{len(results)}")
    print(f"  Avg Retrieval    : {avg('retrieval_score')}")
    print(f"  Avg Keywords     : {avg('keyword_score')}")
    print(f"  Avg Faithfulness : {avg('faithfulness_score')}")
    print(f"  Avg Correctness  : {avg('correctness_score')}")
    print(f"  Avg Overall      : {avg('overall')}")
    print(f"{'='*60}\n")

    # Weakest cases
    weak = [r for r in results if r.overall < 0.7]
    if weak:
        print("  Cases needing improvement:")
        for r in weak:
            print(f"  ❌ {r.id}: {r.question[:55]}... (overall={r.overall})")
    print()


def save_results(results: list[EvalResult], path: str):
    data = [asdict(r) for r in results]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evaluation pipeline")
    parser.add_argument("--verbose", action="store_true", help="Print answers")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    results = run_evaluation(verbose=args.verbose)
    print_summary(results)

    if args.output:
        save_results(results, args.output)
