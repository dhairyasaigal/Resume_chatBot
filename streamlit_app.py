"""
Streamlit UI for the AI Interview Assistant.
Includes:
- PostgreSQL-backed User Authentication (Sign In & Sign Up)
- User-specific Thread & Conversation History Persistence
- Input & Output Security Guardrails
- LangSmith Tracing & Observability
- Real-time Token Streaming
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

import uuid
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.config import get_settings
from app.auth.auth_manager import (
    init_auth_db,
    signup_user,
    authenticate_user,
    get_user_threads,
    save_user_thread,
    delete_user_thread,
)
from app.guardrails.input_guardrail import validate_input_guardrail
from app.guardrails.output_guardrail import validate_output_guardrail
from app.rag.retriever import retrieve, format_context, get_sources
from app.llm.groq_model import get_llm
from app.prompts.system_prompt import SYSTEM_PROMPT

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Interview Assistant — Dhairya Saigal",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for Premium Design ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Main container styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Auth card styling */
    .auth-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 2rem;
        margin-top: 1rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 6px;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .badge-green {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-blue {
        background: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Lazy Resource Initialization (Cached) ────────────────────────────────────
@st.cache_resource(show_spinner="Connecting to database & initializing tables...")
def init_system():
    """Initialize LangGraph checkpointer, auth tables, and pre-warm embeddings."""
    from app.memory.checkpointer import get_checkpointer
    from app.graph.workflow import build_graph
    from app.rag.embeddings import get_embeddings

    try:
        init_auth_db()
        checkpointer = get_checkpointer()
        app = build_graph(checkpointer=checkpointer)
        # Pre-warm embeddings
        get_embeddings()
        return app, None
    except Exception as e:
        return None, str(e)


# ── Session State Initialization ──────────────────────────────────────────────
def init_session_state():
    if "user" not in st.session_state:
        st.session_state.user = None  # {id, username, full_name}
    if "threads" not in st.session_state:
        st.session_state.threads = {}  # thread_id -> {name, created_at, last_updated}
    if "active_thread" not in st.session_state:
        st.session_state.active_thread = None
    if "thread_messages" not in st.session_state:
        st.session_state.thread_messages = {}  # thread_id -> list of messages


def load_user_sessions(user_id: int):
    """Load persistent threads from PostgreSQL for the authenticated user."""
    threads_list = get_user_threads(user_id)
    st.session_state.threads = {}
    for t in threads_list:
        st.session_state.threads[t["thread_id"]] = {
            "name": t["name"],
            "created_at": t["created_at"],
            "last_updated": t["last_updated"],
        }
    if threads_list:
        st.session_state.active_thread = threads_list[-1]["thread_id"]
    else:
        create_new_thread(user_id)


def create_new_thread(user_id: int) -> str:
    """Create a new thread for the user and save it to PostgreSQL."""
    thread_id = f"thread_{uuid.uuid4().hex[:8]}"
    now = datetime.now().strftime("%b %d, %H:%M")
    count = len(st.session_state.threads) + 1
    name = f"Interview {count:03d}"

    st.session_state.threads[thread_id] = {
        "name": name,
        "created_at": now,
        "last_updated": now,
    }
    st.session_state.active_thread = thread_id
    st.session_state.thread_messages[thread_id] = []

    # Persist to database
    save_user_thread(user_id, thread_id, name, now, now)
    return thread_id


def get_thread_history(app, thread_id: str) -> list[dict]:
    """Re-hydrate display messages from LangGraph PostgreSQL checkpoint."""
    if app is None or not thread_id:
        return []
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = app.get_state(config)
        if state is None or not state.values:
            return []

        messages = state.values.get("messages", [])
        display = []
        sources_buffer = state.values.get("sources", [])

        for msg in messages:
            if isinstance(msg, HumanMessage):
                display.append({"role": "user", "content": msg.content, "sources": []})
            elif isinstance(msg, AIMessage):
                display.append({"role": "assistant", "content": msg.content, "sources": []})

        # Attach sources to the last assistant message if available
        if display and display[-1]["role"] == "assistant" and sources_buffer:
            display[-1]["sources"] = sources_buffer

        return display
    except Exception:
        return []


# ── Authentication View (Sign In & Sign Up) ──────────────────────────────────
def render_auth_view():
    st.markdown(
        """
        <div style='text-align:center; padding: 2rem 0 1rem 0;'>
            <h1 style='margin-bottom:0;'>🤖 AI Interview Assistant</h1>
            <p style='color: #9CA3AF; font-size: 1.1rem; margin-top: 6px;'>
                Dhairya Saigal — Interactive Resume & Technical Interview Agent
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab_login, tab_signup = st.tabs(["🔑 Sign In", "📝 Create Account"])

        with tab_login:
            st.markdown("##### Welcome back! Sign in to access your interview history.")
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", placeholder="e.g. recruiter_john").strip()
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submitted = st.form_submit_button("Sign In", use_container_width=True, type="primary")

                if submitted:
                    user, msg = authenticate_user(username, password)
                    if user:
                        st.session_state.user = user
                        st.success(f"Welcome back, {user['full_name']}!")
                        load_user_sessions(user["id"])
                        st.rerun()
                    else:
                        st.error(msg)

        with tab_signup:
            st.markdown("##### New here? Register to persist your interview sessions.")
            with st.form("signup_form", clear_on_submit=False):
                new_name = st.text_input("Full Name", placeholder="e.g. Jane Doe (Tech Recruiter)")
                new_username = st.text_input("Username", placeholder="Choose a username (min 3 chars)").strip()
                new_password = st.text_input("Password", type="password", placeholder="Choose a password (min 6 chars)")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
                signup_submitted = st.form_submit_button("Create Account", use_container_width=True, type="primary")

                if signup_submitted:
                    if new_password != confirm_password:
                        st.error("Passwords do not match. Please re-enter.")
                    else:
                        success, msg = signup_user(new_username, new_password, new_name)
                        if success:
                            st.success(msg)
                            # Auto login
                            user, _ = authenticate_user(new_username, new_password)
                            if user:
                                st.session_state.user = user
                                load_user_sessions(user["id"])
                                st.rerun()
                        else:
                            st.error(msg)


# ── Main Interview Dashboard ─────────────────────────────────────────────────
def render_main_dashboard(app):
    user = st.session_state.user

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"### 👤 {user.get('full_name', user['username'])}")
        st.caption(f"Logged in as **@{user['username']}**")

        # Status Badges
        st.markdown(
            """
            <div>
                <span class='status-badge badge-green'>🛡️ Guardrails Active</span>
                <span class='status-badge badge-blue'>📊 LangSmith Traced</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_new, col_out = st.columns([2, 1])
        with col_new:
            if st.button("＋ New Interview", use_container_width=True, type="primary"):
                create_new_thread(user["id"])
                st.rerun()
        with col_out:
            if st.button("Logout", use_container_width=True):
                st.session_state.user = None
                st.session_state.threads = {}
                st.session_state.active_thread = None
                st.session_state.thread_messages = {}
                st.rerun()

        st.divider()
        st.markdown("#### 🗂 Past Interviews")

        if not st.session_state.threads:
            st.caption("No sessions yet. Click 'New Interview' to start.")
        else:
            for tid, meta in reversed(list(st.session_state.threads.items())):
                is_active = tid == st.session_state.active_thread
                label = f"{'▶ ' if is_active else ''}{meta['name']}"

                col_btn, col_del = st.columns([5, 1])
                with col_btn:
                    if st.button(label, key=f"btn_{tid}", use_container_width=True):
                        st.session_state.active_thread = tid
                        if tid not in st.session_state.thread_messages or not st.session_state.thread_messages[tid]:
                            st.session_state.thread_messages[tid] = get_thread_history(app, tid)
                        st.rerun()
                with col_del:
                    if st.button("🗑️", key=f"del_{tid}", help="Delete this interview"):
                        delete_user_thread(user["id"], tid)
                        st.session_state.threads.pop(tid, None)
                        st.session_state.thread_messages.pop(tid, None)
                        if st.session_state.active_thread == tid:
                            remaining = list(st.session_state.threads.keys())
                            st.session_state.active_thread = remaining[-1] if remaining else None
                        st.rerun()

                st.caption(f"Created: {meta.get('created_at', '')}")

    # ── Main Chat Area ────────────────────────────────────────────────────────
    if not st.session_state.active_thread:
        st.info("👈 Click **＋ New Interview** in the sidebar to start a session.")
        return

    thread_id = st.session_state.active_thread
    thread_meta = st.session_state.threads.get(thread_id, {"name": "Interview", "created_at": ""})

    # Header
    st.markdown(
        f"""
        <div style='padding-bottom: 0.5rem;'>
            <h3 style='margin-bottom: 0;'>🤖 AI Interview Assistant — Dhairya Saigal</h3>
            <span style='color: gray; font-size: 0.9rem;'>Session: <b>{thread_meta['name']}</b> &nbsp;|&nbsp; <code>{thread_id}</code></span>
            <hr style='margin-top: 0.5rem; margin-bottom: 1rem;'>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Re-hydrate messages if empty
    if thread_id not in st.session_state.thread_messages or not st.session_state.thread_messages[thread_id]:
        st.session_state.thread_messages[thread_id] = get_thread_history(app, thread_id)

    # Display Chat History
    chat_container = st.container()
    with chat_container:
        messages = st.session_state.thread_messages.get(thread_id, [])
        for msg in messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    with st.expander("📎 Sources used", expanded=False):
                        for src in msg["sources"]:
                            st.markdown(f"▸ `{src}`")

    # Chat Input
    user_input = st.chat_input("Ask a question about Dhairya's experience, skills, projects, or background...")

    if user_input:
        if not user_input.strip():
            st.warning("Please enter a question.")
            return

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.thread_messages[thread_id].append(
            {"role": "user", "content": user_input, "sources": []}
        )

        # ── Step 1: Input Guardrail Check ──────────────────────────────────────
        is_safe, guardrail_reason = validate_input_guardrail(user_input)

        with st.chat_message("assistant"):
            if not is_safe:
                # Intercepted by Guardrail — Output Safe Refusal
                answer = guardrail_reason
                sources = []
                st.warning(f"🛡️ **Guardrail Triggered:**\n\n{answer}")
            else:
                # Safe Query — Perform RAG Retrieval & Stream Generation
                with st.spinner("Retrieving knowledge base..."):
                    docs = retrieve(user_input)
                    context = format_context(docs)
                    sources = get_sources(docs)

                # Prepare LLM streaming prompt
                system_content = SYSTEM_PROMPT
                if context:
                    system_content += f"\n\n## Retrieved Knowledge Base Context\n\n{context}"
                else:
                    system_content += "\n\n## Retrieved Knowledge Base Context\n\nNo relevant context was retrieved for this query."

                # Fetch past conversation messages from state for context continuity
                history_display = st.session_state.thread_messages.get(thread_id, [])[:-1]
                conversation_msgs = []
                for m in history_display:
                    if m["role"] == "user":
                        conversation_msgs.append(HumanMessage(content=m["content"]))
                    elif m["role"] == "assistant":
                        conversation_msgs.append(AIMessage(content=m["content"]))

                prompt_messages = [SystemMessage(content=system_content)] + conversation_msgs + [HumanMessage(content=user_input)]

                llm = get_llm()

                # Stream tokens in real time
                def stream_tokens():
                    for chunk in llm.stream(prompt_messages):
                        if chunk.content:
                            yield chunk.content

                raw_streamed_answer = st.write_stream(stream_tokens())

                # Apply Output Guardrail
                answer = validate_output_guardrail(raw_streamed_answer)

                if sources:
                    with st.expander("📎 Sources used", expanded=False):
                        for src in sources:
                            st.markdown(f"▸ `{src}`")

            # ── Step 2: Sync with LangGraph Checkpointer ───────────────────────
            try:
                config = {"configurable": {"thread_id": thread_id}}
                app.invoke(
                    {
                        "messages": [
                            HumanMessage(content=user_input),
                            AIMessage(content=answer),
                        ],
                        "sources": sources,
                    },
                    config=config,
                )
            except Exception as e:
                pass

        # Store in session state
        st.session_state.thread_messages[thread_id].append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

        # Update last_updated in PostgreSQL and state
        now = datetime.now().strftime("%b %d, %H:%M")
        st.session_state.threads[thread_id]["last_updated"] = now
        save_user_thread(user["id"], thread_id, thread_meta["name"], thread_meta.get("created_at", now), now)

        st.rerun()


# ── Main Entrypoint ──────────────────────────────────────────────────────────
def main():
    init_session_state()
    app, db_error = init_system()

    if db_error:
        st.error(f"Database connection failed: {db_error}\n\nCheck your POSTGRES_URL in .env")
        st.stop()

    if st.session_state.user is None:
        render_auth_view()
    else:
        render_main_dashboard(app)


if __name__ == "__main__":
    main()
