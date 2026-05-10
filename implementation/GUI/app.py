"""
The Devil's Advocate  Streamlit GUI
Black/red theme with per-message RAG context and fallacy detection panels.
Supports multiple named chat sessions.

Run from the implementation/ directory:
    streamlit run GUI/app.py
"""

import sys
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

# GUI lives at implementation/GUI/app.py; add implementation/ to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.loop import DevilsAdvocate

st.set_page_config(
    page_title="The Devil's Advocate",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS
st.markdown("""
<style>
/*  Base  */
*, *::before, *::after { box-sizing: border-box; }
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: #0a0a0a !important;
    color: #f0f0f0 !important;
}
#MainMenu, footer, header { visibility: hidden; }

/*  Sidebar  */
[data-testid="stSidebar"] {
    background-color: #120000 !important;
    border-right: 1px solid #5c0000 !important;
}
[data-testid="stSidebar"] * { color: #e0e0e0; }

/*  Buttons (global)  */
.stButton > button {
    background-color: #8b0000 !important;
    color: #fff !important;
    border: 1px solid #cc2200 !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background-color: #cc2200 !important;
    box-shadow: 0 0 12px rgba(204,34,0,0.45) !important;
    transform: translateY(-1px) !important;
}

/*  Chat input  */
[data-testid="stChatInput"] {
    background-color: #150000 !important;
    border: 1px solid #5c0000 !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #cc2200 !important;
    box-shadow: 0 0 0 2px rgba(204,34,0,0.25) !important;
}
[data-testid="stChatInputTextArea"] {
    background-color: transparent !important;
    color: #f0f0f0 !important;
    caret-color: #ff4444 !important;
}
[data-testid="stChatInputTextArea"]::placeholder { color: #664444 !important; }
[data-testid="stChatInputSubmitButton"] button {
    background-color: #8b0000 !important;
    border-radius: 8px !important;
}
[data-testid="stChatInputSubmitButton"] button:hover {
    background-color: #cc2200 !important;
}

/*  Chat messages  */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border: none !important;
}

/*  Expanders  */
.stExpander {
    border: 1px solid #330000 !important;
    border-radius: 8px !important;
    background-color: #0d0000 !important;
    margin: 2px 0 6px 0 !important;
}
.stExpander details summary {
    background-color: #1a0000 !important;
    color: #ff7777 !important;
    border-radius: 7px !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    padding: 0.4rem 0.75rem !important;
    text-transform: uppercase !important;
}
.stExpander details summary:hover { background-color: #250000 !important; }
.stExpander details[open] summary {
    border-radius: 7px 7px 0 0 !important;
    border-bottom: 1px solid #330000 !important;
}
[data-testid="stExpanderDetails"] {
    background-color: #080000 !important;
    padding: 0.75rem !important;
}

/*  Code / pre  */
code {
    background-color: #1a0000 !important;
    color: #ff9999 !important;
    border-radius: 4px !important;
    padding: 0.1rem 0.3rem !important;
}
pre {
    background-color: #0d0000 !important;
    border: 1px solid #2d0000 !important;
    border-radius: 8px !important;
}

/*  Horizontal rule  */
hr { border-color: #2d0000 !important; }

/*  Scrollbar  */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0a0a0a; }
::-webkit-scrollbar-thumb { background: #5c0000; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #8b0000; }

/*  Text inputs (rename field)  */
[data-testid="stTextInput"] input {
    background-color: #1a0000 !important;
    color: #f0f0f0 !important;
    border: 1px solid #5c0000 !important;
    border-radius: 6px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #cc2200 !important;
    box-shadow: 0 0 0 2px rgba(204,34,0,0.2) !important;
}

/*  Spinner / status  */
[data-testid="stStatusWidget"] {
    background-color: #150000 !important;
    border: 1px solid #5c0000 !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


#  Helpers 

def _new_session_id() -> str:
    return str(uuid.uuid4())[:8]


def _create_session(name: str = None) -> dict:
    sid = _new_session_id()
    label = name or f"Chat {len(st.session_state.sessions) + 1}"
    return {
        "id": sid,
        "name": label,
        "agent": DevilsAdvocate(),
        "messages": [],
        "created_at": datetime.now().strftime("%H:%M"),
    }


def _parse_fallacy(text: str) -> dict:
    """Return {detected: bool, label: str, explanation: str} from fallacy_context string."""
    if not text:
        return {"detected": False, "label": "", "explanation": "No result."}

    lower = text.lower()

    # Regex fallback  no fallacy
    if lower.startswith("no logical fallacy"):
        return {"detected": False, "label": "No Fallacy", "explanation": text}

    # Model result  no_fallacy
    if "no_fallacy" in lower:
        return {"detected": False, "label": "No Fallacy", "explanation": text}

    # Error path
    if lower.startswith("fallacy detection failed") or lower.startswith("multi-query"):
        return {"detected": False, "label": "Error", "explanation": text}

    # Detected (model or heuristic)
    if "fallacy detected" in lower:
        # Try to split label from explanation
        # Format: "Fallacy Detected: <label>. <explanation>"
        after = text.split(":", 1)[-1].strip()
        parts = after.split(".", 1)
        label = parts[0].strip().replace("_", " ").title()
        explanation = parts[1].strip() if len(parts) > 1 else ""
        return {"detected": True, "label": label, "explanation": explanation}

    return {"detected": False, "label": "Unknown", "explanation": text}


#  Session state init 

if "sessions" not in st.session_state:
    first = _create_session("Chat 1")
    st.session_state.sessions = {first["id"]: first}
    st.session_state.active_id = first["id"]

if "active_id" not in st.session_state:
    st.session_state.active_id = next(iter(st.session_state.sessions))


#  Sidebar 

with st.sidebar:
    st.markdown("""
    <div style='padding:1.2rem 1rem 0.8rem; border-bottom:1px solid #3d0000; text-align:center;'>
        <div style='font-size:1rem; font-weight:800; color:#ff4444;
                    text-transform:uppercase; letter-spacing:0.08em;'>
            Devil's Advocate
        </div>
        <div style='font-size:0.68rem; color:#884444; margin-top:0.2rem;'>
            Pipeline RAG / Fallacy Detection
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.75rem'/>", unsafe_allow_html=True)

    if st.button("+ New Chat", key="new_chat", use_container_width=True):
        sess = _create_session()
        st.session_state.sessions[sess["id"]] = sess
        st.session_state.active_id = sess["id"]
        st.rerun()

    st.markdown("<div style='height:0.5rem'/>", unsafe_allow_html=True)

    # Session list
    for sid, sess in list(st.session_state.sessions.items()):
        is_active = sid == st.session_state.active_id

        # Normal session row
        border_color = "#cc2200" if is_active else "#2d0000"
        bg_color = "#2d0000" if is_active else "transparent"
        name_color = "#ff8888" if is_active else "#ccaaaa"
        msg_count = len([m for m in sess["messages"] if m["role"] == "user"])

        st.markdown(f"""
        <div style='background:{bg_color}; border-left:3px solid {border_color};
                    padding:0.45rem 0.75rem; border-radius:0 6px 6px 0;
                    margin-bottom:2px;'>
            <span style='color:{name_color}; font-size:0.85rem; font-weight:{"700" if is_active else "400"};'>
                {sess["name"]}
            </span>
            <span style='color:#554444; font-size:0.65rem; margin-left:0.5rem;'>
                {msg_count} msg{"s" if msg_count != 1 else ""} - {sess["created_at"]}
            </span>
        </div>
        """, unsafe_allow_html=True)

        col_open, col_del = st.columns([4, 1])
        with col_open:
            if st.button(
                "Open" if not is_active else "Active",
                key=f"open_{sid}",
                use_container_width=True,
                disabled=is_active,
            ):
                st.session_state.active_id = sid
                st.rerun()
        with col_del:
            if st.button("Delete", key=f"del_{sid}", use_container_width=True):
                if len(st.session_state.sessions) > 1:
                    del st.session_state.sessions[sid]
                    if st.session_state.active_id == sid:
                        st.session_state.active_id = next(iter(st.session_state.sessions))
                    st.rerun()

    # Connection status at bottom
    st.markdown("<div style='height:1rem'/>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#2d0000; margin:0 0 0.5rem'/>", unsafe_allow_html=True)
    active_sess = st.session_state.sessions[st.session_state.active_id]
    try:
        models = active_sess["agent"].client.models.list()
        model_id = models.data[0].id if models.data else "unknown"
        st.markdown(
            f"<div style='padding:0 0.75rem; font-size:0.7rem;'>"
            f"<span style='color:#44cc44;'>[ON] Connected</span>"
            f"<span style='color:#664444; margin-left:0.4rem;'>{model_id}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    except Exception:
        st.markdown(
            "<div style='padding:0 0.75rem; font-size:0.7rem;'>"
            "<span style='color:#ff4444;'>[OFF] LMStudio offline</span>"
            "</div>",
            unsafe_allow_html=True,
        )


#  Main chat area 

active = st.session_state.sessions[st.session_state.active_id]
messages = active["messages"]
agent: DevilsAdvocate = active["agent"]

# Header
st.markdown(f"""
<div style='padding:0.75rem 0 0.5rem; border-bottom:1px solid #2d0000; margin-bottom:1rem;
            display:flex; align-items:center; gap:0.75rem;'>
    <span style='font-size:1.2rem; font-weight:700; color:#ff6666;'>{active["name"]}</span>
    <span style='font-size:0.75rem; color:#554444; margin-left:auto;'>
        Pipeline RAG · Adversarial Debate Partner
    </span>
</div>
""", unsafe_allow_html=True)

# Welcome screen when chat is empty
if not messages:
    st.markdown("""
    <div style='text-align:center; padding:4rem 2rem; max-width:550px; margin:0 auto;'>
        <div style='font-size:2rem; font-weight:800;
                    background:linear-gradient(135deg,#ff4444,#8b0000);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text; margin-bottom:0.75rem;'>
            Make Your Argument
        </div>
        <div style='color:#664444; font-size:0.95rem; line-height:1.7;'>
            State your position below. The Devil's Advocate will retrieve
            counter-evidence, expose logical fallacies, and systematically
            dismantle everything you say.
        </div>
        <div style='margin-top:1.5rem; font-size:0.75rem; color:#3d1a1a;'>
            RAG context and fallacy analysis are shown beneath each of your messages.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Render existing messages
for msg in messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])

        #  RAG context expander 
        rag = msg.get("rag_context", "")
        queries = msg.get("reformulated_queries", [])
        with st.expander("RAG Context", expanded=False):
            if queries:
                st.markdown("**Reformulated search queries:**")
                for i, q in enumerate(queries, 1):
                    st.markdown(
                        f"<div style='color:#ff9999; font-size:0.82rem; "
                        f"padding:0.2rem 0.5rem; margin:1px 0; "
                        f"border-left:2px solid #660000;'>{i}. {q}</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown("<div style='height:0.5rem'/>", unsafe_allow_html=True)
            if rag:
                st.markdown("**Retrieved context:**")
                st.markdown(
                    f"<div style='font-size:0.8rem; color:#ccaaaa; "
                    f"white-space:pre-wrap; font-family:monospace; "
                    f"background:#080000; padding:0.75rem; border-radius:6px; "
                    f"border:1px solid #2d0000; max-height:300px; overflow-y:auto;'>"
                    f"{rag}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("<span style='color:#554444; font-size:0.8rem;'>No context retrieved.</span>", unsafe_allow_html=True)

        #  Fallacy detection expander 
        fallacy_raw = msg.get("fallacy_context", "")
        fallacy = _parse_fallacy(fallacy_raw)
        with st.expander("Fallacy Detection", expanded=False):
            if fallacy["detected"]:
                badge_style = (
                    "display:inline-block; padding:0.2rem 0.75rem; border-radius:999px; "
                    "font-size:0.75rem; font-weight:700; text-transform:uppercase; "
                    "letter-spacing:0.06em; background:#5c0000; color:#ff9999; "
                    "border:1px solid #8b0000; margin-bottom:0.5rem;"
                )
                st.markdown(
                    f"<div><span style='{badge_style}'>{fallacy['label']}</span></div>",
                    unsafe_allow_html=True,
                )
                if fallacy["explanation"]:
                    st.markdown(
                        f"<div style='font-size:0.82rem; color:#ddbbbb; "
                        f"line-height:1.6; margin-top:0.25rem;'>"
                        f"{fallacy['explanation']}</div>",
                        unsafe_allow_html=True,
                    )
            else:
                badge_style = (
                    "display:inline-block; padding:0.2rem 0.75rem; border-radius:999px; "
                    "font-size:0.75rem; font-weight:700; text-transform:uppercase; "
                    "letter-spacing:0.06em; background:#003300; color:#88ff88; "
                    "border:1px solid #005500; margin-bottom:0.5rem;"
                )
                st.markdown(
                    f"<div><span style='{badge_style}'>{fallacy['label'] or 'No Fallacy'}</span></div>",
                    unsafe_allow_html=True,
                )
                if fallacy["explanation"] and fallacy["label"] not in ("No Fallacy", ""):
                    st.markdown(
                        f"<div style='font-size:0.8rem; color:#888; margin-top:0.25rem;'>"
                        f"{fallacy['explanation']}</div>",
                        unsafe_allow_html=True,
                    )

    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

#  Chat input 

if user_input := st.chat_input("State your argument...", key="chat_input"):

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process with spinner
    with st.spinner("Fetching RAG context, detecting fallacies, generating response…"):
        try:
            result = agent.respond(user_input)
        except Exception as e:
            result = {
                "reply": f"Error: {e}\n\nMake sure LMStudio is running (`lms server start`).",
                "reformulated_queries": [],
                "rag_context": "",
                "fallacy_context": "",
            }

    # Persist to session state
    active["messages"].append({
        "role": "user",
        "content": user_input,
        "rag_context": result["rag_context"],
        "fallacy_context": result["fallacy_context"],
        "reformulated_queries": result["reformulated_queries"],
    })
    active["messages"].append({
        "role": "assistant",
        "content": result["reply"],
    })

    st.rerun()
