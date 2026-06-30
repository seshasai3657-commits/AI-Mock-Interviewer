import streamlit as st
import random
import re

from document_processor import extract_text
from embeddings import create_chunks
from rag_graph import store_vectors, retrieve_docs
from evaluation import evaluate_answer
from question_generator import generate_question
from question_utils import extract_questions


st.set_page_config(
    page_title="AI Mock Interview",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─────────────────────────────────────────
#               CUSTOM CSS
# ─────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Lora:wght@600;700&display=swap');

/* ── Design tokens ── */
:root {
    --bg:         #f4f6fb;
    --white:      #ffffff;
    --border:     #e2e6f0;
    --accent:     #3b6cf7;
    --accent-lt:  #eef2fe;
    --success:    #16a34a;
    --success-lt: #dcfce7;
    --warning:    #b45309;
    --warning-lt: #fef3c7;
    --danger:     #dc2626;
    --danger-lt:  #fee2e2;
    --text:       #111827;
    --text-2:     #4b5563;
    --muted:      #9ca3af;
    --shadow-sm:  0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow:     0 4px 16px rgba(0,0,0,0.07), 0 1px 4px rgba(0,0,0,0.05);
    --r:          12px;
    --r-sm:       8px;
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 5rem; max-width: 880px; margin: auto; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }

/* ────── SIDEBAR ────── */
[data-testid="stSidebar"] {
    background: var(--white) !important;
    border-right: 1px solid var(--border) !important;
    min-width: 260px !important;
    max-width: 260px !important;
    transform: none !important;
    visibility: visible !important;
}
[data-testid="stSidebar"][aria-expanded="false"] {
    min-width: 260px !important;
    max-width: 260px !important;
    transform: none !important;
    visibility: visible !important;
    display: block !important;
}
/* Hide the collapse arrow button */
[data-testid="collapsedControl"] {
    display: none !important;
}
button[kind="header"] {
    display: none !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.6rem 1.1rem; }

.sb-logo {
    font-family: 'Lora', serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--accent);
    padding-bottom: 1.2rem;
    margin-bottom: 1.4rem;
    border-bottom: 1px solid var(--border);
}

.sb-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 1.3rem 0 0.6rem;
}

.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.65rem 0.85rem;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--r-sm);
    margin-bottom: 0.45rem;
}
.stat-row .sl { font-size: 0.8rem; color: var(--text-2); }
.stat-row .sv { font-family: 'Lora', serif; font-size: 1rem; font-weight: 700; color: var(--text); }
.sv.blue  { color: var(--accent);  }
.sv.green { color: var(--success); }

.diff-pill {
    display: inline-block;
    padding: 0.25rem 0.7rem;
    border-radius: 99px;
    font-size: 0.74rem;
    font-weight: 600;
    margin-top: 0.45rem;
}
.dp-beginner     { background: var(--success-lt); color: var(--success); }
.dp-intermediate { background: var(--warning-lt); color: var(--warning); }
.dp-advanced     { background: var(--danger-lt);  color: var(--danger);  }

.hist-row {
    display: flex;
    justify-content: space-between;
    gap: 0.5rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.8rem;
}
.hist-row .hq { color: var(--text-2); flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.hist-row .hs { font-weight: 700; flex-shrink: 0; }

/* ────── PAGE HEADER ────── */
.page-header {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.6rem;
    display: flex;
    align-items: center;
    gap: 1.1rem;
    box-shadow: var(--shadow-sm);
}
.ph-icon  { font-size: 2.2rem; flex-shrink: 0; }
.ph-title {
    font-family: 'Lora', serif;
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 0.15rem;
}
.ph-sub   { font-size: 0.86rem; color: var(--muted); margin: 0; }

/* ────── UPLOAD ZONE ────── */
.upload-box {
    background: var(--white);
    border: 2px dashed var(--border);
    border-radius: var(--r);
    padding: 2.6rem 2rem;
    text-align: center;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.upload-box:hover { border-color: var(--accent); }
.upload-box .ub-icon { font-size: 2rem; margin-bottom: 0.5rem; }
.upload-box h3 {
    font-family: 'Lora', serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 0.25rem;
}
.upload-box p { font-size: 0.81rem; color: var(--muted); margin: 0; }

/* ────── PROGRESS ────── */
.prog-wrap {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--r-sm);
    padding: 0.95rem 1.2rem;
    margin-bottom: 1.4rem;
    box-shadow: var(--shadow-sm);
}
.prog-meta {
    display: flex;
    justify-content: space-between;
    font-size: 0.76rem;
    font-weight: 600;
    color: var(--muted);
    margin-bottom: 0.55rem;
}
.prog-track {
    height: 5px;
    background: var(--border);
    border-radius: 99px;
    overflow: hidden;
}
.prog-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 99px;
    transition: width 0.4s ease;
}

/* ────── QUESTION CARD ────── */
.q-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: var(--r);
    padding: 1.5rem 1.7rem;
    margin-bottom: 1.3rem;
    box-shadow: var(--shadow-sm);
}
.q-meta {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.6rem;
}
.tag {
    background: var(--accent-lt);
    color: var(--accent);
    border-radius: 99px;
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    padding: 0.15rem 0.5rem;
    text-transform: uppercase;
}
.q-text {
    font-family: 'Lora', serif;
    font-size: 1.06rem;
    font-weight: 600;
    color: var(--text);
    line-height: 1.6;
    margin: 0;
}

/* ────── ANSWER LABEL ────── */
.ans-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 1.1rem 0 0.35rem;
}

/* ────── FEEDBACK CARD ────── */
.fb-card {
    background: #f0f4ff;
    border: 1px solid #c7d5fc;
    border-radius: var(--r);
    padding: 1.3rem 1.5rem;
    margin-top: 1.1rem;
}
.fb-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent);
    margin: 0 0 0.6rem;
}
.fb-body {
    font-size: 0.88rem;
    line-height: 1.7;
    color: var(--text-2);
    margin: 0;
}

/* ────── SCORE BADGE ────── */
.sc-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.32rem 0.85rem;
    border-radius: 99px;
    font-size: 0.8rem;
    font-weight: 700;
    margin-top: 0.8rem;
}
.sc-high { background: var(--success-lt); color: var(--success); }
.sc-mid  { background: var(--warning-lt); color: var(--warning); }
.sc-low  { background: var(--danger-lt);  color: var(--danger);  }

/* ────── REPORT CARD ────── */
.report-wrap {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 2rem 2.2rem;
    margin-top: 1rem;
    box-shadow: var(--shadow);
    text-align: center;
}
.rw-icon  { font-size: 2.8rem; margin-bottom: 0.4rem; }
.rw-score {
    font-family: 'Lora', serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
    margin: 0.2rem 0;
}
.rw-sub { font-size: 0.84rem; color: var(--muted); margin-bottom: 1.5rem; }

.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.9rem;
    margin-bottom: 1.3rem;
}
.metric-box {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--r-sm);
    padding: 0.9rem 0.7rem;
}
.mb-val {
    font-family: 'Lora', serif;
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--text);
}
.mb-lbl { font-size: 0.7rem; color: var(--muted); margin-top: 0.12rem; }

.verdict-box {
    border-radius: var(--r-sm);
    padding: 0.85rem 1.1rem;
    font-size: 0.87rem;
    font-weight: 500;
    line-height: 1.5;
}
.vd-ex { background: var(--success-lt); color: var(--success); }
.vd-go { background: var(--warning-lt); color: var(--warning); }
.vd-nd { background: var(--danger-lt);  color: var(--danger);  }

/* ────── BUTTONS ────── */
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--r-sm) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.87rem !important;
    font-weight: 600 !important;
    padding: 0.52rem 1.3rem !important;
    box-shadow: 0 2px 8px rgba(59,108,247,0.22) !important;
    transition: opacity 0.15s, transform 0.1s !important;
}
.stButton > button:hover  { opacity: 0.9 !important; transform: translateY(-1px); }
.stButton > button:active { transform: translateY(0) !important; }
.stButton > button:disabled { opacity: 0.38 !important; cursor: not-allowed !important; box-shadow: none !important; }

.btn-danger .stButton > button {
    background: transparent !important;
    color: var(--danger) !important;
    border: 1px solid #fca5a5 !important;
    box-shadow: none !important;
}
.btn-danger .stButton > button:hover { background: var(--danger-lt) !important; }

/* ────── INPUTS ────── */
textarea, [data-testid="stTextArea"] textarea {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: var(--r-sm) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.87rem !important;
    min-height: 115px !important;
}
textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(59,108,247,0.09) !important;
}
[data-testid="stSelectbox"] > div > div {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: var(--r-sm) !important;
    font-size: 0.87rem !important;
}
[data-testid="stAlert"] {
    border-radius: var(--r-sm) !important;
    font-size: 0.85rem !important;
}
hr { border-color: var(--border) !important; margin: 1.4rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#            SESSION STATE
# ─────────────────────────────────────────

for k, v in {
    "questions": [],
    "current_q": 0,
    "answered": False,
    "scores": [],
    "feedback_history": [],
    "history_log": [],
    "current_q_score": None,
    "current_q_feedback": None,
    "evaluated": False,
    "eval_preview_score": None,
    "eval_preview_feedback": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────
#            SIDEBAR
# ─────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sb-logo">🎯 InterviewAI</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-label">Settings</div>', unsafe_allow_html=True)
    difficulty = st.selectbox(
        "Difficulty Level",
        ["Beginner", "Intermediate", "Advanced"],
        help="Sets the complexity of generated questions"
    )
    dp_map = {
        "Beginner": "dp-beginner",
        "Intermediate": "dp-intermediate",
        "Advanced": "dp-advanced"
    }
    st.markdown(
        f'<span class="diff-pill {dp_map[difficulty]}">{difficulty}</span>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="sb-label">Session Stats</div>', unsafe_allow_html=True)

    total_q    = len(st.session_state.questions)
    answered_n = len(st.session_state.history_log)
    logged_scores = [e["score"] for e in st.session_state.history_log]
    avg_sc     = (
        round(sum(logged_scores) / len(logged_scores), 1)
        if logged_scores else "—"
    )

    st.markdown(f"""
    <div class="stat-row">
        <span class="sl">Total Questions</span>
        <span class="sv blue">{total_q}</span>
    </div>
    <div class="stat-row">
        <span class="sl">Answered</span>
        <span class="sv">{answered_n}</span>
    </div>
    <div class="stat-row">
        <span class="sl">Avg Score</span>
        <span class="sv green">{avg_sc}</span>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.history_log:
        st.markdown('<div class="sb-label">Score History</div>', unsafe_allow_html=True)
        for i, entry in enumerate(st.session_state.history_log):
            color = (
                "var(--success)" if entry["score"] >= 8
                else "var(--warning)" if entry["score"] >= 5
                else "var(--danger)"
            )
            short_q = entry["question"][:35] + "…" if len(entry["question"]) > 35 else entry["question"]
            st.markdown(f"""
            <div class="hist-row">
                <span class="hq">Q{i+1}. {short_q}</span>
                <span class="hs" style="color:{color}">{entry["score"]}/10</span>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#            INTRO PAGE
# ─────────────────────────────────────────
#            FILE UPLOAD
# ─────────────────────────────────────────

if len(st.session_state.questions) == 0:
    st.markdown("""
    <div class="upload-box">
        <div class="ub-icon">📄</div>
        <h3>Upload your study materials</h3>
        <p>PDF, DOCX, and TXT supported — multiple files accepted</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        with st.spinner("Processing your documents…"):
            text_data = ""
            for file in uploaded_files:
                text_data += extract_text(file)

            existing_questions = extract_questions(text_data)

            if len(existing_questions) >= 3:
                questions = existing_questions
                st.info("Questions detected in the document — using them directly.")
            else:
                chunks = create_chunks(text_data)
                store_vectors(chunks)
                docs = retrieve_docs("interview questions")
                questions = []
                for doc in docs:
                    q = generate_question(doc, difficulty)
                    questions.append(q)
                st.info(f"Generating {difficulty} level questions from your material.")

            questions = list(set(questions))
            random.shuffle(questions)

            st.session_state.questions           = questions
            st.session_state.current_q           = 0
            st.session_state.answered            = False
            st.session_state.scores              = []
            st.session_state.feedback_history    = []
            st.session_state.history_log         = []
            st.session_state.current_q_score       = None
            st.session_state.current_q_feedback    = None
            st.session_state.evaluated             = False
            st.session_state.eval_preview_score    = None
            st.session_state.eval_preview_feedback = None

        st.success("Documents processed — your interview is ready!")
        st.rerun()


# ─────────────────────────────────────────
#            PROGRESS BAR
# ─────────────────────────────────────────

if st.session_state.questions:
    total   = len(st.session_state.questions)
    current = st.session_state.current_q
    pct     = min(int((current / total) * 100), 100)

    st.markdown(f"""
    <div class="prog-wrap">
        <div class="prog-meta">
            <span>Progress</span>
            <span>{min(current, total)} of {total} questions</span>
        </div>
        <div class="prog-track">
            <div class="prog-fill" style="width:{pct}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
#            QUESTION DISPLAY
# ─────────────────────────────────────────

qs = st.session_state.questions
cq = st.session_state.current_q

if qs and cq < len(qs):
    question = qs[cq]

    st.markdown(f"""
    <div class="q-card">
        <div class="q-meta">
            Question {cq + 1} of {len(qs)}
            <span class="tag">{difficulty}</span>
        </div>
        <p class="q-text">{question}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="ans-label">Your Answer</div>', unsafe_allow_html=True)
    answer = st.text_area(
        "Answer",
        placeholder="Write your answer here — be as detailed as possible…",
        key=f"answer_box_{cq}",
        label_visibility="collapsed"
    )

    is_last_question = (cq == len(qs) - 1)
    next_label       = "Finish ✓" if is_last_question else "Next →"

    # ════════════════════════════════════════════
    # STAGE 1 — EVALUATE  (no evaluation yet)
    # ════════════════════════════════════════════
    if not st.session_state.answered and not st.session_state.evaluated:
        col_ev, _ = st.columns([1, 5])
        with col_ev:
            if st.button("🔍 Evaluate", use_container_width=True):
                if not answer.strip():
                    st.warning("Please write an answer before evaluating.")
                else:
                    with st.spinner("Evaluating your answer…"):
                        _fb = evaluate_answer(answer)
                    _sm = re.search(r"(?i)score[\s:*]*\(?(\d+)\)?", _fb)
                    st.session_state.eval_preview_score    = int(_sm.group(1)) if _sm else None
                    st.session_state.eval_preview_feedback = _fb
                    st.session_state.evaluated             = True
                    st.rerun()

    # ════════════════════════════════════════════
    # STAGE 2 — PREVIEW  (evaluated, not submitted)
    # ════════════════════════════════════════════
    elif st.session_state.evaluated and not st.session_state.answered:
        _sc = st.session_state.eval_preview_score
        _fb = st.session_state.eval_preview_feedback
        if _sc is not None:
            _cls  = "sc-high" if _sc >= 8 else "sc-mid" if _sc >= 5 else "sc-low"
            _icon = "✓" if _sc >= 8 else "~" if _sc >= 5 else "✗"
            _sh   = f'<div class="sc-badge {_cls}">{_icon} Score: {_sc} / 10</div>'
        else:
            _sh = ""
        st.markdown(f"""
        <div class="fb-card">
            <div class="fb-title">📋 Evaluation Preview — Review before submitting</div>
            <p class="fb-body">{_fb}</p>
            {_sh}
        </div>""", unsafe_allow_html=True)

        col_sub, col_next, _ = st.columns([1, 1, 4])
        with col_sub:
            if st.button("✅ Submit", use_container_width=True):
                _sc = st.session_state.eval_preview_score
                _fb = st.session_state.eval_preview_feedback
                # Use 0 as fallback score if evaluation failed to parse
                _final_score = _sc if _sc is not None else 0
                st.session_state.current_q_score    = _final_score
                st.session_state.current_q_feedback = _fb
                st.session_state.scores.append(_final_score)
                st.session_state.feedback_history.append(_fb)
                # ✅ Always log to history on Submit (score 0 if unparseable)
                if not any(e["q_index"] == cq for e in st.session_state.history_log):
                    st.session_state.history_log.append({
                        "q_index":  cq,
                        "question": qs[cq],
                        "score":    _final_score,
                    })
                st.session_state.answered  = True
                st.session_state.evaluated = False
                st.rerun()
        with col_next:
            st.button(next_label, disabled=True, use_container_width=True)

    # ════════════════════════════════════════════
    # STAGE 3 — SUBMITTED  (answered = True)
    # ════════════════════════════════════════════
    elif st.session_state.answered:
        _sc = st.session_state.current_q_score if st.session_state.current_q_score is not None else 0
        _fb = st.session_state.current_q_feedback
        _cls  = "sc-high" if _sc >= 8 else "sc-mid" if _sc >= 5 else "sc-low"
        _icon = "✓" if _sc >= 8 else "~" if _sc >= 5 else "✗"
        _sh   = f'<div class="sc-badge {_cls}">{_icon} Score: {_sc} / 10</div>'""
        st.markdown(f"""
        <div class="fb-card">
            <div class="fb-title">✅ Submitted — AI Feedback</div>
            <p class="fb-body">{_fb}</p>
            {_sh}
        </div>""", unsafe_allow_html=True)

        col_nxt, _ = st.columns([1, 5])
        with col_nxt:
            if st.button(f"{'🏁' if is_last_question else '➡️'} {next_label}", use_container_width=True):
                # Reset all flags for next question
                st.session_state.current_q             += 1
                st.session_state.answered               = False
                st.session_state.evaluated              = False
                st.session_state.current_q_score        = None
                st.session_state.current_q_feedback     = None
                st.session_state.eval_preview_score     = None
                st.session_state.eval_preview_feedback  = None
                st.rerun()


# ─────────────────────────────────────────
#            FINAL REPORT
# ─────────────────────────────────────────

if qs and cq >= len(qs):
    # Use only history_log entries as source of truth (only submitted+nexted questions)
    total_done    = len(st.session_state.history_log)
    logged_scores = [e["score"] for e in st.session_state.history_log]

    if logged_scores:
        avg  = sum(logged_scores) / len(logged_scores)
        high = max(logged_scores)
        low  = min(logged_scores)

        if avg >= 8:
            icon, vd_cls = "🏆", "vd-ex"
            msg = "Excellent performance! You've demonstrated strong command of the subject."
        elif avg >= 5:
            icon, vd_cls = "📈", "vd-go"
            msg = "Good attempt. A few areas need attention — keep practicing."
        else:
            icon, vd_cls = "📚", "vd-nd"
            msg = "Needs improvement. Review the key concepts and try again."

        st.markdown(f"""
        <div class="report-wrap">
            <div class="rw-icon">{icon}</div>
            <div class="rw-score">{round(avg, 1)} / 10</div>
            <div class="rw-sub">Average score across {total_done} question{"s" if total_done != 1 else ""}</div>
            <div class="metric-grid">
                <div class="metric-box">
                    <div class="mb-val">{total_done}</div>
                    <div class="mb-lbl">Answered</div>
                </div>
                <div class="metric-box">
                    <div class="mb-val" style="color:var(--success)">{high}</div>
                    <div class="mb-lbl">Best Score</div>
                </div>
                <div class="metric-box">
                    <div class="mb-val" style="color:var(--danger)">{low}</div>
                    <div class="mb-lbl">Lowest Score</div>
                </div>
            </div>
            <div class="verdict-box {vd_cls}">{msg}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────
#            RESTART
# ─────────────────────────────────────────

if qs:
    st.markdown("---")
    st.markdown('<div class="btn-danger">', unsafe_allow_html=True)
    if st.button("↺ Restart Interview"):
        for k, v in {
            "questions": [], "current_q": 0,
            "answered": False, "scores": [], "feedback_history": [],
            "history_log": [], "current_q_score": None, "current_q_feedback": None,
            "evaluated": False, "eval_preview_score": None, "eval_preview_feedback": None
        }.items():
            st.session_state[k] = v
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)