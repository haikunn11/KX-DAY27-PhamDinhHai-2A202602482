"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path
from exercises.exercise_4_audit import build_graph

load_dotenv()


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review Agent", layout="wide")
st.title("HITL PR Review Agent")


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
async def fetch_recent_sessions():
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id,
                   pr_url,
                   MIN(timestamp)        AS started,
                   MAX(timestamp)        AS last_event,
                   MAX(risk_level)       AS worst_risk,
                   COUNT(*)              AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT 15
            """
        ) as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]


with st.sidebar:
    st.header("Recent Sessions")
    try:
        sessions = asyncio.run(fetch_recent_sessions())
        if sessions:
            for s in sessions:
                short_id = s["thread_id"][:8]
                risk = s["worst_risk"]
                badge = "🔴" if risk == "high" else ("🟡" if risk == "med" else "🟢")
                label = f"{badge} `{short_id}` ({s['events']} events)"
                if st.button(label, key=f"session_{s['thread_id']}", help=f"PR: {s['pr_url']}"):
                    st.session_state.pr_url = s["pr_url"]
                    st.session_state.thread_id = s["thread_id"]
                    st.session_state.interrupt_payload = None
                    st.session_state.final = None
                    st.rerun()
        else:
            st.caption("No recent sessions in audit trail yet.")
    except Exception as e:
        st.caption(f"Error loading sessions: {e}")


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL",
        value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run Review", type="primary")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.subheader(f"Approval Requested — Confidence: {conf:.0%}")
    st.caption(payload.get("confidence_reasoning", ""))
    st.markdown(payload.get("summary", ""))

    comments = payload.get("comments", [])
    if comments:
        st.write("### Proposed Comments")
        for c in comments:
            st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("Diff Preview", expanded=False):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Reviewer Feedback (optional)", key="approval_feedback")
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary", use_container_width=True):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject", use_container_width=True):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit", use_container_width=True):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"⚠️ Strong Escalation — Confidence: {conf:.0%}")
    st.caption(payload.get("confidence_reasoning", ""))
    if payload.get("risk_factors"):
        st.error("Identified Risks: " + ", ".join(payload["risk_factors"]))
    st.markdown(payload.get("summary", ""))

    with st.form("escalation"):
        st.write("### Clarification Questions")
        answers: dict[str, str] = {}
        for i, q in enumerate(payload.get("questions", [])):
            answers[q] = st.text_input(f"Q{i+1}: {q}", key=f"escalate_q_{i}")
        form_submitted = st.form_submit_button("Submit Answers & Synthesize Review", type="primary")
        if form_submitted:
            return answers
    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        if resume_value is None:
            result = await app.ainvoke(
                {"pr_url": pr_url, "thread_id": thread_id}, cfg
            )
        else:
            result = await app.ainvoke(Command(resume=resume_value), cfg)
        return result


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None

    with st.spinner("Fetching PR & analyzing diff with LLM..."):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload.get("kind")
    answer = (
        render_approval_card(payload)
        if kind == "approval_request"
        else render_escalation_card(payload)
    )
    if answer is not None:
        with st.spinner("Resuming agent workflow..."):
            result = asyncio.run(
                run_graph(
                    st.session_state.pr_url,
                    st.session_state.thread_id,
                    resume_value=answer,
                )
            )
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    if action.startswith("auto") or action.startswith("committed"):
        st.success(f"✓ Review committed ({action}) — comment posted to {st.session_state.pr_url}")
    elif action == "rejected":
        st.warning("Review rejected by human — no comment posted to GitHub.")
    else:
        st.info(f"Final outcome: {action}")

    if "analysis" in final:
        a = final["analysis"]
        with st.expander("Review Details", expanded=True):
            st.markdown(f"**Confidence:** {a.confidence:.0%}")
            st.markdown(f"**Summary:** {a.summary}")
            if a.comments:
                st.markdown("**Comments:**")
                for c in a.comments:
                    st.markdown(f"- `[{c.severity}]` `{c.file}:{c.line or '?'}` — {c.body}")

    st.caption(
        f"Thread ID: `{st.session_state.thread_id}` · Replay CLI: "
        f"`python -m audit.replay --thread {st.session_state.thread_id}`"
    )
