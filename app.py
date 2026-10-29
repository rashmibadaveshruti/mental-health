"""
app.py
MindBridge — Streamlit UI
Agentic AI for Mental Health Awareness & Suicide Prevention
Powered by IBM Granite · IBM watsonx.ai · IBM Cloud
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv

# Ensure the package is importable when running from the project root
sys.path.insert(0, os.path.dirname(__file__))

load_dotenv()

from core.orchestrator import MindBridgeOrchestrator, AgentResponse

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MindBridge — Mental Health Companion",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }
    .risk-badge-none     { background:#d1fae5; color:#065f46; padding:2px 10px; border-radius:20px; font-size:12px; font-weight:600; }
    .risk-badge-low      { background:#dbeafe; color:#1e40af; padding:2px 10px; border-radius:20px; font-size:12px; font-weight:600; }
    .risk-badge-medium   { background:#fef3c7; color:#92400e; padding:2px 10px; border-radius:20px; font-size:12px; font-weight:600; }
    .risk-badge-high     { background:#fee2e2; color:#991b1b; padding:2px 10px; border-radius:20px; font-size:12px; font-weight:600; }
    .risk-badge-critical { background:#7f1d1d; color:#ffffff; padding:2px 10px; border-radius:20px; font-size:12px; font-weight:600; }
    .alert-box { background:#fef2f2; border:1px solid #fca5a5; border-radius:8px; padding:12px; margin:8px 0; }
    .info-box  { background:#eff6ff; border:1px solid #93c5fd; border-radius:8px; padding:12px; margin:8px 0; }
    .tool-chip { background:#f3f4f6; border:1px solid #e5e7eb; border-radius:12px;
                 padding:2px 8px; font-size:11px; color:#374151; margin-right:4px; display:inline-block; }
    .mindbridge-header { text-align:center; padding: 10px 0 4px; }
    .mindbridge-header h1 { font-size:2rem; color:#1e3a5f; margin:0; }
    .mindbridge-header p  { color:#6b7280; margin:0; font-size:0.95rem; }
    .disclaimer-banner { background:#fffbeb; border:1px solid #fbbf24; border-radius:8px;
                         padding:10px 14px; font-size:0.82rem; color:#78350f; margin-bottom:10px; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────
def init_session():
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = MindBridgeOrchestrator(
            country=st.session_state.get("country", "india"),
            consent_given=st.session_state.get("consent", False),
            api_key=os.getenv("WATSONX_API_KEY"),
            project_id=os.getenv("WATSONX_PROJECT_ID"),
            region=os.getenv("WATSONX_REGION", "us-south"),
        )
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "responses" not in st.session_state:
        st.session_state.responses = []
    if "consent_given" not in st.session_state:
        st.session_state.consent_given = False
    if "country" not in st.session_state:
        st.session_state.country = "india"
    if "show_diagnostics" not in st.session_state:
        st.session_state.show_diagnostics = False


init_session()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 MindBridge")
    st.markdown("*Agentic AI Mental Health Companion*")
    st.markdown("---")

    # Country selector
    country_options = {"🇮🇳 India": "india", "🇺🇸 USA": "usa",
                       "🇬🇧 UK": "uk", "🇦🇺 Australia": "australia"}
    selected_country_label = st.selectbox(
        "📍 Your Location (for helplines)",
        list(country_options.keys()),
        index=0,
    )
    new_country = country_options[selected_country_label]
    if new_country != st.session_state.country:
        st.session_state.country = new_country
        st.session_state.orchestrator.country = new_country

    st.markdown("---")

    # Consent toggle
    consent = st.toggle(
        "📊 Enable Risk Monitoring",
        value=st.session_state.consent_given,
        help="Allows MindBridge to track distress patterns and send proactive alerts.",
    )
    if consent != st.session_state.consent_given:
        st.session_state.consent_given = consent
        st.session_state.orchestrator.set_consent(consent)

    # Diagnostics toggle
    st.session_state.show_diagnostics = st.toggle(
        "🔬 Show Agent Diagnostics",
        value=st.session_state.show_diagnostics,
        help="Display risk scores, tools used, and RAG signals for each turn.",
    )

    st.markdown("---")

    # Crisis helplines quick reference
    st.markdown("### 🆘 Crisis Helplines")
    country_helplines = {
        "india":     [("iCall", "9152987821"), ("Vandrevala", "1860-2662-345")],
        "usa":       [("988 Lifeline", "988"), ("Crisis Text", "Text HOME to 741741")],
        "uk":        [("Samaritans", "116 123"), ("CALM", "0800 58 58 58")],
        "australia": [("Lifeline", "13 11 14"), ("Beyond Blue", "1300 22 4636")],
    }
    for name, number in country_helplines.get(st.session_state.country, []):
        st.markdown(f"📞 **{name}**: `{number}`")

    st.markdown("---")

    # Session controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.responses = []
            st.session_state.orchestrator.reset_session()
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px; color:#9ca3af; text-align:center;'>"
        "Powered by<br><b>IBM Granite · watsonx.ai · IBM Cloud</b>"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Main header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="mindbridge-header">
    <h1>🧠 MindBridge</h1>
    <p>Agentic AI for Mental Health Awareness & Suicide Prevention</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-banner">
⚕️ <strong>Important Disclaimer</strong>: MindBridge is an AI companion and is <strong>NOT a substitute</strong>
for professional medical advice, diagnosis, or treatment.
If you or someone you know is in immediate danger, please call <strong>emergency services (911 / 112 / 100)</strong>
or a crisis helpline immediately.
</div>
""", unsafe_allow_html=True)


# ── Risk Level Badge helper ───────────────────────────────────────────────────
def risk_badge(level: str) -> str:
    labels = {
        "none":     "✅ No distress",
        "low":      "🔵 Mild",
        "medium":   "🟡 Moderate",
        "high":     "🔴 High",
        "critical": "🚨 Critical",
    }
    return f'<span class="risk-badge-{level}">{labels.get(level, level.title())}</span>'


# ── Welcome message ───────────────────────────────────────────────────────────
if not st.session_state.messages:
    welcome = (
        "Hello, I'm **MindBridge** 💙 — your compassionate AI mental health companion.\n\n"
        "I'm here to listen, support, and guide you toward wellness resources. "
        "You can share how you're feeling, ask about coping strategies, or just talk.\n\n"
        "**How are you feeling today?**\n\n"
        "_Remember: I'm an AI companion — for clinical support, please consult a licensed "
        "mental health professional. Crisis helplines are available in the sidebar._"
    )
    with st.chat_message("assistant", avatar="🧠"):
        st.markdown(welcome)


# ── Display conversation history ──────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    avatar = "🧠" if msg["role"] == "assistant" else "🧑"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

    # Show diagnostics for assistant turns
    if (msg["role"] == "assistant" and st.session_state.show_diagnostics
            and i // 2 < len(st.session_state.responses)):
        resp_idx = i // 2
        if resp_idx < len(st.session_state.responses):
            resp: AgentResponse = st.session_state.responses[resp_idx]
            with st.expander(f"🔬 Agent Diagnostics — Turn {resp.turn_index}", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Risk Level**")
                    st.markdown(risk_badge(resp.risk_level), unsafe_allow_html=True)
                with col2:
                    st.metric("Risk Score", f"{resp.risk_score:.2f}")
                with col3:
                    st.metric("Trend", resp.trend.title())

                st.markdown("**Tools Used:**")
                tools_html = " ".join(f'<span class="tool-chip">⚙️ {t}</span>' for t in resp.tools_used)
                st.markdown(tools_html, unsafe_allow_html=True)

                if resp.distress_signals:
                    st.markdown("**Detected Signals:**")
                    for sig in resp.distress_signals[:5]:
                        st.markdown(f"  - `{sig}`")

                if resp.alert:
                    st.markdown(
                        f'<div class="alert-box">🚨 <b>Alert ({resp.alert.level.upper()})</b>: '
                        f'{resp.alert.message}<br>'
                        f'<b>Recommended Actions:</b> {", ".join(resp.alert.actions[:2])}</div>',
                        unsafe_allow_html=True,
                    )


# ── Show critical banner if last response was critical ────────────────────────
if st.session_state.responses:
    last_resp: AgentResponse = st.session_state.responses[-1]
    if last_resp.risk_level == "critical":
        st.error(
            "🚨 **Crisis alert detected.** Please reach out to a helpline immediately. "
            "Helpline numbers are shown in the sidebar.",
            icon="🚨",
        )
    elif last_resp.risk_level == "high" and last_resp.alert:
        st.warning(
            "⚠️ **High distress detected.** Professional support is recommended. "
            "See sidebar for helpline numbers.",
            icon="⚠️",
        )


# ── Session stats ─────────────────────────────────────────────────────────────
if st.session_state.responses and st.session_state.show_diagnostics:
    with st.expander("📊 Session Risk Summary", expanded=False):
        summary = st.session_state.orchestrator.get_session_summary()
        risk_summary = summary.get("risk_summary", {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Turns", risk_summary.get("turns", 0))
        col2.metric("Peak Risk", risk_summary.get("peak_risk", "none").title())
        col3.metric("Avg Score", risk_summary.get("avg_score", 0.0))
        col4.metric("Alerts", risk_summary.get("alerts", 0))


# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Share how you're feeling, or ask anything…"):
    # Display user message
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Run agentic turn
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("MindBridge is thinking…"):
            response: AgentResponse = st.session_state.orchestrator.run(prompt)

        st.markdown(response.message)

        # Inline risk indicator
        if response.risk_level != "none":
            st.markdown(
                f"<div style='margin-top:6px;'>{risk_badge(response.risk_level)}"
                f" &nbsp; <span style='font-size:11px;color:#9ca3af;'>"
                f"Trend: {response.trend}</span></div>",
                unsafe_allow_html=True,
            )

    st.session_state.messages.append({"role": "assistant", "content": response.message})
    st.session_state.responses.append(response)
    st.rerun()
