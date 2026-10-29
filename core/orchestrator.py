"""
orchestrator.py
Agentic AI Orchestrator for MindBridge.

Implements the core agentic loop:
  1. Receive user input
  2. Run safety filter (pre-check)
  3. Run RAG distress analysis
  4. Record risk score and evaluate alerts
  5. Decide which tools to invoke (resources, crisis connector, or plain empathy)
  6. Build enriched prompt context
  7. Call IBM Granite via watsonx.ai
  8. Post-process output (safety filter + disclaimer)
  9. Return structured AgentResponse

The orchestrator is stateful per-session and tracks conversation history.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

from .granite_client    import GraniteClient
from .rag_engine        import RAGEngine, DistressAnalysis
from .risk_scorer       import RiskScorer, RiskReport, RiskAlert
from .resources         import ResourceRecommender
from .crisis_connector  import CrisisConnector
try:
    from ..utils.safety_filter import SafetyFilter          # package import
except ImportError:
    from utils.safety_filter import SafetyFilter            # direct / pytest run


# ── Agent response ────────────────────────────────────────────────────────────

@dataclass
class AgentResponse:
    """Full structured response from one agent turn."""
    message:          str               # final response shown to the user
    risk_level:       str               # "none" | "low" | "medium" | "high" | "critical"
    risk_score:       float
    trend:            str
    alert:            Optional[RiskAlert]
    escalation_needed: bool
    tools_used:       list[str]         # which internal tools were invoked
    distress_signals: list[str]
    turn_index:       int
    timestamp:        float = field(default_factory=time.time)


# ── Tool definitions (agentic tool registry) ──────────────────────────────────

AVAILABLE_TOOLS = {
    "rag_analyze":          "Analyze user input for distress signals using RAG",
    "risk_record":          "Record risk score and compute session trend",
    "resource_recommend":   "Retrieve relevant mental health resources",
    "crisis_connect":       "Generate crisis helpline message and escalation",
    "safety_boundary":      "Return safe boundary response for harmful-method queries",
    "empathetic_chat":      "Generate empathetic response via IBM Granite",
}


# ── Orchestrator ──────────────────────────────────────────────────────────────

class MindBridgeOrchestrator:
    """
    The central Agentic AI orchestrator for MindBridge.

    Each call to `run()` executes one full agentic turn:
    observe → think → act (tool selection) → respond.
    """

    def __init__(
        self,
        country: str = "india",
        consent_given: bool = False,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
    ):
        self.country = country

        # Initialise all components
        self._granite   = GraniteClient(api_key=api_key, project_id=project_id, region=region)
        self._rag       = RAGEngine()
        self._risk      = RiskScorer(consent_given=consent_given)
        self._resources = ResourceRecommender()
        self._crisis    = CrisisConnector()
        self._safety    = SafetyFilter()

        # Session state
        self._history:     list[dict]  = []     # conversation history
        self._turn_index:  int         = 0
        self._consent:     bool        = consent_given

    # ── Public API ────────────────────────────────────────────────────────────

    def set_consent(self, consent: bool) -> None:
        """Update user consent for monitoring."""
        self._consent = consent
        self._risk.set_consent(consent)

    def run(self, user_message: str) -> AgentResponse:
        """
        Execute one full agentic turn.

        Args:
            user_message: The user's latest input text.

        Returns:
            AgentResponse with the final message and full diagnostic data.
        """
        self._turn_index += 1
        tools_used: list[str] = []

        # ── Step 1: Safety pre-filter ─────────────────────────────────────
        if self._safety.is_harmful_request(user_message):
            tools_used.append("safety_boundary")
            boundary_msg = self._crisis.get_boundary_response(self.country)
            self._append_history("user", user_message)
            self._append_history("assistant", boundary_msg)
            return AgentResponse(
                message=boundary_msg,
                risk_level="critical",
                risk_score=1.0,
                trend="critical",
                alert=None,
                escalation_needed=True,
                tools_used=tools_used,
                distress_signals=["harmful_method_request"],
                turn_index=self._turn_index,
            )

        # ── Step 2: RAG distress analysis ─────────────────────────────────
        tools_used.append("rag_analyze")
        analysis: DistressAnalysis = self._rag.analyze(
            user_message, self._history
        )

        # ── Step 3: Risk scoring ──────────────────────────────────────────
        tools_used.append("risk_record")
        report: RiskReport = self._risk.record(
            analysis.risk_level,
            analysis.risk_score,
            analysis.detected_signals,
        )

        # ── Step 4: Agentic decision — which tools to invoke? ─────────────
        extra_context_parts: list[str] = []

        # Always inject RAG knowledge-base context
        kb_context = self._rag.retrieve_context_string(user_message, top_k=2)
        if kb_context:
            extra_context_parts.append(
                f"[Relevant knowledge for your response]:\n{kb_context}"
            )

        # Crisis / high-risk → inject crisis message template + helplines
        if analysis.risk_level in ("critical", "high"):
            tools_used.append("crisis_connect")
            crisis_msg = self._crisis.get_crisis_message(
                analysis.risk_level, self.country
            )
            extra_context_parts.append(
                f"[Crisis response template — use as a basis for your response]:\n{crisis_msg}"
            )
        else:
            # Medium/low/none → inject relevant coping resources
            tools_used.append("resource_recommend")
            resource_context = self._resources.format_for_prompt(
                analysis.risk_level, count=1
            )
            if resource_context:
                extra_context_parts.append(resource_context)

        # Inject risk recommendation for the model
        extra_context_parts.append(
            f"[Agent guidance — follow this in your response]:\n{report.recommendation}"
        )

        extra_context = "\n\n".join(extra_context_parts)

        # ── Step 5: Generate response via IBM Granite ─────────────────────
        tools_used.append("empathetic_chat")
        raw_response = self._granite.chat(
            user_message=user_message,
            conversation_history=self._history,
            extra_context=extra_context,
            max_new_tokens=512,
        )

        # ── Step 6: Post-process (safety filter + disclaimer) ─────────────
        safe_response = self._safety.sanitize_output(raw_response)

        # Add disclaimer for medium+ risk or if not already present
        if analysis.risk_level in ("medium", "high", "critical") or self._turn_index == 1:
            final_response = self._safety.add_disclaimer(safe_response)
        else:
            final_response = safe_response

        # ── Step 7: Update conversation history ───────────────────────────
        self._append_history("user", user_message)
        self._append_history("assistant", final_response)

        return AgentResponse(
            message=final_response,
            risk_level=analysis.risk_level,
            risk_score=analysis.risk_score,
            trend=report.trend,
            alert=report.alert_issued,
            escalation_needed=report.escalation_needed,
            tools_used=tools_used,
            distress_signals=analysis.detected_signals,
            turn_index=self._turn_index,
        )

    def get_session_summary(self) -> dict:
        """Return a summary of the current session."""
        return {
            "turns": self._turn_index,
            "risk_summary": self._risk.get_session_summary(),
            "history_length": len(self._history),
        }

    def reset_session(self) -> None:
        """Clear session state — use when starting a new conversation."""
        self._history    = []
        self._turn_index = 0
        self._risk       = RiskScorer(consent_given=self._consent)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _append_history(self, role: str, content: str) -> None:
        """Append a turn to conversation history (capped at 20 turns for memory)."""
        self._history.append({"role": role, "content": content})
        if len(self._history) > 20:
            # Keep system context by dropping oldest non-critical turns
            self._history = self._history[-20:]
