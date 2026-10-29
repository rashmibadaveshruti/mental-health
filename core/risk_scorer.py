"""
risk_scorer.py
Risk prediction, trend analysis, and alert system for MindBridge.

Tracks risk scores across conversation turns to detect escalating distress,
predict stress peaks, and issue timely alerts — all with user consent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class RiskSnapshot:
    """A single risk measurement at a point in time."""
    timestamp:  float
    risk_level: str
    risk_score: float
    signals:    list[str]
    turn_index: int


@dataclass
class RiskAlert:
    """An alert issued when risk thresholds are crossed."""
    level:      str    # "warning" | "high_risk" | "critical"
    message:    str
    actions:    list[str]   # recommended actions
    timestamp:  float = field(default_factory=time.time)


@dataclass
class RiskReport:
    """Full risk analysis report for the current session."""
    current_risk:       str
    current_score:      float
    trend:              str    # "stable" | "escalating" | "de-escalating" | "volatile"
    peak_score:         float
    session_alerts:     list[RiskAlert]
    recommendation:     str
    alert_issued:       Optional[RiskAlert]
    escalation_needed:  bool


# ── Risk Scorer ───────────────────────────────────────────────────────────────

class RiskScorer:
    """
    Tracks distress risk across conversation turns, identifies trends,
    predicts escalation, and generates contextual alerts.
    """

    # Thresholds
    HIGH_RISK_THRESHOLD    = 0.65
    CRITICAL_THRESHOLD     = 0.90
    ESCALATION_SLOPE       = 0.15   # risk increase over last 3 turns triggering alert
    VOLATILITY_THRESHOLD   = 0.30   # score variance triggering "volatile" label

    def __init__(self, consent_given: bool = False):
        self._history:     list[RiskSnapshot] = []
        self._alerts:      list[RiskAlert]    = []
        self._consent:     bool               = consent_given
        self._turn_index:  int                = 0

    # ── Public API ───────────────────────────────────────────────────────────

    def set_consent(self, consent: bool) -> None:
        """Update user consent for risk monitoring and alerts."""
        self._consent = consent

    def record(self, risk_level: str, risk_score: float, signals: list[str]) -> RiskReport:
        """
        Record a new risk measurement and return a full risk report.

        Args:
            risk_level: "none" | "low" | "medium" | "high" | "critical"
            risk_score: float 0.0–1.0
            signals:    list of detected signal strings

        Returns:
            RiskReport with trend analysis and any new alerts.
        """
        self._turn_index += 1
        snapshot = RiskSnapshot(
            timestamp=time.time(),
            risk_level=risk_level,
            risk_score=risk_score,
            signals=signals,
            turn_index=self._turn_index,
        )
        self._history.append(snapshot)

        trend       = self._compute_trend()
        peak_score  = max((s.risk_score for s in self._history), default=0.0)
        alert       = self._evaluate_alerts(risk_level, risk_score, trend)

        if alert:
            self._alerts.append(alert)

        return RiskReport(
            current_risk=risk_level,
            current_score=risk_score,
            trend=trend,
            peak_score=peak_score,
            session_alerts=list(self._alerts),
            recommendation=self._build_recommendation(risk_level, trend),
            alert_issued=alert,
            escalation_needed=(risk_level in ("high", "critical") or
                                (trend == "escalating" and risk_score >= self.HIGH_RISK_THRESHOLD)),
        )

    def get_session_summary(self) -> dict:
        """Return a summary of the current session's risk trajectory."""
        if not self._history:
            return {"turns": 0, "peak_risk": "none", "trend": "stable", "alerts": 0}
        scores = [s.risk_score for s in self._history]
        return {
            "turns":      self._turn_index,
            "peak_risk":  max((s.risk_level for s in self._history),
                              key=lambda l: ["none","low","medium","high","critical"].index(l)),
            "peak_score": round(max(scores), 3),
            "avg_score":  round(sum(scores) / len(scores), 3),
            "trend":      self._compute_trend(),
            "alerts":     len(self._alerts),
        }

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _compute_trend(self) -> str:
        """Analyse the last N snapshots to determine trend direction."""
        n = len(self._history)
        if n < 2:
            return "stable"

        recent = [s.risk_score for s in self._history[-4:]]

        # Slope (simple linear trend over last 3 points) — check first
        if len(recent) >= 3:
            slope = (recent[-1] - recent[-3]) / 2
            if slope >= self.ESCALATION_SLOPE:
                return "escalating"
            if slope <= -self.ESCALATION_SLOPE:
                return "de-escalating"

        # Volatility — only when slope is flat but range is wide
        if n >= 4:
            variance = max(recent) - min(recent)
            if variance >= self.VOLATILITY_THRESHOLD:
                return "volatile"

        return "stable"

    def _evaluate_alerts(
        self, risk_level: str, risk_score: float, trend: str
    ) -> Optional[RiskAlert]:
        """Check whether a new alert should be issued."""
        # Suppress if no consent (except critical — always alert)
        if not self._consent and risk_level != "critical":
            return None

        # Avoid duplicate alerts on consecutive identical levels
        if self._alerts:
            last = self._alerts[-1]
            if last.level == risk_level and (time.time() - last.timestamp) < 120:
                return None  # don't repeat within 2 minutes

        if risk_level == "critical":
            return RiskAlert(
                level="critical",
                message="🚨 CRITICAL: Immediate crisis signals detected in conversation.",
                actions=[
                    "Display crisis helpline numbers immediately",
                    "Offer to connect to a live counselor",
                    "Provide grounding technique",
                    "Do NOT leave the conversation",
                ],
            )

        if risk_level == "high" or (trend == "escalating" and risk_score >= self.HIGH_RISK_THRESHOLD):
            return RiskAlert(
                level="high_risk",
                message="⚠️ High distress level detected — professional support recommended.",
                actions=[
                    "Recommend therapist or counselor referral",
                    "Share helpline numbers",
                    "Offer coping strategies",
                    "Increase empathy in responses",
                ],
            )

        if trend == "escalating" and risk_score >= 0.35:
            return RiskAlert(
                level="warning",
                message="📈 Distress level is increasing — preventive support recommended.",
                actions=[
                    "Introduce mindfulness or breathing exercise",
                    "Validate emotions and check in",
                    "Offer self-care resources",
                ],
            )

        return None

    def _build_recommendation(self, risk_level: str, trend: str) -> str:
        """Build a recommendation string for the orchestrator."""
        rec_map = {
            "critical": (
                "IMMEDIATE ACTION REQUIRED: Display all crisis helplines. "
                "Provide grounding technique. Offer connection to live support."
            ),
            "high": (
                "Respond with high empathy. Share professional referral options. "
                "Provide crisis helpline numbers. Do not dismiss or minimize."
            ),
            "medium": (
                "Offer emotional validation and coping strategies. "
                "Gently suggest professional support if distress persists."
            ),
            "low": (
                "Provide supportive listening. Share self-care tips and "
                "psychoeducation resources."
            ),
            "none": (
                "Continue supportive conversation. "
                "Promote wellness and awareness."
            ),
        }
        base = rec_map.get(risk_level, "Continue monitoring.")
        if trend == "escalating":
            base += " Note: distress is escalating — increase attentiveness."
        elif trend == "de-escalating":
            base += " Positive sign: distress appears to be decreasing."
        return base
