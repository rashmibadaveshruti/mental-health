"""
test_mindbridge.py
Unit and integration tests for MindBridge core modules.
Run: python -m pytest test_mindbridge.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from core.rag_engine       import RAGEngine, DistressAnalysis
from core.risk_scorer      import RiskScorer
from core.resources        import ResourceRecommender
from core.crisis_connector import CrisisConnector
from core.granite_client   import GraniteClient
from core.orchestrator     import MindBridgeOrchestrator
from utils.safety_filter   import SafetyFilter


# ── RAG Engine Tests ──────────────────────────────────────────────────────────

class TestRAGEngine:
    def setup_method(self):
        self.rag = RAGEngine()

    def test_loads_knowledge_base(self):
        assert len(self.rag._kb) > 0

    def test_no_distress_in_neutral_message(self):
        result = self.rag.analyze("Hello, how are you?")
        assert result.risk_level == "none"
        assert result.risk_score == 0.0

    def test_low_distress_in_mild_message(self):
        result = self.rag.analyze("I've been feeling a bit stressed lately")
        assert result.risk_level in ("low", "medium")

    def test_medium_distress_detected(self):
        result = self.rag.analyze("I feel so depressed and can't stop crying")
        assert result.risk_level in ("medium", "high")

    def test_high_distress_detected(self):
        result = self.rag.analyze("I feel completely hopeless and nothing will change")
        assert result.risk_level in ("high", "critical")
        assert result.risk_score >= 0.65

    def test_critical_distress_detected(self):
        result = self.rag.analyze("I want to end my life")
        assert result.risk_level == "critical"
        assert result.risk_score == 1.0
        assert result.needs_escalation is True

    def test_retrieves_relevant_chunks(self):
        result = self.rag.analyze("I'm struggling with anxiety and panic attacks")
        assert len(result.retrieved_context) > 0
        # At least one chunk should be relevant to anxiety
        categories = [c.category for c in result.retrieved_context]
        assert any(cat in ("anxiety", "stress", "coping") for cat in categories)

    def test_retrieve_context_string(self):
        ctx = self.rag.retrieve_context_string("feeling depressed and sad")
        assert isinstance(ctx, str)
        assert len(ctx) > 0


# ── Risk Scorer Tests ─────────────────────────────────────────────────────────

class TestRiskScorer:
    def setup_method(self):
        self.scorer = RiskScorer(consent_given=True)

    def test_initial_state(self):
        summary = self.scorer.get_session_summary()
        assert summary["turns"] == 0

    def test_records_single_snapshot(self):
        report = self.scorer.record("low", 0.2, ["low:general_distress:stressed"])
        assert report.current_risk == "low"
        assert report.current_score == 0.2
        assert report.trend == "stable"

    def test_stable_trend_on_consistent_scores(self):
        for _ in range(4):
            report = self.scorer.record("low", 0.2, [])
        assert report.trend == "stable"

    def test_escalating_trend_detected(self):
        scores = [0.1, 0.2, 0.3, 0.5, 0.7]
        for s in scores:
            report = self.scorer.record("medium", s, [])
        assert report.trend == "escalating"

    def test_high_risk_alert_issued(self):
        report = self.scorer.record("high", 0.8, ["high:hopelessness:hopeless"])
        assert report.alert_issued is not None
        assert report.alert_issued.level == "high_risk"
        assert report.escalation_needed is True

    def test_critical_alert_always_fires(self):
        scorer_no_consent = RiskScorer(consent_given=False)
        report = scorer_no_consent.record("critical", 1.0, ["critical:suicidal_ideation:want to die"])
        assert report.alert_issued is not None
        assert report.alert_issued.level == "critical"

    def test_session_summary(self):
        self.scorer.record("none", 0.0, [])
        self.scorer.record("low", 0.2, [])
        self.scorer.record("medium", 0.45, [])
        summary = self.scorer.get_session_summary()
        assert summary["turns"] == 3
        assert summary["peak_risk"] == "medium"


# ── Safety Filter Tests ───────────────────────────────────────────────────────

class TestSafetyFilter:
    def setup_method(self):
        self.sf = SafetyFilter()

    def test_allows_normal_message(self):
        assert self.sf.is_harmful_request("I'm feeling anxious today") is False

    def test_detects_method_request(self):
        assert self.sf.is_harmful_request("how do I commit suicide") is True

    def test_detects_overdose_query(self):
        assert self.sf.is_harmful_request("what pills to take to die") is True

    def test_sanitize_clean_output(self):
        clean = "I'm here to support you and listen."
        assert self.sf.sanitize_output(clean) == clean

    def test_adds_disclaimer_when_missing(self):
        text = "I hear you, that sounds really hard."
        result = self.sf.add_disclaimer(text)
        assert "not a substitute" in result.lower()

    def test_does_not_duplicate_disclaimer(self):
        text = "I hear you. ⚕️ Disclaimer: I am not a substitute for professional advice."
        result = self.sf.add_disclaimer(text)
        assert result.count("not a substitute") == 1


# ── Resource Recommender Tests ────────────────────────────────────────────────

class TestResourceRecommender:
    def setup_method(self):
        self.rec = ResourceRecommender()

    def test_returns_resources_for_each_level(self):
        for level in ("none", "low", "medium", "high", "critical"):
            resources = self.rec.get_resources(level, count=2)
            assert len(resources) >= 1

    def test_format_for_prompt_returns_string(self):
        result = self.rec.format_for_prompt("medium", count=1)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_mindfulness_tip_is_string(self):
        tip = self.rec.get_mindfulness_tip()
        assert isinstance(tip, str)
        assert len(tip) > 10


# ── Crisis Connector Tests ────────────────────────────────────────────────────

class TestCrisisConnector:
    def setup_method(self):
        self.connector = CrisisConnector()

    def test_loads_helplines(self):
        lines = self.connector.get_helplines("india")
        assert len(lines) > 0

    def test_international_fallback_included(self):
        lines = self.connector.get_helplines("india")
        countries = [h.country for h in lines]
        assert "international" in countries

    def test_crisis_message_contains_helplines(self):
        msg = self.connector.get_crisis_message("critical", "india")
        assert "9152987821" in msg or "iCall" in msg

    def test_boundary_response_safe(self):
        resp = self.connector.get_boundary_response("india")
        assert "crisis" in resp.lower() or "helpline" in resp.lower()
        # Must NOT contain harmful method information
        assert "overdose" not in resp.lower()
        assert "method" not in resp.lower() or "safe" in resp.lower()


# ── Orchestrator Integration Tests ────────────────────────────────────────────

class TestOrchestrator:
    def setup_method(self):
        # No API key → runs in demo mode (tests orchestration logic without IBM Cloud)
        self.agent = MindBridgeOrchestrator(
            country="india",
            consent_given=True,
            api_key="",
            project_id="",
        )

    def test_neutral_message_completes(self):
        resp = self.agent.run("Hello, how are you?")
        assert resp.message
        assert resp.risk_level == "none"
        assert resp.turn_index == 1

    def test_distress_message_triggers_resources(self):
        resp = self.agent.run("I've been feeling very stressed and anxious")
        assert resp.message
        assert resp.risk_level in ("low", "medium", "high")
        assert "resource_recommend" in resp.tools_used or "crisis_connect" in resp.tools_used

    def test_critical_message_uses_crisis_connector(self):
        resp = self.agent.run("I want to end my life")
        assert resp.risk_level == "critical"
        assert resp.escalation_needed is True
        # Safety boundary fires first
        assert "safety_boundary" in resp.tools_used or "crisis_connect" in resp.tools_used

    def test_harmful_request_blocked(self):
        resp = self.agent.run("how do I commit suicide")
        assert resp.risk_level == "critical"
        assert "safety_boundary" in resp.tools_used
        # Response should NOT contain harmful information
        assert "how to" not in resp.message.lower() or "support" in resp.message.lower()

    def test_history_tracked(self):
        self.agent.run("I'm feeling okay today")
        self.agent.run("Actually I've been really down lately")
        summary = self.agent.get_session_summary()
        assert summary["turns"] == 2

    def test_session_reset(self):
        self.agent.run("Hello")
        self.agent.reset_session()
        summary = self.agent.get_session_summary()
        assert summary["turns"] == 0

    def test_escalating_trend_detected_across_turns(self):
        messages = [
            "I've been a bit tired",
            "I'm feeling really down and hopeless",
            "I just feel like I can't go on anymore",
        ]
        responses = [self.agent.run(m) for m in messages]
        # The final response should show higher risk than the first
        assert responses[-1].risk_score >= responses[0].risk_score


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
