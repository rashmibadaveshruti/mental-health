"""
agent.py
MindBridge — CLI entry point for the Agentic AI Mental Health Companion.
Run: python agent.py
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

from core.orchestrator import MindBridgeOrchestrator

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║        🧠  M I N D B R I D G E                              ║
║        Agentic AI — Mental Health Awareness & Prevention     ║
║        Powered by IBM Granite · watsonx.ai · IBM Cloud       ║
╠══════════════════════════════════════════════════════════════╣
║  ⚕️  DISCLAIMER: This is an AI companion, NOT a substitute  ║
║  for professional medical advice.  In crisis?  Call:         ║
║  🇮🇳 iCall: 9152987821  |  🇺🇸 988  |  🇬🇧 116 123         ║
╚══════════════════════════════════════════════════════════════╝
Type 'quit' or 'exit' to end the session.
Type 'summary' to view your session risk summary.
Type 'consent on/off' to toggle risk monitoring.
"""

RISK_COLOURS = {
    "none":     "\033[92m",   # green
    "low":      "\033[94m",   # blue
    "medium":   "\033[93m",   # yellow
    "high":     "\033[91m",   # red
    "critical": "\033[41m",   # red background
}
RESET = "\033[0m"


def main():
    print(BANNER)

    # Country selection
    print("Available regions: india, usa, uk, australia")
    country = input("Your location [india]: ").strip().lower() or "india"
    consent_input = input("Enable risk monitoring? (y/n) [y]: ").strip().lower()
    consent = consent_input != "n"

    orchestrator = MindBridgeOrchestrator(
        country=country,
        consent_given=consent,
        api_key=os.getenv("WATSONX_API_KEY"),
        project_id=os.getenv("WATSONX_PROJECT_ID"),
        region=os.getenv("WATSONX_REGION", "us-south"),
    )

    print("\n" + "─" * 64)
    print("MindBridge: Hello 💙 I'm here to listen and support you.")
    print("MindBridge: How are you feeling today?")
    print("─" * 64 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nMindBridge: Take care of yourself. Goodbye 💙")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("\nMindBridge: Take care of yourself. Goodbye 💙")
            break

        if user_input.lower() == "summary":
            summary = orchestrator.get_session_summary()
            risk_s  = summary.get("risk_summary", {})
            print(f"\n📊 Session Summary:")
            print(f"   Turns       : {risk_s.get('turns', 0)}")
            print(f"   Peak Risk   : {risk_s.get('peak_risk', 'none').upper()}")
            print(f"   Avg Score   : {risk_s.get('avg_score', 0.0):.2f}")
            print(f"   Alerts      : {risk_s.get('alerts', 0)}")
            print(f"   Trend       : {risk_s.get('trend', 'stable').title()}\n")
            continue

        if user_input.lower().startswith("consent "):
            val = user_input.split()[-1].lower()
            orchestrator.set_consent(val == "on")
            print(f"MindBridge: Risk monitoring {'enabled' if val == 'on' else 'disabled'}.\n")
            continue

        # Run agent
        response = orchestrator.run(user_input)

        # Risk indicator
        colour    = RISK_COLOURS.get(response.risk_level, "")
        risk_str  = f"{colour}[{response.risk_level.upper()}]{RESET}"
        trend_str = f"↗" if response.trend == "escalating" else (
                    "↘" if response.trend == "de-escalating" else "→")

        print(f"\nMindBridge {risk_str}{trend_str}: {response.message}\n")

        # Alert output
        if response.alert:
            alert = response.alert
            print(f"  ⚡ ALERT ({alert.level.upper()}): {alert.message}")
            for action in alert.actions[:2]:
                print(f"     • {action}")
            print()

        # Tool trace (developer mode)
        if os.getenv("DEBUG_AGENT", "false").lower() == "true":
            print(f"  [Tools: {' → '.join(response.tools_used)}]")
            if response.distress_signals:
                print(f"  [Signals: {response.distress_signals[:3]}]")
            print()


if __name__ == "__main__":
    main()
