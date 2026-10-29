# 🧠 MindBridge — Agentic AI for Mental Health Awareness & Suicide Prevention

**Powered by IBM Granite 4 (granite-4-h-small) · IBM watsonx.ai · IBM Cloud**

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![IBM watsonx](https://img.shields.io/badge/IBM-watsonx.ai-black?logo=ibm)](https://dataplatform.cloud.ibm.com)
[![IBM Granite](https://img.shields.io/badge/IBM-Granite%204-purple?logo=ibm)](https://www.ibm.com/granite)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

> ⚕️ **Disclaimer**: MindBridge is an AI companion and is **NOT a substitute** for professional
> medical advice, diagnosis, or treatment. If you or someone you know is in immediate danger,
> please call emergency services (**911 / 112 / 100**) or a crisis helpline immediately.

---

## 📌 Overview

Mental health issues such as depression, anxiety, and suicidal thoughts are growing worldwide.
Stigma, lack of awareness, and limited access to timely help prevent many individuals from
seeking support. Traditional systems are often reactive, leaving gaps in early detection and
preventive care.

**MindBridge** is an **Agentic AI**-powered Mental Health Awareness & Suicide Prevention
companion that is empathetic, proactive, and safe. It:

- 🧘 **Promotes Awareness & Education** — shares mindfulness, coping strategies, and self-care resources
- 🔍 **Enables Early Detection (RAG)** — identifies distress signals using Retrieval-Augmented Generation
- 💬 **Provides Empathetic Support** — listens actively and responds with warmth and compassion
- 📊 **Predicts & Prevents Risks** — forecasts stress escalation and issues consent-gated alerts
- 🆘 **Connects to Human Support** — links users to crisis helplines and counselors

---

## 🏗️ Architecture

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────────┐
│              Agentic Orchestrator                   │
│  (Observe → Think → Act → Respond)                 │
├──────────────────────────────────────────────────── ┤
│  Step 1 │ Safety Pre-Filter (AFSP/SAMHSA guidelines)│
│  Step 2 │ RAG Distress Analyzer (TF-IDF + KB)       │
│  Step 3 │ Risk Scorer & Trend Predictor             │
│  Step 4 │ Tool Selection                            │
│         │   ├─ Crisis Connector  (high/critical)    │
│         │   └─ Resource Recommender (low/medium)    │
│  Step 5 │ IBM Granite 4 via watsonx.ai              │
│  Step 6 │ Safety Post-Filter + Disclaimer           │
└─────────────────────────────────────────────────────┘
    │
    ▼
Empathetic Response → User
```

---

## 🛠️ Tech Stack

| Component              | Technology                                      |
|------------------------|-------------------------------------------------|
| LLM / Conversational AI | IBM Granite 4 — `granite-4-h-small` (watsonx.ai) |
| RAG / Knowledge Retrieval | In-memory TF-IDF cosine similarity            |
| Agentic Orchestration  | Python agentic loop (custom)                    |
| Cloud Platform         | IBM Cloud + IBM watsonx.ai                      |
| UI                     | Streamlit                                       |
| Credentials / Config   | python-dotenv + IBM Cloud IAM                   |
| Testing                | pytest (35 tests)                               |

---

## 📁 Project Structure

```
mental_health_agent/
├── app.py                      # Streamlit web UI
├── agent.py                    # CLI entry point
├── requirements.txt            # Python dependencies
├── .env.example                # Credential template
├── .gitignore
├── README.md
├── core/
│   ├── __init__.py
│   ├── orchestrator.py         # Agentic loop (main brain)
│   ├── granite_client.py       # IBM Granite 4 via watsonx.ai
│   ├── rag_engine.py           # RAG distress detection
│   ├── risk_scorer.py          # Risk prediction & alerts
│   ├── resources.py            # Coping resources & psychoeducation
│   └── crisis_connector.py     # Crisis helpline connector
├── utils/
│   ├── __init__.py
│   └── safety_filter.py        # Safe Messaging Guidelines filter
├── data/
│   ├── knowledge_base.json     # 12-entry mental health knowledge base
│   └── helplines.json          # Crisis lines: India, USA, UK, Australia
└── test_mindbridge.py          # 35 unit + integration tests
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- IBM Cloud account (free tier works) — [Register here](https://cloud.ibm.com/registration)
- IBM watsonx.ai project — [Create here](https://dataplatform.cloud.ibm.com/wx/home)

### 1. Clone the repository

```bash
git clone https://github.com/rashmibadaveshruti/mental-health.git
cd mental-health
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure credentials

```bash
cp .env.example .env
```

Edit `.env` and fill in your IBM Cloud credentials:

```env
WATSONX_API_KEY=your_ibm_cloud_api_key
WATSONX_PROJECT_ID=your_watsonx_project_id
WATSONX_REGION=us-south
GRANITE_MODEL_ID=ibm/granite-4-h-small-curated
```

> 💡 **Demo mode**: If you leave the placeholders, the app runs in demo mode with
> scripted safe responses — no IBM Cloud account needed to try it out.

### 4. Run the Streamlit UI

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### 5. (Optional) Run in CLI mode

```bash
python agent.py
```

---

## 🔑 Getting IBM Cloud Credentials

| Credential | Where to get it |
|---|---|
| `WATSONX_API_KEY` | [cloud.ibm.com](https://cloud.ibm.com) → Manage → Access (IAM) → API keys → Create |
| `WATSONX_PROJECT_ID` | [dataplatform.cloud.ibm.com](https://dataplatform.cloud.ibm.com) → Your project → Manage → General |
| `WATSONX_REGION` | `us-south` (default, recommended) |
| `GRANITE_MODEL_ID` | `ibm/granite-4-h-small-curated` (already set) |

---

## 🧪 Running Tests

```bash
python -m pytest test_mindbridge.py -v
```

Expected output: **35 passed**

```
test_mindbridge.py::TestRAGEngine::test_loads_knowledge_base PASSED
test_mindbridge.py::TestRAGEngine::test_critical_distress_detected PASSED
test_mindbridge.py::TestRiskScorer::test_escalating_trend_detected PASSED
test_mindbridge.py::TestSafetyFilter::test_detects_method_request PASSED
test_mindbridge.py::TestOrchestrator::test_harmful_request_blocked PASSED
... 35 passed in 1.20s
```

---

## 🛡️ Safety & Ethics

MindBridge is built on **AFSP / SAMHSA Safe Messaging Guidelines**:

- ✅ **Pre-filter** blocks harmful-method queries before they reach the LLM
- ✅ **Post-filter** scans model output and redacts any unsafe content
- ✅ **Never** provides methods, means, or step-by-step information related to self-harm
- ✅ **Always** provides crisis helpline numbers when risk signals are detected
- ✅ **Mandatory disclaimer** on every medium/high/critical risk response
- ✅ **Consent-gated** risk monitoring — users opt in to alerts (except critical, always active)
- ✅ **No data persistence** — conversations are in-memory only (session-scoped)

---

## 🆘 Crisis Helplines

| Country | Name | Number |
|---|---|---|
| 🇮🇳 India | iCall (TISS) | 9152987821 |
| 🇮🇳 India | Vandrevala Foundation | 1860-2662-345 |
| 🇺🇸 USA | 988 Suicide & Crisis Lifeline | 988 |
| 🇺🇸 USA | Crisis Text Line | Text HOME to 741741 |
| 🇬🇧 UK | Samaritans | 116 123 |
| 🇦🇺 Australia | Lifeline | 13 11 14 |
| 🌍 International | IASP Crisis Centres | [iasp.info](https://www.iasp.info/resources/Crisis_Centres/) |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- **IBM Granite** — Foundation model powering empathetic conversations
- **IBM watsonx.ai** — Enterprise AI platform
- **AFSP / SAMHSA** — Safe Messaging Guidelines for suicide prevention
- **iCall, Vandrevala Foundation, Samaritans** — Crisis support organizations

---

<div align="center">
  <sub>Built with ❤️ for mental health awareness · Powered by IBM Granite 4 · IBM watsonx.ai · IBM Cloud</sub>
</div>
