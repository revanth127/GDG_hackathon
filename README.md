## 🛠️ Tech Stack

This project is built using a lightweight and modular backend-focused stack, optimized for rapid prototyping and AI-agent experimentation.

### Core Technologies
- **Python 3.10+** — Primary language for data processing and agent logic
- **Google ADK (Agent Development Kit)** — For building and structuring AI agents
- **LLM (via Google AI Studio / Ollama)** — Natural language reasoning and analysis
- **CSV / JSON** — Lightweight data storage for race telemetry and processed outputs

### Supporting Tools
- **Git & GitHub** — Version control and collaboration
- **Virtual Environment (venv)** — Dependency isolation
- **Streamlit (optional UI layer)** — For quick local interaction and demos

---

## ▶️ How to Run the Project

Follow these steps to run the project locally.

### 1️⃣ Clone the repository
git clone https://github.com/revanth127/GDG_Hackathon.git
cd GDG_Hackathon
2️⃣ Create and activate a virtual environment
Windows
python -m venv venv
venv\Scripts\activate
Mac / Linux
python3 -m venv venv
source venv/bin/activate
3️⃣ Install dependencies
pip install -r requirements.txt
4️⃣ Prepare data (if required)
If working with raw race data:
python csv_to_json.py
This converts CSV telemetry data into structured JSON for agent consumption.

5️⃣ Run the agent logic
python agents.py
This will:

Load race context data

Analyze driver pace and tyre life

Generate natural-language insights and recommendations

(Optional) Run with Streamlit UI
If a UI layer is included:
streamlit run agents.py