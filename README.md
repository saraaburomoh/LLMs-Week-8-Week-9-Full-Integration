# 🚀 Yelp Recommendation Crew Profiler

A sophisticated multi-agent pipeline built with **CrewAI** and **Astral uv** that analyzes Yelp user behavior, business attributes, and real-time web trends to predict highly accurate star ratings and review texts.

## 🏗️ System Architecture
This project implements a three-stage analysis pipeline using a specialized crew of AI agents:
1.  **User Profiling**: Extracts behavior patterns, "Elite" status history, and rating tendencies.
2.  **Item Analysis**: Decodes business attributes (WiFi, Parking, etc.) and analyzes historical customer feedback.
3.  **Web Research**: Supplements local data with real-time business status and external trends.
4.  **Prediction**: Synthesizes all context into a valid JSON recommendation report.

## 📂 Project Structure
```bash
Rag_Crew_Profiler/
├── src/
│   └── first_crew/
│       ├── config/
│       │   ├── agents.yaml    # Agent roles, goals, and backstories
│       │   └── tasks.yaml     # Task descriptions and expected outputs
│       ├── tools/             # Custom lookup and RAG tools
│       ├── crew.py            # CrewBase orchestration logic
│       └── main.py            # Flow control and entry point
├── docs/                      # Knowledge base (Schema & EDA findings)
├── data/                      # Local JSON subsets (User, Item, Review)
├── pyproject.toml             # uv package management
└── .env                       # Environment configuration
```

## 🛠️ Setup & Installation

### 1. Requirements
*   Python 3.10+
*   [Astral uv](https://github.com/astral-sh/uv) (Highly recommended for performance)

### 2. Install Dependencies
```bash
uv sync
```

### 3. Environment Variables
Create a `.env` file in the root directory:
```env
LLM_PROVIDER=nvidia
NVIDIA_API_KEY=your_key_here
NVIDIA_MODEL_NAME=meta/llama-3.3-70b-instruct
SERPER_API_KEY=your_key_here
PROCESS_TYPE=sequential  # Options: sequential, collaborative, hierarchical
```

## 🚀 Execution

The project supports three execution modes. You can override the mode via command line arguments:

### Sequential Mode (Default)
Executes tasks in a strict linear order (User -> Item -> Web -> Prediction).
```bash
uv run first_crew sequential
```

### Hierarchical Mode
Uses a **Project Manager** agent to oversee delegation and quality control.
```bash
uv run first_crew hierarchical
```

### Collaborative Mode
Enables agents to delegate tasks to each other dynamically.
```bash
uv run first_crew collaborative
```

## 🧠 Core Components

### `crew.py`
The heart of the project. It uses the `@CrewBase` decorator to:
*   Initialize **Knowledge** (RAG) using local `sentence-transformer` embeddings.
*   Configure **Lookup Tools** for deterministic data retrieval from local JSON subsets.
*   Define the crew of agents: `user_analyst`, `item_analyst`, `web_researcher`, and `prediction_modeler`.

### `main.py`
Implements a **CrewAI Flow** (`YelpRecommendationFlow`) to handle parallel data fetching and state management. It ensures that the final prediction only runs once all context (User + Item + Web) has converged.

### `config/agents.yaml`
Follows the **Strict Separation** principle. Defines agent roles such as the **Yelp User Profiler** and **Review Prediction Expert**, ensuring each agent has a clear goal and specific instructions to ignore knowledge retrieval failures regarding IDs.

### `config/tasks.yaml`
Defines the workflow steps. Includes specific `[KNOWLEDGE]`, `[DATA]`, and `[HISTORY]` steps to ensure agents follow a systematic analysis process.

## 📊 Output
The system generates a `report.json` (or `{mode}_report.json`) containing the final prediction:
```json
{
  "stars": 4.5,
  "review": "The atmosphere was great, and the service was top-notch..."
}
```

