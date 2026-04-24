# 🚀 Yelp Recommendation Crew Profiler

A high-performance multi-agent system built with **CrewAI** and **Astral uv** designed to predict user ratings and reviews by synthesizing historical behavior, business characteristics, and real-time trends.

## 🌊 Execution Flow (YelpRecommendationFlow)
The system utilizes a sophisticated **Parallel Flow** architecture to ensure data integrity and efficient execution. The flow orchestrates the agents through the following state-managed steps:

1.  **Initialization**: Captures the target `user_id` and `item_id`.
2.  **Parallel Profiling**:
    *   **User Analyst**: Executes `analyze_user_task` to extract behavior patterns and rating habits.
    *   **Item Analyst**: Executes `analyze_item_task` to identify business features (WiFi, Parking, etc.). *Includes a 10-second stagger to prevent API rate limits.*
3.  **Web Synthesis**: Once the item profile is ready, the **Web Researcher** uses the business name to find real-time status and external trends.
4.  **Convergence & Prediction**: When both the User Profile and Web Research are complete, the data converges. The **Prediction Modeler** then generates a JSON report containing the predicted stars and review text, matching the user's historical tone.
5.  **Finalization**: Results are automatically parsed and saved to a dedicated mode-specific JSON report (e.g., `sequential_report.json`).

## 🧠 Multi-Agent Core

### 1. Crew Orchestration (`crew.py`)
The system supports **Sequential**, **Collaborative**, and **Hierarchical** process modes. It integrates:
*   **Knowledge Base (RAG)**: Automatically retrieves schema definitions and EDA findings to interpret complex data fields correctly.
*   **Deterministic Lookup Tools**: Python-based tools (`lookup_user_by_id`, etc.) ensure exact data retrieval from local JSON subsets, preventing LLM hallucinations.

### 2. Specialized Agents (`agents.yaml`)
*   **Yelp User Profiler**: Analyzes review counts, elite status, and social influence to build a psychological profile of the reviewer.
*   **Yelp Restaurant Analyst**: Decodes complex business attributes and synthesizes historical customer feedback.
*   **External Trend Researcher**: Monitors the live web for the latest business status (open/closed) and social sentiment.
*   **Review Prediction Expert**: A master of behavioral prediction that generates structured JSON outputs.

### 3. Structured Tasks (`tasks.yaml`)
Each task is strictly separated into three logical phases:
*   **[KNOWLEDGE]**: Retrieval of dictionary definitions and schema context.
*   **[DATA]**: Exact lookup of target profile information.
*   **[HISTORY]**: Comprehensive analysis of historical interaction data.

## 🚀 How to Run

### Installation
```bash
uv sync
```

### Modes of Operation
You can run the flow in different architectural patterns by passing the mode as an argument:

*   **Sequential Mode**: `uv run first_crew sequential`
*   **Collaborative Mode**: `uv run first_crew collaborative`
*   **Hierarchical Mode**: `uv run first_crew hierarchical`

## 📊 Final Prediction Output
The system produces a strict JSON response designed for programmatic consumption:
```json
{
  "stars": 4.5,
  "review": "The atmosphere was exactly what I expected based on my previous visits to similar spots..."
}
```
