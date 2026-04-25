#!/usr/bin/env python
import sys
import warnings
import json
import re
import os
import time
from pydantic import BaseModel
from crewai.flow.flow import Flow, listen, start
from first_crew.crew import FirstCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# === Step 1: Define Flow State ===
class YelpRecommendationState(BaseModel):
    user_id: str = ""
    item_id: str = ""
    user_profile: str = ""
    item_profile: str = ""
    web_research: str = ""
    raw_result: str = ""
    final_report: dict = {}

# === Step 2: Define the Parallel Flow ===
class YelpRecommendationFlow(Flow[YelpRecommendationState]):
    def _sleep_seconds(self, env_key: str, default_value: int) -> None:
        try:
            wait_seconds = int(os.getenv(env_key, str(default_value)))
        except ValueError:
            wait_seconds = default_value
        if wait_seconds > 0:
            time.sleep(wait_seconds)

    def _build_single_task_crew(self, task_name: str):
        crew_instance = FirstCrew().crew()
        selected_task = next(t for t in crew_instance.tasks if t.name == task_name)
        crew_instance.tasks = [selected_task]
        return crew_instance

    def _run_with_retries(self, task_name: str, inputs: dict, retries: int = 3):
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                crew_instance = self._build_single_task_crew(task_name)
                result = crew_instance.kickoff(inputs=inputs)
                raw_text = str(result.raw)
                degraded_markers = (
                    "Error executing task with agent",
                    "mentioned not found",
                    "Too Many Requests",
                    "Timeout Error",
                )
                if any(marker in raw_text for marker in degraded_markers):
                    raise RuntimeError(f"Degraded output detected for {task_name}: {raw_text[:240]}")
                return result
            except Exception as exc:
                last_error = exc
                if attempt < retries:
                    backoff = 30 * attempt  # 30s, 60s — enough for rate limit to reset
                    print(f"⚠️ [Retry {attempt}/{retries}] {task_name} failed: {exc}. Waiting {backoff}s...")
                    time.sleep(backoff)
                else:
                    print(f"❌ [Retry Exhausted] {task_name} failed after {retries} attempts.")
        raise RuntimeError(f"Task {task_name} failed after retries: {last_error}")
    
    @start()
    def initialize_request(self):
        print(f"🚀 [Flow Start]: Initializing Parallel Flow for User: {self.state.user_id} | Item: {self.state.item_id}")
        return "initialized"

    @listen(initialize_request)
    def fetch_user_profile(self):
        print(f"👤 [Flow Action]: Analyzing User Profile...")
        result = self._run_with_retries("analyze_user_task", {
            'user_id': self.state.user_id,
            'item_id': self.state.item_id,
            'user_context': '',
            'item_context': '',
            'web_context': ''
        })
        self.state.user_profile = result.raw
        return "user_ready"

    @listen(fetch_user_profile)
    def fetch_item_profile(self):
        # Sequential after user to avoid simultaneous LLM calls (NVIDIA 429 rate limit)
        print(f"🏠 [Flow Action]: User profile done. Waiting 45s before Item analysis...")
        self._sleep_seconds("FLOW_STAGGER_SECONDS", 45)
        print(f"🏠 [Flow Action]: Analyzing Item Profile...")
        result = self._run_with_retries("analyze_item_task", {
            'user_id': self.state.user_id,
            'item_id': self.state.item_id,
            'user_context': '',
            'item_context': '',
            'web_context': ''
        })
        self.state.item_profile = result.raw
        return "item_ready"

    @listen(fetch_item_profile)
    def fetch_web_research(self):
        print(f"🔍 [Flow Action]: Breathing for 30 seconds before Web Research...")
        self._sleep_seconds("FLOW_WEB_WAIT_SECONDS", 30)
        print(f"🌐 [Flow Action]: Item profile ready. Now searching the web for real-time trends...")
        # We now have the item_profile, so the agent can see the business name!
        result = self._run_with_retries("web_research_task", {
            'user_id': self.state.user_id,
            'item_id': self.state.item_id,
            'user_context': self.state.user_profile,
            'item_context': self.state.item_profile,
            'web_context': ''
        })
        self.state.web_research = result.raw
        return "web_ready"

    @listen(fetch_web_research)
    def run_final_prediction(self):
        print(f"⚖️ [Flow Action]: Data converged. Breathing for 60 seconds before Final Prediction...")
        self._sleep_seconds("FLOW_PREDICTION_WAIT_SECONDS", 60)
        process_type = os.getenv("PROCESS_TYPE", "hierarchical").upper()
        print(f"⚖️ [Flow Action]: Kicking off {process_type} Crew for final prediction...")

        # Truncate contexts to avoid blowing the TPM budget on the prediction prompt
        MAX_CTX = int(os.getenv("PREDICTION_MAX_CTX_CHARS", "1200"))
        inputs = {
            'user_id': self.state.user_id,
            'item_id': self.state.item_id,
            'user_context': self.state.user_profile[:MAX_CTX],
            'item_context': self.state.item_profile[:MAX_CTX],
            'web_context': self.state.web_research[:600]
        }

        try:
            result = self._run_with_retries("predict_review_task", inputs)
        except Exception as primary_error:
            original_mode = os.getenv("PROCESS_TYPE", "hierarchical")
            if original_mode.lower() == "hierarchical":
                print(f"⚠️ [Fallback] Hierarchical prediction failed ({primary_error}). Retrying in collaborative mode...")
                os.environ["PROCESS_TYPE"] = "collaborative"
                result = self._run_with_retries("predict_review_task", inputs)
                os.environ["PROCESS_TYPE"] = original_mode
            else:
                raise

        self.state.raw_result = result.raw
        return "prediction_completed"

    @listen(run_final_prediction)
    def process_and_save_results(self):
        process_type = os.getenv("PROCESS_TYPE", "hierarchical").lower().replace("_report", "")
        filename = f"{process_type}_report.json"
        
        print(f"📊 [Flow Finalizing]: Parsing results and saving to {filename}...")
        report = self.extract_json_from_output(self.state.raw_result)
        self.state.final_report = report
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        review_snippet = str(report.get('review', report.get('text', '')))[:60]
        print(f"[Flow Success]: Stars: {report.get('stars')} | Review: {review_snippet}... | Saved to {filename}")
        return report


    def extract_json_from_output(self, raw_output: str) -> dict:
        """Extract and sanitize JSON from LLM raw output."""
        text = str(raw_output).strip()
        
        # Remove markdown code blocks if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
            
        # Try to find anything between { and }
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Final fallback: Manual extraction of key fields
            stars_match = re.search(r'"stars":\s*([\d.]+)', text)
            review_match = re.search(r'"review":\s*"(.*)"', text, re.DOTALL)
            return {
                "stars": float(stars_match.group(1)) if stars_match else None,
                "review": review_match.group(1) if review_match else text,
                "_parse_manual": True
            }

def run():
    valid_modes = ["hierarchical", "collaborative", "sequential"]
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in valid_modes:
            os.environ["PROCESS_TYPE"] = arg
            print(f"🎯 [Mode Override]: Switching to {arg.upper()} mode.")

    test_json_path = "data/test_review_subset.json"
    if not os.path.exists(test_json_path):
        print(f"Error: {test_json_path} not found.")
        return

    with open(test_json_path, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f if line.strip()]

    # Index 0: first record as required by Lab 9 ("第一筆測試資料")
    # User: Karen (Elite 2008-2021, avg_stars=3.69) | Item: Pho Street, Philadelphia PA
    first_case = test_data[0]


    flow = YelpRecommendationFlow()
    flow.state.user_id = first_case['user_id']
    flow.state.item_id = first_case['item_id']
    
    flow.kickoff()

def train(): pass
def replay(): pass
def test(): pass

if __name__ == "__main__":
    run()
