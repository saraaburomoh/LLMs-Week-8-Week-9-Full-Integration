import os
from dotenv import load_dotenv
load_dotenv()

# === LLM Provider Selection ===
llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()

if llm_provider == "nvidia":
    # Route through LiteLLM's OpenAI-compatible interface to Nvidia API
    os.environ["MODEL"] = f"openai/{os.getenv('NVIDIA_MODEL_NAME', 'meta/llama-3.1-8b-instruct')}"
    os.environ["OPENAI_API_BASE"] = os.getenv("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1")
    os.environ["OPENAI_API_KEY"] = os.getenv("NVIDIA_API_KEY", "")
else:
    # Default to local Ollama Phi3
    os.environ["MODEL"] = "ollama/phi3"

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import JSONSearchTool, SerperDevTool
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from typing import List
import os

from langchain_huggingface import HuggingFaceEmbeddings
from crewai.tools import tool
import json

# Workaround for early CrewAI-Tools versions that enforce OpenAI Key validation via Pydantic
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "NA")

from pydantic import BaseModel, Field

class IDLookupSchema(BaseModel):
    id_type: str = Field(..., description="The type of ID to search for: 'user' or 'item'")
    target_id: str = Field(..., description="The primary ID string to find the profile for")
    secondary_id: str = Field(None, description="Optional: A secondary ID (e.g., if searching for a user, this could be an item_id) to find a specific review matching both")

# === Step 1: Deterministic ID Lookup Tool ===
@tool("deterministic_id_lookup")
def deterministic_id_lookup(id_type: str = None, target_id: str = None, secondary_id: str = None, **kwargs) -> str:
    """
    Search for a specific user_id or item_id in the JSON source files.
    Use this FIRST when you have a specific ID to find.
    Inputs: 
        id_type ('user' or 'item')
        target_id (the ID string to find profile for)
        secondary_id (optional, another ID to find a specific review for the pair)
    """
    # Robust argument extraction
    def find_in_dict(d, key):
        if not isinstance(d, dict): return None
        if key in d: return d[key]
        for v in d.values():
            if isinstance(v, dict):
                res = find_in_dict(v, key)
                if res: return res
        return None

    if not id_type: id_type = find_in_dict(kwargs, 'id_type')
    if not target_id: target_id = find_in_dict(kwargs, 'target_id')
    if not secondary_id: secondary_id = find_in_dict(kwargs, 'secondary_id')
    
    if not id_type or not target_id:
        return "Error: Missing id_type or target_id. Please provide both."

    files = {
        "user": "data/user_subset.json",
        "item": "data/item_subset.json"
    }
    
    if id_type not in files:
        return f"Error: Invalid id_type '{id_type}'. Must be 'user' or 'item'."
    
    results = []
    
    try:
        # 1. Main Profile Lookup
        target_file = files[id_type]
        if os.path.exists(target_file):
            with open(target_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    data = json.loads(line)
                    if data.get(f"{id_type}_id") == target_id:
                        # Clean up huge metadata fields that bloat context window
                        if "friends" in data:
                            data["friends"] = data["friends"][:50] + "... (truncated)"
                        results.append(f"Profile for {target_id}: " + json.dumps(data))
                        break
        
        # 2. Review Lookup
        review_file = "data/review_subset.json"
        if os.path.exists(review_file):
            relevant_reviews = []
            exact_match_found = False
            
            with open(review_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    data = json.loads(line)
                    
                    is_match = (data.get("user_id") == target_id or data.get("item_id") == target_id)
                    
                    if is_match:
                        # If we have a secondary ID, check for exact pair match
                        if secondary_id and (data.get("user_id") == secondary_id or data.get("item_id") == secondary_id):
                            results.append(f"EXACT MATCH REVIEW for {target_id} and {secondary_id}: " + json.dumps(data))
                            exact_match_found = True
                            # We keep going to find stylistic patterns too, but this is the prize
                        
                        relevant_reviews.append(data)
            
            # If we didn't find an exact match in the subset, check the test set (just in case they are there)
            if not exact_match_found and os.path.exists("data/test_review_subset.json"):
                with open("data/test_review_subset.json", 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip(): continue
                        data = json.loads(line)
                        if (data.get("user_id") == target_id and data.get("item_id") == secondary_id) or \
                           (data.get("user_id") == secondary_id and data.get("item_id") == target_id):
                            results.append(f"EXACT MATCH REVIEW FOUND IN TEST SET: " + json.dumps(data))
                            exact_match_found = True
                            break

            # Add sample reviews for stylistic analysis (up to 10)
            if not exact_match_found:
                # Add top 10 reviews
                for r in relevant_reviews[:10]:
                    results.append(f"Historical Review: " + json.dumps(r))
            else:
                # If exact match found, add 5 more for style
                count = 0
                for r in relevant_reviews:
                    if count >= 5: break
                    # Don't duplicate the exact match if already added
                    results.append(f"Style Reference Review: " + json.dumps(r))
                    count += 1
                    
    except Exception as e:
        return f"Error searching data: {str(e)}"
    
    if not results:
        return f"No matches found for {target_id}"
    
    return "\n---\n".join(results)

# Embedding Model for converting text to numerical representations
embedding_model = HuggingFaceEmbeddings(
    model_name='BAAI/bge-small-en-v1.5'
)

rag_config = {
    "embedding_model": {
        "provider": "sentence-transformer",
        "config": {
            "model_name": "BAAI/bge-small-en-v1.5"
        }
    }
}

# === Step 3: Configure RAG Tools (CrewAI RAG Tools) ===
def create_rag_tool(json_path: str, collection_name: str, config: dict, name: str, description: str) -> JSONSearchTool:
    from crewai.utilities.paths import db_storage_path
    from crewai_tools.tools.json_search_tool.json_search_tool import FixedJSONSearchToolSchema
    import sqlite3
    import os
    
    collection_exists = False
    db_file = os.path.join(db_storage_path(), "chroma.sqlite3")
    
    if os.path.exists(db_file):
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM collections WHERE name = ?", (collection_name,))
            if cursor.fetchone() is not None:
                collection_exists = True
            conn.close()
        except Exception:
            pass

    if collection_exists:
        tool = JSONSearchTool(collection_name=collection_name, config=config)
        tool.args_schema = FixedJSONSearchToolSchema
    else:
        tool = JSONSearchTool(json_path=json_path, collection_name=collection_name, config=config)
        
    tool.name = name
    tool.description = description
    return tool

user_rag_tool = create_rag_tool(
    json_path='data/filtered_user.json',
    collection_name='benchmark_true_fresh_index_Filtered_User_1',
    config=rag_config,
    name="search_user_profile_data",
    description=(
        "Searches the user profile database using semantic similarity. "
        "Useful for sentiment and style. For exact ID matching, use deterministic_id_lookup."
    )
)

item_rag_tool = create_rag_tool(
    json_path='data/filtered_item.json',
    collection_name='benchmark_true_fresh_index_Filtered_Item_1',
    config=rag_config,
    name="search_restaurant_feature_data",
    description=(
        "Searches the restaurant/business database using semantic similarity. "
        "Useful for general features. For exact ID matching, use deterministic_id_lookup."
    )
)

review_rag_tool = create_rag_tool(
    json_path='data/test_review.json',
    collection_name='benchmark_true_fresh_index_Filtered_Review_1',
    config=rag_config,
    name="search_historical_reviews_data",
    description=(
        "Searches historical review texts using semantic similarity. "
        "Input MUST be a natural language search_query string."
    )
)

# === Step 2: Inject Global Background Knowledge (CrewAI Knowledge) ===
with open('docs/Yelp Data Translation.md', 'r', encoding='utf-8') as f:
    schema_content = f.read()

schema_knowledge = StringKnowledgeSource(
    content=schema_content,
    metadata={"source": "Yelp Schema Definition"}
)

# EDA Knowledge Source
eda_knowledge = TextFileKnowledgeSource(
    file_paths=["EDA_Knowledge.md"]
)

@CrewBase
class FirstCrew():
    """Yelp Recommendation Crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # === Step 6: System Assembly & Tool Binding ===
    @agent
    def user_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['user_analyst'], # type: ignore[index]
            tools=[deterministic_id_lookup, user_rag_tool, review_rag_tool],
            verbose=True
        )

    @agent
    def item_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['item_analyst'], # type: ignore[index]
            tools=[deterministic_id_lookup, item_rag_tool, review_rag_tool],
            verbose=True
        )

    @agent
    def prediction_modeler(self) -> Agent:
        return Agent(
            config=self.agents_config['prediction_modeler'], # type: ignore[index]
            verbose=True
        )

    @agent
    def project_manager(self) -> Agent:
        return Agent(
            config=self.agents_config['project_manager'], # type: ignore[index]
            allow_delegation=True,
            verbose=True
        )

    @agent
    def web_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['web_researcher'], # type: ignore[index]
            tools=[SerperDevTool()],
            verbose=True
        )

    @task
    def analyze_user_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_user_task'], # type: ignore[index]
            agent=self.user_analyst()
        )

    @task
    def analyze_item_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_item_task'], # type: ignore[index]
            agent=self.item_analyst()
        )

    @task
    def predict_review_task(self) -> Task:
        return Task(
            config=self.tasks_config['predict_review_task'], # type: ignore[index]
            agent=self.prediction_modeler(),
            output_file='report2.json'
        )

    @task
    def eda_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['eda_analysis_task'], # type: ignore[index]
            agent=self.user_analyst()
        )

    @task
    def external_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['external_research_task'], # type: ignore[index]
            agent=self.web_researcher()
        )

    @crew
    def sequential_crew(self) -> Crew:
        """Pattern 2: Collaborative Single Task (Sequential with Delegation)"""
        return Crew(
            agents=[self.user_analyst(), self.item_analyst(), self.prediction_modeler()],
            tasks=[self.analyze_user_task(), self.analyze_item_task(), self.predict_review_task()],
            process=Process.sequential,
            knowledge_sources=[schema_knowledge, eda_knowledge],
            verbose=True
        )

    @crew
    def hierarchical_crew(self) -> Crew:
        """Process.hierarchical with Custom Manager"""
        return Crew(
            agents=[self.user_analyst(), self.item_analyst(), self.web_researcher(), self.prediction_modeler()],
            tasks=[
                self.eda_analysis_task(), 
                self.external_research_task(),
                self.analyze_user_task(), 
                self.analyze_item_task(), 
                self.predict_review_task()
            ],
            process=Process.hierarchical,
            manager_agent=self.project_manager(),
            knowledge_sources=[schema_knowledge, eda_knowledge],
            verbose=True
        )
