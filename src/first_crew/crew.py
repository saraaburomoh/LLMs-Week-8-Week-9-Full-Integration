# Load environment variables from .env
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
from crewai_tools import JSONSearchTool, SerperDevTool
from crewai.knowledge.knowledge import Knowledge
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.tools import tool
from typing import List
import os
import json

from langchain_huggingface import HuggingFaceEmbeddings

# Workaround for early CrewAI-Tools versions that enforce OpenAI Key validation via Pydantic
os.environ["HF_HUB_OFFLINE"] = "1"
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "NA"

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
def ensure_chroma_index():
    from crewai.utilities.paths import db_storage_path
    import tarfile
    import os
    
    db_dir = db_storage_path()
    db_file = os.path.join(db_dir, "chroma.sqlite3")
    archive_path = r"C:\Users\MCC\Downloads\chroma_index.tar.gz"
    
    if not os.path.exists(db_file) and os.path.exists(archive_path):
        print(f"Index database not found. Extracting from {archive_path}...")
        if not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=db_dir)
            print("Extraction complete.")
        except Exception as e:
            print(f"Error extracting index: {e}")

# Call the ensure function before creating tools
ensure_chroma_index()

def create_rag_tool(json_path: str, collection_name: str, config: dict, name: str, description: str) -> JSONSearchTool:
    from crewai.utilities.paths import db_storage_path
    from crewai_tools.tools.json_search_tool.json_search_tool import FixedJSONSearchToolSchema
    import sqlite3
    import os
    
    collection_exists = False
    db_file = os.path.join(db_storage_path(), "chroma.sqlite3")
    
    if os.path.exists(db_file):
        try:
            # Check native sqlite3 for existing collection to heavily avoid 100% JSON text synchronous chunking bottleneck
            # and avoid ChromaDB singleton initialization conflicts with CrewAI's internal Settings
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
        # CRITICAL: Force the Pydantic schema to hide json_path from the Agent, 
        # so it doesn't trigger validation errors or pass the path and trigger the 3-hour hash loop!
        tool.args_schema = FixedJSONSearchToolSchema
    else:
        tool = JSONSearchTool(json_path=json_path, collection_name=collection_name, config=config)
        
    tool.name = name
    tool.description = description
    return tool

user_rag_tool = create_rag_tool(
    json_path='data/user_subset.json',
    collection_name='benchmark_true_fresh_index_Filtered_User_1',
    config=rag_config,
    name="search_user_profile_data",
    description=(
        "Searches the user profile database using semantic similarity. "
        "Input MUST be a natural language search_query string, e.g. "
        "'What are the review habits and average stars for user _BcWyKQL16?'. "
        "Do NOT pass raw user_id or JSON objects directly."
    )
)

item_rag_tool = create_rag_tool(
    json_path='data/item_subset.json',
    collection_name='benchmark_true_fresh_index_Filtered_Item_1',
    config=rag_config,
    name="search_restaurant_feature_data",
    description=(
        "Searches the restaurant/business database using semantic similarity. "
        "Input MUST be a natural language search_query string, e.g. "
        "'What are the categories, location, and star rating for business abc123?'. "
        "Do NOT pass raw item_id or JSON objects directly."
    )
)

review_rag_tool = create_rag_tool(
    json_path='data/review_subset.json',
    collection_name='benchmark_true_fresh_index_Filtered_Review_1',
    config=rag_config,
    name="search_historical_reviews_data",
    description=(
        "Searches historical review texts using semantic similarity. "
        "Input MUST be a natural language search_query string, e.g. "
        "'Find past reviews written by user _BcWyKQL16 about food quality and service'. "
        "Do NOT pass raw user_id, item_id, or JSON objects directly."
    )
)

# === Step 2: Inject Global Background Knowledge (CrewAI Knowledge) ===
with open('docs/Yelp Data Translation.md', 'r', encoding='utf-8') as f:
    schema_content = f.read()

schema_knowledge = StringKnowledgeSource(
    content=schema_content,
    metadata={"source": "Yelp Schema Definition"}
)

# New Knowledge: Exploratory Data Analysis (EDA)
with open('docs/eda_findings.md', 'r', encoding='utf-8') as f:
    eda_content = f.read()
eda_knowledge = StringKnowledgeSource(
    content=eda_content,
    metadata={"source": "Exploratory Data Analysis findings"}
)

# === Step 4: Configure Exact Lookup Tools (Python Tools) ===
@tool("lookup_user_by_id")
def lookup_user_by_id(user_id: str) -> str:
    """Look up a user's exact profile. 
    Input MUST be just the raw user_id string. 
    Example: {"user_id": "nnImk681KaRqUVHlSfZjGQ"}"""
    json_path = 'data/user_subset.json'
    if not os.path.exists(json_path):
        return f"File {json_path} not found."
        
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            if record.get('user_id') == user_id:
                # Remove long friends list to save tokens
                record.pop('friends', None)
                return json.dumps(record)
    return f"User {user_id} not found in exact lookup. Use search tools as fallback."

@tool("lookup_item_by_id")
def lookup_item_by_id(item_id: str) -> str:
    """Look up a business's profile. 
    Input MUST be just the raw item_id string. 
    Example: {"item_id": "-7GjicSH_rM8JeZGCXGcUg"}"""
    json_path = 'data/item_subset.json'
    if not os.path.exists(json_path):
        return f"File {json_path} not found."

    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            if record.get('business_id') == item_id or record.get('item_id') == item_id:
                return json.dumps(record)
    return f"Item {item_id} not found in exact lookup. Use search tools as fallback."

@tool("lookup_reviews_by_user_id")
def lookup_reviews_by_user_id(user_id: str) -> str:
    """Look up historical reviews by user_id. 
    Input MUST be just the raw user_id string. 
    Example: {"user_id": "nnImk681KaRqUVHlSfZjGQ"}"""
    json_path = 'data/review_subset.json'
    if not os.path.exists(json_path):
        return f"File {json_path} not found."
    
    matches = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            if record.get('user_id') == user_id:
                matches.append(record)
    
    if matches:
        return json.dumps(matches[:10]) # Return up to 10 latest reviews
    return f"No reviews found for user {user_id}. Use search tools as fallback."

@tool("lookup_reviews_by_business_id")
def lookup_reviews_by_business_id(business_id: str) -> str:
    """Look up all reviews for a specific business_id (item_id). Returns a list of review objects."""
    json_path = 'data/review_subset.json'
    if not os.path.exists(json_path):
        return f"File {json_path} not found."
    
    matches = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            if record.get('business_id') == business_id or record.get('item_id') == business_id:
                matches.append(record)
    
    if matches:
        return json.dumps(matches[:10]) # Return up to 10 latest reviews
    return f"No reviews found for business {business_id}. Use search tools as fallback."

@CrewBase
class FirstCrew():
    """Yelp Recommendation Crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # === Step 6: Dynamic Agent Definitions ===
    @agent
    def user_analyst(self) -> Agent:
        process_type = os.getenv("PROCESS_TYPE", "hierarchical").lower()
        return Agent(
            config=self.agents_config['user_analyst'], # type: ignore[index]
            tools=[lookup_user_by_id, lookup_reviews_by_user_id, user_rag_tool, review_rag_tool],
            verbose=True,
            allow_delegation=(process_type == "collaborative"), # Enabled ONLY for collaborative mode
            max_iter=6
        )

    @agent
    def item_analyst(self) -> Agent:
        process_type = os.getenv("PROCESS_TYPE", "hierarchical").lower()
        return Agent(
            config=self.agents_config['item_analyst'], # type: ignore[index]
            tools=[lookup_item_by_id, lookup_reviews_by_business_id, item_rag_tool, review_rag_tool],
            verbose=True,
            allow_delegation=(process_type == "collaborative"),
            max_iter=6
        )

    @agent
    def prediction_modeler(self) -> Agent:
        return Agent(
            config=self.agents_config['prediction_modeler'], # type: ignore[index]
            verbose=True,
            allow_delegation=False,
            max_iter=6
        )

    @agent
    def manager(self) -> Agent:
        return Agent(
            config=self.agents_config['manager'], # type: ignore[index]
            verbose=True,
            allow_delegation=True
        )

    @agent
    def web_researcher(self) -> Agent:
        tools = []
        if os.getenv("SERPER_API_KEY"):
            tools.append(SerperDevTool())
        
        return Agent(
            config=self.agents_config['web_researcher'], # type: ignore[index]
            tools=tools,
            verbose=True,
            allow_delegation=False
        )

    # === Step 7: Dynamic Task Binding ===
    @task
    def analyze_user_task(self) -> Task:
        return Task(config=self.tasks_config['analyze_user_task'])

    @task
    def analyze_item_task(self) -> Task:
        return Task(config=self.tasks_config['analyze_item_task'])

    @task
    def web_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['web_research_task'],
            context=[self.analyze_item_task()]
        )

    @task
    def predict_review_task(self) -> Task:
        return Task(
            config=self.tasks_config['predict_review_task'],
            context=[self.analyze_user_task(), self.analyze_item_task(), self.web_research_task()],
            output_file='report.json'
        )

    @crew
    def crew(self) -> Crew:
        # Get process type from environment
        process_type = os.getenv("PROCESS_TYPE", "hierarchical").lower()
        
        # Configure based on mode
        if process_type == "hierarchical":
            print("🏗️  Mode: HIERARCHICAL (Manager-led)")
            exec_process = Process.hierarchical
            exec_manager = self.manager()
            # Explicitly list workers to ensure they are all visible to the manager
            worker_agents = [
                self.user_analyst(),
                self.item_analyst(),
                self.prediction_modeler(),
                self.web_researcher()
            ]
        else:
            print(f"⛓️  Mode: {process_type.upper()} (Sequential)")
            exec_process = Process.sequential
            exec_manager = None
            worker_agents = [
                self.user_analyst(),
                self.item_analyst(),
                self.prediction_modeler(),
                self.web_researcher()
            ]

        return Crew(
            agents=worker_agents,
            tasks=self.tasks,
            process=exec_process,
            manager_agent=exec_manager,
            max_rpm=int(os.getenv("CREW_MAX_RPM", "12")),
            knowledge=Knowledge(
                collection_name="yelp_knowledge_v6",
                sources=[schema_knowledge, eda_knowledge],
                embedder={
                    "provider": "sentence-transformer",
                    "config": {"model_name": "BAAI/bge-small-en-v1.5"}
                }
            ),
            embedder={
                "provider": "sentence-transformer",
                "config": {"model_name": "BAAI/bge-small-en-v1.5"}
            },
            verbose=True
        )
