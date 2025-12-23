import requests
import json
from datetime import datetime
from setup import (
    PERPLEXITY_API_KEY,
    PERPLEXITY_MODEL,
    SEARCH_QUERIES,
    START_DATE,
    END_DATE
)

API_URL = "https://api.perplexity.ai/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
    "Content-Type": "application/json"
}

def build_prompt(query: str) -> str:
    return f"""
    You are assisting in a structured academic literature search intended to update
    the state of the art of a previously conducted systematic literature review.

    Search for peer-reviewed or widely recognized preprint academic papers published
    between {START_DATE} and {END_DATE} that satisfy the following PICO-based criteria:

    Population (P):
    - Multi UAV systems
    - Autonomous drones
    - UAV swarms
    - Unmanned aerial vehicle swarms

    Intervention (I):
    - Multi-agent reinforcement learning (MARL)
    - Deep reinforcement learning (DRL)
    - Reinforcement learning
    - Deep learning and neural networks applied to decision-making or control

    Comparison / Task Context (C):
    - Exploration
    - Navigation and path planning
    - Collision avoidance
    - Formation control
    - Coordination and cooperation

    Outcome (O):
    - Performance evaluation
    - Efficiency (computational or energetic)
    - Adaptability
    - Robustness
    - Scalability

    The specific thematic focus of this query is:
    {query}

    Constraints:
    - Focus exclusively on UAV swarm scenarios (single UAV or ground robots must be excluded)
    - Prioritize decentralized execution or cooperative multi-agent settings
    - Prefer studies validated in 3D simulation environments or real-world experiments
    - Exclude purely theoretical works without experimental validation

    For each relevant work, return the information in a structured manner:
    - Title
    - Authors
    - Year
    - Publication venue (journal, conference, or preprint repository)
    - Main contribution (2–3 sentences, technical and objective)
    - Task addressed (e.g., navigation, formation, target tracking)
    - Learning paradigm (e.g., MARL, DRL, centralized training with decentralized execution)
    - Evaluation setup (simulation, real-world, or both)
    - Link (DOI, publisher page, or arXiv)

    Important:
    - Do not fabricate references
    - Do not infer missing bibliographic information
    - Only include verifiable academic sources
    """


def run_search():
    all_results = {}

    for theme, query in SEARCH_QUERIES.items():
        print(f"[INFO] Searching theme: {theme}")

        payload = {
            "model": PERPLEXITY_MODEL,
            "messages": [
                {"role": "system", "content": "You are an academic research assistant."},
                {"role": "user", "content": build_prompt(query)}
            ],
            "temperature": 0.1
        }

        response = requests.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"]
        all_results[theme] = {
            "query": query,
            "response": content,
            "timestamp": datetime.utcnow().isoformat()
        }

    with open("perplexity_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("[INFO] Search completed. Results saved to perplexity_results.json")

if __name__ == "__main__":
    run_search()
