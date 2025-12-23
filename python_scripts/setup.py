import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# ===============================
# PERPLEXITY CONFIG
# ===============================
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

if PERPLEXITY_API_KEY is None:
    raise RuntimeError(
        "PERPLEXITY_API_KEY not found. "
        "Make sure you created a .env file with PERPLEXITY_API_KEY."
    )

PERPLEXITY_MODEL = "sonar-pro"

# ===============================
# TEMPORAL FILTER
# ===============================
START_DATE = "2024-06"
END_DATE   = "2025-12"

# ===============================
# PICO TERMS
# ===============================
P_POPULATION = [
    "Multi UAV",
    "autonomous drones",
    "UAV swarm",
    "unmanned aerial vehicle swarm",
    "autonomous UAV"
]

I_INTERVENTION = [
    "multi-agent reinforcement learning",
    "deep reinforcement learning",
    "reinforcement learning",
    "deep learning",
    "neural networks",
    "MARL",
    "DRL"
]

C_COMPARISON = [
    "exploration",
    "avoidance",
    "planning",
    "formation",
    "coordination"
]

O_OUTCOME = [
    "performance",
    "efficiency",
    "adaptability",
    "robustness",
    "scalability"
]

# ===============================
# SEARCH QUERIES (PICO-ALIGNED)
# ===============================
SEARCH_QUERIES = {
    "pico_core": f"""
    ({' OR '.join(P_POPULATION)}) AND
    ({' OR '.join(I_INTERVENTION)}) AND
    ({' OR '.join(C_COMPARISON)}) AND
    ({' OR '.join(O_OUTCOME)})
    """,

    "pico_control_and_navigation": f"""
    ({' OR '.join(P_POPULATION)}) AND
    ({' OR '.join(I_INTERVENTION)}) AND
    (planning OR navigation OR collision avoidance OR formation control)
    """,

    "pico_coordination_and_cooperation": f"""
    ({' OR '.join(P_POPULATION)}) AND
    ({' OR '.join(I_INTERVENTION)}) AND
    (coordination OR cooperation OR decentralized execution)
    """,

    "pico_scalability_and_robustness": f"""
    ({' OR '.join(P_POPULATION)}) AND
    ({' OR '.join(I_INTERVENTION)}) AND
    (scalability OR robustness OR fault tolerance)
    """,

    "pico_performance_metrics": f"""
    ({' OR '.join(P_POPULATION)}) AND
    ({' OR '.join(I_INTERVENTION)}) AND
    (performance evaluation OR efficiency OR success rate OR energy efficiency)
    """
}
