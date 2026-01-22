import random

ADJECTIVES = [
    "Silent",
    "Hidden",
    "Neon",
    "Deep",
    "Ancient",
    "Urban",
    "Subterranean",
    "Forgotten",
    "Shadowed",
    "Flooded",
    "Concrete",
    "Steel",
]

NOUNS = [
    "Tunnel",
    "River",
    "Depths",
    "City",
    "Passage",
    "Vault",
    "Drain",
    "Chamber",
    "System",
    "Network",
]


def random_chat_name():
    return f"{random.choice(ADJECTIVES)} {random.choice(NOUNS)}"
