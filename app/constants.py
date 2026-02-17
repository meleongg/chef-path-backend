# Embedding and vectorization configuration for ChefPath

GENERATIVE_MODEL = "gpt-4o-mini"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
BATCH_SIZE = 50

SECRET_KEY = "your-secret-key"  # Replace with a secure key in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS = 14
REFRESH_TOKEN_COOKIE_NAME = "chefpath_refresh"
REFRESH_TOKEN_COOKIE_PATH = "/"
REFRESH_TOKEN_COOKIE_SAMESITE = "lax"
REFRESH_TOKEN_COOKIE_SECURE = False

# Cooking goal expanded descriptions (for LLM prompts)
COOKING_GOAL_DESCRIPTIONS = {
    "techniques": "Learn New Techniques (e.g., knife skills, sauces, dough)",
    "cuisine": "Master a Cuisine (e.g., Italian, Thai, Baking)",
    "health": "Eat Healthier (e.g., weight loss, more vegetables, less sodium)",
    "efficiency": "Cook Faster/More Efficiently (e.g., weeknight meals, meal prep)",
    "confidence": "Gain Confidence (e.g., hosting dinner parties, improvising recipes)",
}

# Skill level descriptions (for LLM prompts)
SKILL_LEVEL_DESCRIPTIONS = {
    "beginner": "Beginner - Just starting out, needs simple recipes with clear instructions",
    "intermediate": "Intermediate - Knows the basics, ready for moderate complexity",
    "advanced": "Advanced - Experienced cook, can handle complex techniques",
}

# Cuisine options (for reference/validation)
CUISINE_OPTIONS = ["Italian", "Chinese", "Mexican", "American"]

THEMEALDB_BASE_URL = "https://www.themealdb.com/api/json/v1/1"

# Recipe swap configuration
MAX_SWAPS_PER_WEEK = 3
