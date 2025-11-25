# ChefPath Backend Scripts

This directory contains scripts for managing and maintaining the ChefPath backend database and AI infrastructure.

## Available Scripts

### Hydrate Recipes (`scripts/hydrate_recipes.py`)

Fetches and stores all available recipes from TheMealDB API using exhaustive search:

```bash
python scripts/hydrate_recipes.py
```

- Populates the database with all unique recipes.
- Ensures each recipe is enriched for downstream AI use.

### Generate Embeddings (`scripts/generate_embeddings.py`)

Generates vector embeddings for all recipes missing an embedding:

```bash
python scripts/generate_embeddings.py
```

- Uses OpenAI's `text-embedding-3-small` model.
- Optimized for batch processing and cost efficiency.

### Clear Database (`scripts/clear_database.py`)

Removes all data while keeping table structure:

```bash
python scripts/clear_database.py
python scripts/clear_database.py --force  # Skip confirmation
```
