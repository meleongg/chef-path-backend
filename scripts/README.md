# Database Management Scripts

This directory contains scripts for managing the ChefPath backend database.

## Quick Start

Use the convenient wrapper script:

```bash
# Full database reset (clear + seed with sample data)
python manage_db.py reset

# View database contents
python manage_db.py view --summary
```

## Available Scripts

### 1. Database Management Wrapper (`manage_db.py`)

Convenient wrapper for all database operations:

```bash
python manage_db.py clear              # Clear database
python manage_db.py seed               # Seed with sample data
python manage_db.py seed --clear       # Clear then seed
python manage_db.py view               # View all data
python manage_db.py view --summary     # View summary only
python manage_db.py reset              # Clear and seed (full reset)
```

### 2. Clear Database (`scripts/clear_database.py`)

Removes all data while keeping table structure:

```bash
python scripts/clear_database.py
python scripts/clear_database.py --force  # Skip confirmation
```

### 3. Seed Database (`scripts/seed_database.py`)

Populates database with sample data:

```bash
python scripts/seed_database.py
python scripts/seed_database.py --clear   # Clear first
```

**Sample data includes:**

- 4 sample users with different cuisines and skill levels
- Real recipes fetched from TheMealDB API
- Generated weekly plans for each user
- Simulated progress for demonstration

### 4. View Database (`scripts/view_database.py`)

Displays database contents in readable format:

```bash
python scripts/view_database.py                    # View all tables
python scripts/view_database.py --summary          # Summary only
python scripts/view_database.py --table users     # Specific table
python scripts/view_database.py --limit 5         # Limit records shown
```

## Sample Users Created

The seed script creates these sample users:

1. **Alice Johnson** - Italian cuisine, beginner, 8-week course
2. **Bob Chen** - Chinese cuisine, intermediate, 12-week course
3. **Carol Martinez** - Mexican cuisine, advanced, 6-week course
4. **David Smith** - American cuisine, beginner, 10-week course

## Development Workflow

1. **Start fresh**: `python manage_db.py reset`
2. **Develop your feature**
3. **Check data**: `python manage_db.py view --summary`
4. **Clear when needed**: `python manage_db.py clear`

## Notes

- All scripts can be run from the backend root directory
- Scripts will create tables automatically if they don't exist
- The seed script fetches real recipes from TheMealDB API
- Use `--help` with any script for detailed options
