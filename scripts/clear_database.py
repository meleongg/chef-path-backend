#!/usr/bin/env python3
"""
Database Clear Script for ChefPath Backend

This script removes all data from the database tables while keeping the table structure.
Use this to reset the database to a clean state.

Usage:
    python scripts/clear_database.py
"""

import sys
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from sqlalchemy.orm import Session
from app.database import engine, get_db
from app.models import User, Recipe, WeeklyPlan, UserRecipeProgress


def clear_database():
    """Clear all data from database tables"""
    print("🗑️  Clearing ChefPath Database...")

    # Create a database session
    db = Session(bind=engine)

    try:
        # Delete in reverse order of dependencies to avoid foreign key constraints
        print("  - Clearing user recipe progress...")
        db.query(UserRecipeProgress).delete()

        print("  - Clearing weekly plans...")
        db.query(WeeklyPlan).delete()

        print("  - Clearing recipes...")
        db.query(Recipe).delete()

        print("  - Clearing users...")
        db.query(User).delete()

        # Commit all deletions
        db.commit()
        print("✅ Database cleared successfully!")

        # Show table counts to verify
        print("\n📊 Table counts after clearing:")
        print(f"  - Users: {db.query(User).count()}")
        print(f"  - Recipes: {db.query(Recipe).count()}")
        print(f"  - Weekly Plans: {db.query(WeeklyPlan).count()}")
        print(f"  - User Recipe Progress: {db.query(UserRecipeProgress).count()}")

    except Exception as e:
        print(f"❌ Error clearing database: {e}")
        db.rollback()
        return False
    finally:
        db.close()

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clear ChefPath database")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    if not args.force:
        response = input(
            "⚠️  This will delete ALL data from the database. Continue? (y/N): "
        )
        if response.lower() != "y":
            print("Operation cancelled.")
            sys.exit(0)

    success = clear_database()
    sys.exit(0 if success else 1)
