#!/usr/bin/env python3
"""
Database Viewer Script for ChefPath Backend

This script displays all data in the database in a readable format.
Use this to inspect the current state of your database.

Usage:
    python scripts/view_database.py
    python scripts/view_database.py --table users     # View specific table
    python scripts/view_database.py --summary         # Show summary only
"""

import sys
from pathlib import Path
from datetime import datetime

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from sqlalchemy.orm import Session
from app.database import engine
from app.models import User, Recipe, WeeklyPlan, UserRecipeProgress
from typing import Optional


def view_table(db: Session, table_name: str, model_class, limit: Optional[int] = None):
    """View data from a specific table"""
    print(f"📊 TABLE: {table_name.upper()}")
    print("-" * 50)

    # Get count
    count = db.query(model_class).count()
    print(f"Total records: {count}")

    if count == 0:
        print("No records found")
        print()
        return

    # Get data (with optional limit)
    query = db.query(model_class)
    if limit:
        query = query.limit(limit)
    records = query.all()

    # Display records
    for i, record in enumerate(records, 1):
        print(f"\nRecord {i}:")

        # Get all attributes dynamically
        for column in model_class.__table__.columns:
            attr_name = column.name
            value = getattr(record, attr_name)

            # Format datetime values
            if isinstance(value, datetime):
                value = value.strftime("%Y-%m-%d %H:%M:%S UTC")

            print(f"  {attr_name}: {value}")

        if i >= 10 and not limit:  # Limit display to 10 records by default
            remaining = count - 10
            if remaining > 0:
                print(f"  ... and {remaining} more records")
            break

    print("-" * 50)
    print()


def view_database_summary(db: Session):
    """Show a quick summary of the database"""
    print("=" * 60)
    print("CHEFPATH DATABASE SUMMARY")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Table counts
    tables = [
        ("users", User),
        ("recipes", Recipe),
        ("weekly_plans", WeeklyPlan),
        ("user_recipe_progress", UserRecipeProgress),
    ]

    print("📊 TABLE COUNTS:")
    for table_name, model_class in tables:
        count = db.query(model_class).count()
        print(f"  {table_name}: {count}")

    print()

    # Quick insights
    users = db.query(User).all()
    if users:
        print("👥 USERS OVERVIEW:")
        for user in users:
            progress_count = (
                db.query(UserRecipeProgress)
                .filter(UserRecipeProgress.user_id == user.id)
                .count()
            )
            completed_count = (
                db.query(UserRecipeProgress)
                .filter(
                    UserRecipeProgress.user_id == user.id,
                )
                .filter(UserRecipeProgress.status == "completed")
                .count()
            )

            completion_rate = (
                (completed_count / progress_count * 100) if progress_count > 0 else 0
            )

            print(
                f"  - {user.name}: {getattr(user, 'cuisine')} cuisine, "
                f"{getattr(user, 'skill_level')} level, "
                f"{completed_count}/{progress_count} recipes completed "
                f"({completion_rate:.1f}%)"
            )

    print()


def view_full_database(db: Session, limit: Optional[int] = None):
    """View all tables in the database"""
    print("=" * 60)
    print("CHEFPATH DATABASE CONTENTS")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    tables = [
        ("users", User),
        ("recipes", Recipe),
        ("weekly_plans", WeeklyPlan),
        ("user_recipe_progress", UserRecipeProgress),
    ]

    for table_name, model_class in tables:
        view_table(db, table_name, model_class, limit)


def main():
    """Main function to handle command line arguments"""
    import argparse

    parser = argparse.ArgumentParser(description="View ChefPath database contents")
    parser.add_argument(
        "--table",
        choices=["users", "recipes", "weekly_plans", "user_recipe_progress"],
        help="View specific table only",
    )
    parser.add_argument("--summary", action="store_true", help="Show summary only")
    parser.add_argument(
        "--limit", type=int, help="Limit number of records to display per table"
    )

    args = parser.parse_args()

    # Create database session
    db = Session(bind=engine)

    try:
        if args.summary:
            view_database_summary(db)
        elif args.table:
            # View specific table
            table_models = {
                "users": User,
                "recipes": Recipe,
                "weekly_plans": WeeklyPlan,
                "user_recipe_progress": UserRecipeProgress,
            }
            view_table(db, args.table, table_models[args.table], args.limit)
        else:
            # View all tables
            view_full_database(db, args.limit)

    finally:
        db.close()


if __name__ == "__main__":
    main()
