#!/usr/bin/env python3
"""
Database Seed Script for ChefPath Backend

This script populates the database with sample data for testing and development.
It creates sample users, fetches real recipes from TheMealDB, and sets up weekly plans.

Usage:
    python scripts/seed_database.py
    python scripts/seed_database.py --clear  # Clear database first
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime, timezone

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from sqlalchemy.orm import Session
from app.database import engine, create_tables
from app.models import User, Recipe, WeeklyPlan, UserRecipeProgress
from app.services.themealdb import TheMealDBService
from app.services.weekly_plan import WeeklyPlanService


async def seed_database(clear_first: bool = False):
    """Seed database with sample data"""
    print("🌱 Seeding ChefPath Database...")

    # Clear database if requested
    if clear_first:
        from scripts.clear_database import clear_database

        print("Clearing existing data first...")
        clear_database()

    # Ensure tables exist
    create_tables()

    # Create database session
    db = Session(bind=engine)

    # Initialize services
    meal_service = TheMealDBService()
    plan_service = WeeklyPlanService()

    try:
        # Create sample users
        print("\n👥 Creating sample users...")
        sample_users = [
            {
                "name": "Alice Johnson",
                "cuisine": "Italian",
                "frequency": 3,
                "skill_level": "beginner",
                "course_duration": 8,
            },
            {
                "name": "Bob Chen",
                "cuisine": "Chinese",
                "frequency": 4,
                "skill_level": "intermediate",
                "course_duration": 12,
            },
            {
                "name": "Carol Martinez",
                "cuisine": "Mexican",
                "frequency": 2,
                "skill_level": "advanced",
                "course_duration": 6,
            },
            {
                "name": "David Smith",
                "cuisine": "American",
                "frequency": 5,
                "skill_level": "beginner",
                "course_duration": 10,
            },
        ]

        created_users = []
        for user_data in sample_users:
            user = User(**user_data)
            db.add(user)
            db.commit()
            db.refresh(user)
            created_users.append(user)
            print(f"  ✅ Created user: {user.name} (ID: {user.id})")

        # Generate some recipes by fetching from TheMealDB
        print(f"\n🍽️  Fetching sample recipes from TheMealDB...")
        cuisines = ["Italian", "Chinese", "Mexican", "American"]
        all_recipes = []

        for cuisine in cuisines:
            print(f"  - Fetching {cuisine} recipes...")
            try:
                recipes = await meal_service.get_recipes_for_user(
                    cuisine.lower(), "beginner", 3, db
                )
                all_recipes.extend(recipes)
                print(f"    ✅ Added {len(recipes)} {cuisine} recipes")
            except Exception as e:
                print(f"    ⚠️  Could not fetch {cuisine} recipes: {e}")

        print(f"  📊 Total recipes in database: {db.query(Recipe).count()}")

        # Generate weekly plans for each user
        print(f"\n📅 Generating weekly plans...")
        for user in created_users:
            try:
                # Generate first 2 weeks for each user
                for week in [1, 2]:
                    plan = await plan_service.generate_weekly_plan(user, week, db)
                    print(f"  ✅ Generated week {week} plan for {user.name}")

                # Simulate some progress for demonstration
                if user.name == "Alice Johnson":
                    # Mark some recipes as completed for Alice
                    progress_entries = (
                        db.query(UserRecipeProgress)
                        .filter(
                            UserRecipeProgress.user_id == user.id,
                            UserRecipeProgress.week_number == 1,
                        )
                        .limit(2)
                        .all()
                    )

                    for progress in progress_entries:
                        setattr(progress, "status", "completed")
                        setattr(progress, "feedback", "just_right")
                        setattr(progress, "completed_at", datetime.now(timezone.utc))

                    db.commit()
                    print(f"  ✅ Simulated progress for {user.name}")

            except Exception as e:
                print(f"  ⚠️  Could not generate plans for {user.name}: {e}")

        # Display summary
        print(f"\n📊 Database seeding complete! Summary:")
        print(f"  - Users: {db.query(User).count()}")
        print(f"  - Recipes: {db.query(Recipe).count()}")
        print(f"  - Weekly Plans: {db.query(WeeklyPlan).count()}")
        print(f"  - Progress Entries: {db.query(UserRecipeProgress).count()}")

        # Show sample data
        print(f"\n👥 Sample Users:")
        users = db.query(User).all()
        for user in users:
            print(
                f"  - {user.name} (ID: {user.id}) - {getattr(user, 'cuisine')} cuisine, {getattr(user, 'skill_level')} level"
            )

        return True

    except Exception as e:
        print(f"❌ Error seeding database: {e}")
        db.rollback()
        return False
    finally:
        await meal_service.close()
        await plan_service.close()
        db.close()


def main():
    """Main function to handle command line arguments"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Seed ChefPath database with sample data"
    )
    parser.add_argument(
        "--clear", action="store_true", help="Clear database before seeding"
    )
    args = parser.parse_args()

    # Run the async seeding function
    success = asyncio.run(seed_database(clear_first=args.clear))

    if success:
        print("\n🎉 Database seeding completed successfully!")
        print("\n💡 Next steps:")
        print("  1. Start the API server: python main.py")
        print("  2. Test endpoints with the sample users")
        print("  3. Use scripts/view_database.py to inspect the data")
    else:
        print("\n❌ Database seeding failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
