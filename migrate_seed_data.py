import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# --- Configuration ---
from app.models import User, Recipe, WeeklyPlan, UserRecipeProgress

load_dotenv()

# --- Connection Strings ---
SQLITE_URL = "sqlite:///./database.db"
POSTGRES_URL = os.environ.get("DATABASE_URL")

# --- Engines and Sessions ---
# Source: SQLite
SQLiteEngine = create_engine(SQLITE_URL)
SQLiteSession = sessionmaker(bind=SQLiteEngine)

# Target: Supabase PostgreSQL
PostgresEngine = create_engine(POSTGRES_URL)
PostgresSession = sessionmaker(bind=PostgresEngine)

print("Setup complete. Ready to migrate.")


def migrate_data():
    """Reads data from SQLite and writes it to Supabase PostgreSQL."""
    pg_session = PostgresSession()
    lite_session = SQLiteSession()

    try:
        # A. Migrate independent tables first (Recipe, User)
        print("Migrating Recipes...")
        recipes_to_migrate = lite_session.query(Recipe).all()
        for recipe in recipes_to_migrate:
            pg_session.merge(recipe)

        print("Migrating Users...")
        users_to_migrate = lite_session.query(User).all()
        for user in users_to_migrate:
            pg_session.merge(user)

        pg_session.commit()
        print(
            f"Successfully migrated {len(recipes_to_migrate)} Recipes and {len(users_to_migrate)} Users."
        )

        # B. Migrate dependent tables (WeeklyPlan, UserRecipeProgress)
        print("Migrating Weekly Plans...")
        plans_to_migrate = lite_session.query(WeeklyPlan).all()
        for plan in plans_to_migrate:
            pg_session.merge(plan)

        print("Migrating User Progress Records...")
        progress_to_migrate = lite_session.query(UserRecipeProgress).all()
        for progress in progress_to_migrate:
            pg_session.merge(progress)

        pg_session.commit()
        print("Dependent data migrated successfully.")

    except Exception as e:
        print(f"Migration FAILED: {e}")
        pg_session.rollback()

    finally:
        pg_session.close()
        lite_session.close()


if __name__ == "__main__":
    migrate_data()
