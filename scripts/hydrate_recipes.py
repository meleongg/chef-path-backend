import httpx
import os
import sys
import string
from sqlalchemy.orm import Session
from sqlalchemy import update
from typing import Set

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, project_root)

from app.database import engine
from app.models import Recipe
from app.constants import THEMEALDB_BASE_URL


# Helper function to generate content_text for a recipe
def generate_content_text(recipe_data):
    import json

    ingredients_list = json.loads(recipe_data.get("ingredients", "[]"))
    ingredients_names = [item["name"] for item in ingredients_list if "name" in item]
    content = f"""
    Recipe Name: {recipe_data.get('name')}
    Cuisine: {recipe_data.get('cuisine')}
    Difficulty: {recipe_data.get('difficulty')}
    Tags: {recipe_data.get('tags', 'none')}
    Ingredients: {', '.join(ingredients_names)}
    Instructions: {recipe_data.get('instructions')}
    """
    return " ".join(content.split()).strip()


async def fetch_all_meals_by_alpha():
    """
    Fetches all meals from TheMealDB by looping through the alphabet (A-Z).
    Returns a set of all unique meal IDs found.
    """
    all_meal_ids: Set[str] = set()
    async with httpx.AsyncClient() as client:
        for letter in string.ascii_lowercase:
            url = f"{THEMEALDB_BASE_URL}/search.php?f={letter}"
            try:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                meals = data.get("meals", [])
                if meals:
                    for meal in meals:
                        if meal and meal.get("idMeal"):
                            all_meal_ids.add(meal["idMeal"])
            except Exception as e:
                print(f"Warning: Failed to fetch meals for letter {letter}. Error: {e}")
                continue
    return all_meal_ids


async def hydrate_recipes_exhaustive():
    """Fetches ALL unique recipes from TheMealDB and saves/enriches them."""
    db_session = Session(bind=engine)
    try:
        total_ids = await fetch_all_meals_by_alpha()
        existing_ids = set(
            str(r.external_id)
            for r in db_session.query(Recipe.external_id).all()
            if r.external_id
        )
        ids_to_fetch = total_ids - existing_ids
        print(f"Found {len(total_ids)} unique recipes in TheMealDB index.")
        print(f"Need to fetch and save {len(ids_to_fetch)} new recipes.")
        for meal_id in ids_to_fetch:
            details = await get_recipe_details(meal_id)
            if details and details.get("strInstructions"):
                recipe_obj = save_recipe_to_db(details, db_session)
                new_content = generate_content_text(recipe_obj.__dict__)
                db_session.execute(
                    update(Recipe)
                    .where(Recipe.id == recipe_obj.id)
                    .values(content_text=new_content)
                )
                db_session.commit()
                print(f"  -> Added and enriched: {recipe_obj.name}")
        final_count = db_session.query(Recipe).count()
        print(f"\n🎉 Hydration complete. Total recipes in DB: {final_count}")
    except Exception as e:
        db_session.rollback()
        print(f"❌ Critical error during exhaustive hydration: {e}")
    finally:
        db_session.close()


async def get_recipe_details(meal_id: str):
    url = f"{THEMEALDB_BASE_URL}/lookup.php?i={meal_id}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            meals = data.get("meals", [])
            if meals:
                return meals[0]
            return None
        except Exception as e:
            print(f"Warning: Failed to fetch details for meal {meal_id}. Error: {e}")
            return None


def save_recipe_to_db(details, db_session):
    # Placeholder: Replace with your actual Recipe model logic
    recipe = Recipe(
        external_id=details.get("idMeal"),
        name=details.get("strMeal"),
        cuisine=details.get("strArea"),
        difficulty="medium",  # Default or derive from details
        tags=details.get("strTags"),
        ingredients="[]",  # Should be parsed from details
        instructions=details.get("strInstructions"),
        is_ai_generated=False,
    )
    db_session.add(recipe)
    db_session.commit()
    return recipe
