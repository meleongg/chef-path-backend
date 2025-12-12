"""
Optimized Recipe Processing Pipeline
Fetches from TheMealDB → AI augments → generates embeddings → saves to DB in one pass
"""

import httpx
import os
import sys
import string
import json
import asyncio
from typing import Set, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, project_root)

from app.database import engine
from app.models import Recipe
from app.constants import THEMEALDB_BASE_URL, GENERATIVE_MODEL, EMBEDDING_MODEL
from app.utils.recipe_formatters import instructions_json_to_text


class InstructionStep(BaseModel):
    step: int
    text: str


class RecipeMetadata(BaseModel):
    dietary_tags: List[str]
    allergens: List[str]
    portion_size: str
    prep_time_minutes: int
    cook_time_minutes: int
    skill_level_validated: str
    instructions_formatted: List[InstructionStep]


# Initialize clients
llm = ChatOpenAI(model=GENERATIVE_MODEL, temperature=0.3)
embeddings_client = OpenAIEmbeddings(model=EMBEDDING_MODEL)


async def fetch_all_meal_ids() -> Set[str]:
    """Fetch all unique meal IDs from TheMealDB"""
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
                print(f"⚠️  Failed to fetch letter {letter}: {e}")
    return all_meal_ids


async def fetch_recipe_details(meal_id: str) -> Optional[dict]:
    """Fetch full recipe details from TheMealDB"""
    url = f"{THEMEALDB_BASE_URL}/lookup.php?i={meal_id}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            meals = data.get("meals", [])
            return meals[0] if meals else None
        except Exception as e:
            print(f"⚠️  Failed to fetch meal {meal_id}: {e}")
            return None


def parse_mealdb_ingredients(meal_data: dict) -> List[dict]:
    """Extract ingredients from TheMealDB format"""
    ingredients = []
    for i in range(1, 21):  # TheMealDB has up to 20 ingredients
        ingredient = meal_data.get(f"strIngredient{i}", "").strip()
        measure = meal_data.get(f"strMeasure{i}", "").strip()
        if ingredient:
            ingredients.append({"name": ingredient, "measure": measure or "to taste"})
    return ingredients


async def augment_recipe_with_ai(
    meal_data: dict, ingredients: List[dict]
) -> Optional[RecipeMetadata]:
    """Use AI to augment recipe with metadata and formatted instructions"""
    try:
        ingredients_text = ", ".join(
            [f"{item['name']} ({item['measure']})" for item in ingredients]
        )

        system_prompt = """You are a professional chef and nutritionist analyzing recipes.
Extract accurate metadata and reformat instructions into clean, structured steps.
Be conservative with allergens. Provide realistic time estimates."""

        user_prompt = f"""Analyze this recipe:

Recipe: {meal_data.get('strMeal')}
Cuisine: {meal_data.get('strArea')}
Ingredients: {ingredients_text}
Instructions: {meal_data.get('strInstructions')}

Provide: dietary tags, allergens, portion size, prep/cook times, skill level, and formatted instruction steps."""

        structured_llm = llm.with_structured_output(RecipeMetadata)
        metadata = structured_llm.invoke(
            [HumanMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )

        return metadata
    except Exception as e:
        print(f"⚠️  AI augmentation failed: {e}")
        return None


def generate_content_text(
    name: str, cuisine: str, ingredients: List[dict], instructions_json: List[dict]
) -> str:
    """Generate content_text for embedding"""
    ingredients_text = " ".join(
        [f"{item['name']} {item['measure']}" for item in ingredients]
    )
    instructions_text = instructions_json_to_text(instructions_json)
    return f"{name} {cuisine} {ingredients_text} {instructions_text}"


async def process_single_recipe(meal_id: str, db: Session) -> bool:
    """Process one recipe: fetch → augment → embed → save"""
    try:
        # Fetch from TheMealDB
        meal_data = await fetch_recipe_details(meal_id)
        if not meal_data or not meal_data.get("strInstructions"):
            return False

        # Parse ingredients
        ingredients = parse_mealdb_ingredients(meal_data)

        # AI augmentation
        metadata = await augment_recipe_with_ai(meal_data, ingredients)
        if not metadata:
            return False

        # Prepare instructions JSON
        instructions_json = [
            {"step": s.step, "text": s.text} for s in metadata.instructions_formatted
        ]

        # Generate content_text for embedding
        content_text = generate_content_text(
            meal_data.get("strMeal"),
            meal_data.get("strArea"),
            ingredients,
            instructions_json,
        )

        # Generate embedding
        embedding = embeddings_client.embed_query(content_text)

        # Save to database with ALL data at once
        recipe = Recipe(
            external_id=meal_data.get("idMeal"),
            name=meal_data.get("strMeal"),
            cuisine=meal_data.get("strArea"),
            ingredients=json.dumps(ingredients),
            instructions=json.dumps(instructions_json),
            difficulty=metadata.skill_level_validated,
            tags=meal_data.get("strTags"),
            image_url=meal_data.get("strMealThumb"),
            is_ai_generated=False,
            # Augmented metadata
            dietary_tags=json.dumps(metadata.dietary_tags),
            allergens=json.dumps(metadata.allergens),
            portion_size=metadata.portion_size,
            prep_time_minutes=metadata.prep_time_minutes,
            cook_time_minutes=metadata.cook_time_minutes,
            skill_level_validated=metadata.skill_level_validated,
            # Embedding fields
            content_text=content_text,
            embedding=embedding,
        )

        db.add(recipe)
        db.commit()

        print(
            f"✅ {meal_data.get('strMeal')} | {metadata.dietary_tags[:2]} | {metadata.prep_time_minutes}+{metadata.cook_time_minutes}min"
        )
        return True

    except Exception as e:
        print(f"❌ Failed to process {meal_id}: {e}")
        db.rollback()
        return False


async def process_all_recipes(batch_size: int = 5, limit: Optional[int] = None):
    """Main pipeline: fetch all recipes and process them"""
    db = Session(bind=engine)

    try:
        # Get all meal IDs from TheMealDB
        print("\n🔍 Fetching recipe list from TheMealDB...")
        all_meal_ids = await fetch_all_meal_ids()

        # Check what's already in DB
        existing_ids = set(
            str(r.external_id)
            for r in db.scalars(select(Recipe.external_id)).all()
            if r.external_id
        )

        ids_to_process = list(all_meal_ids - existing_ids)

        if limit:
            ids_to_process = ids_to_process[:limit]

        total = len(ids_to_process)
        print(f"\n📊 Found {len(all_meal_ids)} total recipes")
        print(f"   Already in DB: {len(existing_ids)}")
        print(f"   To process: {total}")
        print(f"   Batch size: {batch_size}\n")

        if total == 0:
            print("✅ All recipes already processed!")
            return

        success_count = 0
        fail_count = 0

        # Process in batches
        for i in range(0, total, batch_size):
            batch_ids = ids_to_process[i : i + batch_size]
            batch_num = i // batch_size + 1

            print(
                f"\n--- Batch {batch_num} ({i + 1}-{min(i + batch_size, total)} of {total}) ---"
            )

            # Process batch concurrently
            tasks = [process_single_recipe(meal_id, db) for meal_id in batch_ids]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, bool) and result:
                    success_count += 1
                else:
                    fail_count += 1

            print(f"Progress: {success_count} ✅ | {fail_count} ❌")

        print(f"\n{'='*60}")
        print(f"🎉 Processing complete!")
        print(f"   Success: {success_count}/{total}")
        print(f"   Failed: {fail_count}/{total}")
        print(f"{'='*60}\n")

    finally:
        db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process recipes: fetch → augment → embed → save"
    )
    parser.add_argument(
        "--batch-size", type=int, default=5, help="Concurrent processing batch size"
    )
    parser.add_argument(
        "--limit", type=int, help="Limit recipes to process (for testing)"
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"🚀 Optimized Recipe Processing Pipeline")
    print(f"{'='*60}")

    asyncio.run(process_all_recipes(batch_size=args.batch_size, limit=args.limit))
