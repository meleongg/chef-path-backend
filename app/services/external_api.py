import httpx
import json
from typing import Dict, Optional
from sqlalchemy.orm import Session
from app.models import Recipe

class MealDBAcquisitionService:
    BASE_URL = "https://www.themealdb.com/api/json/v1/1"

    def __init__(self):
        self.client = httpx.AsyncClient()

    async def close(self):
        await self.client.aclose()

    async def search_by_cuisine(self, cuisine: str):
        try:
            response = await self.client.get(f"{self.BASE_URL}/filter.php?a={cuisine}")
            response.raise_for_status()
            data = response.json()
            return data.get("meals", []) or []
        except httpx.RequestError:
            return []

    async def get_recipe_details(self, meal_id: str) -> Optional[Dict]:
        try:
            response = await self.client.get(f"{self.BASE_URL}/lookup.php?i={meal_id}")
            response.raise_for_status()
            data = response.json()
            meals = data.get("meals", [])
            return meals[0] if meals else None
        except httpx.RequestError:
            return None

    async def get_random_recipes(self, count: int = 5):
        recipes = []
        for _ in range(count):
            try:
                response = await self.client.get(f"{self.BASE_URL}/random.php")
                response.raise_for_status()
                data = response.json()
                meals = data.get("meals", [])
                if meals:
                    recipes.append(meals[0])
            except httpx.RequestError:
                continue
        return recipes

    def determine_difficulty(self, instructions: str, ingredients_count: int) -> str:
        instruction_length = len(instructions.split())
        if ingredients_count <= 5 and instruction_length <= 50:
            return "easy"
        elif ingredients_count <= 10 and instruction_length <= 150:
            return "medium"
        else:
            return "hard"

    def format_ingredients(self, meal_data: Dict) -> str:
        ingredients = []
        for i in range(1, 21):
            ingredient = meal_data.get(f"strIngredient{i}")
            measure = meal_data.get(f"strMeasure{i}")
            if ingredient and ingredient.strip():
                ingredient_obj = {
                    "name": ingredient.strip(),
                    "measure": measure.strip() if measure else "",
                }
                ingredients.append(ingredient_obj)
        return json.dumps(ingredients)

    def generate_content_text(self, meal_data: Dict, difficulty: str, ingredients_json: str) -> str:
        """Generate content_text for a recipe from TheMealDB API data."""
        return (
            f"Recipe Name: {meal_data.get('strMeal', '')} "
            f"Cuisine: {meal_data.get('strArea', '')} "
            f"Difficulty: {difficulty} "
            f"Tags: {meal_data.get('strTags', 'none')} "
            f"Ingredients: {ingredients_json} "
            f"Instructions: {meal_data.get('strInstructions', '')}"
        )

    async def save_recipe_to_db(self, meal_data: Dict, db: Session) -> Recipe:
        existing_recipe = db.query(Recipe).filter(Recipe.external_id == meal_data["idMeal"]).first()
        if existing_recipe:
            return existing_recipe
        ingredients_json = self.format_ingredients(meal_data)
        ingredients_count = len(json.loads(ingredients_json))
        difficulty = self.determine_difficulty(meal_data.get("strInstructions", ""), ingredients_count)
        content_text = self.generate_content_text(meal_data, difficulty, ingredients_json)
        recipe = Recipe(
            external_id=meal_data["idMeal"],
            name=meal_data["strMeal"],
            cuisine=meal_data.get("strArea", "Unknown"),
            ingredients=ingredients_json,
            instructions=meal_data.get("strInstructions", ""),
            difficulty=difficulty,
            tags=meal_data.get("strTags", ""),
            image_url=meal_data.get("strMealThumb"),
            is_ai_generated=False,
            content_text=content_text,
        )
        db.add(recipe)
        db.commit()
        db.refresh(recipe)
        return recipe
