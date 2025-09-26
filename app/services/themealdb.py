import httpx
import json
import random
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from app.models import Recipe


class TheMealDBService:
    BASE_URL = "https://www.themealdb.com/api/json/v1/1"

    def __init__(self):
        self.client = httpx.AsyncClient()

    async def close(self):
        await self.client.aclose()

    async def search_by_cuisine(self, cuisine: str) -> List[Dict]:
        """Search recipes by cuisine from TheMealDB"""
        try:
            response = await self.client.get(f"{self.BASE_URL}/filter.php?a={cuisine}")
            response.raise_for_status()
            data = response.json()
            return data.get("meals", []) or []
        except httpx.RequestError:
            return []

    async def get_recipe_details(self, meal_id: str) -> Optional[Dict]:
        """Get detailed recipe information by meal ID"""
        try:
            response = await self.client.get(f"{self.BASE_URL}/lookup.php?i={meal_id}")
            response.raise_for_status()
            data = response.json()
            meals = data.get("meals", [])
            return meals[0] if meals else None
        except httpx.RequestError:
            return None

    async def get_random_recipes(self, count: int = 5) -> List[Dict]:
        """Get random recipes from TheMealDB"""
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
        """Determine recipe difficulty based on instructions length and ingredient count"""
        instruction_length = len(instructions.split())

        if ingredients_count <= 5 and instruction_length <= 50:
            return "easy"
        elif ingredients_count <= 10 and instruction_length <= 150:
            return "medium"
        else:
            return "hard"

    def format_ingredients(self, meal_data: Dict) -> str:
        """Format ingredients from TheMealDB format to JSON string"""
        ingredients = []
        for i in range(1, 21):  # TheMealDB has up to 20 ingredients
            ingredient = meal_data.get(f"strIngredient{i}")
            measure = meal_data.get(f"strMeasure{i}")

            if ingredient and ingredient.strip():
                ingredient_obj = {
                    "name": ingredient.strip(),
                    "measure": measure.strip() if measure else "",
                }
                ingredients.append(ingredient_obj)

        return json.dumps(ingredients)

    async def save_recipe_to_db(self, meal_data: Dict, db: Session) -> Recipe:
        """Save a recipe from TheMealDB to local database"""
        # Check if recipe already exists
        existing_recipe = (
            db.query(Recipe).filter(Recipe.external_id == meal_data["idMeal"]).first()
        )

        if existing_recipe:
            return existing_recipe

        # Format ingredients and determine difficulty
        ingredients_json = self.format_ingredients(meal_data)
        ingredients_count = len(json.loads(ingredients_json))
        difficulty = self.determine_difficulty(
            meal_data.get("strInstructions", ""), ingredients_count
        )

        # Create new recipe
        recipe = Recipe(
            external_id=meal_data["idMeal"],
            name=meal_data["strMeal"],
            cuisine=meal_data.get("strArea", "Unknown"),
            ingredients=ingredients_json,
            instructions=meal_data.get("strInstructions", ""),
            difficulty=difficulty,
            tags=meal_data.get("strTags", ""),
            image_url=meal_data.get("strMealThumb"),
        )

        db.add(recipe)
        db.commit()
        db.refresh(recipe)
        return recipe

    async def get_recipes_for_user(
        self, user_cuisine: str, user_skill: str, count: int, db: Session
    ) -> List[Recipe]:
        """Get recipes suitable for user's preferences"""
        recipes = []

        # First try to get recipes from user's preferred cuisine
        if user_cuisine.lower() != "any":
            cuisine_recipes = await self.search_by_cuisine(user_cuisine)

            # Get detailed info and save to DB
            for meal in cuisine_recipes[: count * 2]:  # Get more than needed to filter
                details = await self.get_recipe_details(meal["idMeal"])
                if details:
                    recipe = await self.save_recipe_to_db(details, db)

                    # Filter by skill level
                    if self.is_suitable_for_skill(str(recipe.difficulty), user_skill):
                        recipes.append(recipe)
                        if len(recipes) >= count:
                            break

        # If we don't have enough recipes, get random ones
        if len(recipes) < count:
            remaining_count = count - len(recipes)
            random_meals = await self.get_random_recipes(remaining_count * 3)

            for meal in random_meals:
                if len(recipes) >= count:
                    break

                recipe = await self.save_recipe_to_db(meal, db)
                if self.is_suitable_for_skill(str(recipe.difficulty), user_skill):
                    recipes.append(recipe)

        return recipes[:count]

    def is_suitable_for_skill(self, recipe_difficulty: str, user_skill: str) -> bool:
        """Check if recipe difficulty matches user skill level"""
        skill_mapping = {
            "beginner": ["easy"],
            "intermediate": ["easy", "medium"],
            "advanced": ["easy", "medium", "hard"],
        }

        return recipe_difficulty in skill_mapping.get(user_skill, ["easy"])
