# ChefPath Backend - Adaptive Cooking Mentor

An intelligent FastAPI backend that delivers personalized cooking education through adaptive meal planning and real-time difficulty adjustment based on user feedback.

## Key Features

- **Adaptive Learning Engine**: Automatically adjusts recipe difficulty based on user feedback (too easy/just right/too hard)
- **Progressive Content Unlocking**: Week-by-week meal plan unlocking system with completion tracking
- **Intelligent Recipe Curation**: Fetches and filters recipes from TheMealDB API based on user preferences and skill level
- **Real-time Progress Analytics**: Comprehensive user progress tracking and skill assessment

## Technologies

- **Backend Framework**: FastAPI (Python)
- **Database**: SQLite with SQLAlchemy ORM
- **Data Validation**: Pydantic
- **External API**: TheMealDB - Recipe data source with 300+ recipes

## Getting Started

```bash
./start.sh                           # Auto-setup and run
# OR
pip install -r requirements.txt     # Manual setup
python manage_db.py reset           # Setup database with sample data
uvicorn main:app --reload           # Start server
```
