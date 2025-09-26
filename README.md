# ChefPath Backend - Adaptive Cooking Mentor

An intelligent FastAPI backend that delivers personalized cooking education through adaptive meal planning and real-time difficulty adjustment based on user feedback.

## 🚀 Key Features

- **Adaptive Learning Engine**: Automatically adjusts recipe difficulty based on user feedback (too easy/just right/too hard)
- **Progressive Content Unlocking**: Week-by-week meal plan unlocking system with completion tracking
- **Intelligent Recipe Curation**: Fetches and filters recipes from TheMealDB API based on user preferences and skill level
- **Real-time Progress Analytics**: Comprehensive user progress tracking and skill assessment
- **RESTful API Design**: Clean, documented endpoints with automatic OpenAPI/Swagger documentation

## 🛠️ Technologies Used

- **Backend Framework**: FastAPI (Python) - Modern, fast web framework with automatic API documentation
- **Database**: SQLite with SQLAlchemy ORM - Lightweight database with object-relational mapping
- **API Integration**: httpx - Async HTTP client for external API calls
- **Data Validation**: Pydantic - Type validation and serialization
- **External API**: TheMealDB - Recipe data source with 1000+ recipes

## 🏗️ Architecture

- **4 Core Models**: User profiles, Recipes, Weekly plans, Progress tracking
- **5 API Endpoints**: User management, meal planning, recipe details, feedback collection, progress analytics
- **Adaptive Algorithm**: Skill-based recipe filtering with feedback-driven difficulty adjustment

## 📋 API Endpoints

### User Management

- `POST /api/user` - Create/update user profile
- `GET /api/user/{id}` - Retrieve user details

### Meal Planning

- `GET /api/weekly-plan` - Get adaptive weekly meal plan
- `GET /api/recipe/{id}` - Get recipe details

### Learning Analytics

- `POST /api/feedback` - Submit recipe difficulty feedback
- `GET /api/progress` - Get user progress analytics

## 🚀 Getting Started

```bash
./start.sh                           # Auto-setup and run
# OR
pip install -r requirements.txt     # Manual setup
python manage_db.py reset           # Setup database with sample data
uvicorn main:app --reload           # Start server
```

## 🗄️ Database Management

Powerful scripts for database operations:

```bash
python manage_db.py reset           # Full reset: clear + seed with sample data
python manage_db.py view --summary  # View database contents
python manage_db.py clear           # Clear all data
python manage_db.py seed             # Add sample data
```

See `scripts/README.md` for detailed documentation.
