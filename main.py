from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import create_tables
from app.routers import users, recipes, weekly_plans, feedback


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create database tables
    create_tables()
    yield
    # Shutdown: Clean up resources (if needed)


app = FastAPI(title="ChefPath Backend", version="1.0.0", lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(users.router, prefix="/api", tags=["users"])
app.include_router(recipes.router, prefix="/api", tags=["recipes"])
app.include_router(weekly_plans.router, prefix="/api", tags=["weekly-plans"])
app.include_router(feedback.router, prefix="/api", tags=["feedback"])


@app.get("/")
async def root():
    return {"message": "ChefPath Backend API is running!"}
