from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import create_tables
from app.routers import users, recipes, weekly_plans, feedback, auth, plan_agent
from app.agents.checkpoint_setup import initialize_postgres_saver


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create database tables
    create_tables()

    # Initialize PostgresSaver for checkpointing
    checkpointer = initialize_postgres_saver()

    # Set on app.state
    app.state.checkpoint_saver = checkpointer
    print("✅ Checkpoint saver is active and ready.")

    # Yield control to the application
    yield

    # Shutdown: Close the connection properly
    print("🔄 Shutting down checkpoint saver...")
    if checkpointer and hasattr(checkpointer, "conn"):
        checkpointer.conn.close()
        print("✅ Checkpoint connection closed.")


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
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(plan_agent.router, prefix="/plan", tags=["plan generation"])


@app.get("/")
async def root():
    return {"message": "ChefPath Backend API is running!"}
