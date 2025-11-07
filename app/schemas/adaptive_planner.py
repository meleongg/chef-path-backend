import uuid
from typing import List, Optional
from pydantic import BaseModel, Field

# --- Input Schema Definition ---
class HybridSearchInput(BaseModel):
    """Input parameters for retrieving recipes using hybrid search."""
    intent_query: str = Field(
        description="The detailed natural language query describing the desired recipe type, cuisine, or flavor profile."
    )
    user_id: uuid.UUID = Field(
        description="The UUID of the authenticated user requesting the recipe candidates."
    )
    exclude_ids: Optional[List[uuid.UUID]] = Field(
        default_factory=list,
        description="A list of Recipe UUIDs that the system should exclude (e.g., recipes rated 'too hard')."
    )
    similarity_threshold: Optional[float] = Field(
        default=0.7,
        description="The minimum semantic similarity score (0.0 to 1.0) required to consider a recipe relevant. Default is 0.7."
    )
    limit: Optional[int] = Field(
        default=10,
        description="The maximum number of recipe candidates to return."
    )