from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Gemini, for AI-gen fixes after a review is posted
    gemini_api_key: str = Field(..., validation_alias="GEMINI_API_KEY")
    gemini_model: str = Field("gemini-flash-latest", validation_alias="GEMINI_MODEL")
    # var : type = Field ("model", validation_alias = key)

    # Voyage AI
    voyage_api_key: str = Field(..., validation_alias="VOYAGE_API_KEY")
    voyage_model: str = Field("voyage-code-2", validation_alias="VOYAGE_MODEL")

    # Pinecone
    pinecone_api_key: str = Field(..., validation_alias="PINECONE_API_KEY")
    pinecone_index: str = Field("prism-code-index", validation_alias="PINECONE_INDEX")
    pinecone_top_k: int = Field(10, validation_alias="PINECONE_TOP_K")

    # GitHub
    github_token: str = Field(..., validation_alias="GITHUB_TOKEN")

    # GCP
    gcp_project_id: str = Field(..., validation_alias="GCP_PROJECT_ID")

    # Modal
    modal_app_name: str = Field("prism-sandbox", validation_alias="MODAL_APP_NAME")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "populate_by_name": True,
        "extra" : "ignore"
    }


settings = Settings()
