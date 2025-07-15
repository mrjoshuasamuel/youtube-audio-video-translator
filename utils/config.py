import os
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Config(BaseSettings):
    """Configuration management for the YouTube processing workflow"""
    
    # Use the new model_config instead of nested Config class
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )
    
    # API Keys
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    murf_api_key: str = Field(..., env="MURF_API_KEY")
    notion_api_key: str = Field(..., env="NOTION_API_KEY")
    notion_database_id: str = Field(..., env="NOTION_DATABASE_ID")
    
    # Model configurations
    openai_model: str = Field(default="gpt-4o", env="OPENAI_MODEL")
    whisper_model: str = Field(default="base", env="WHISPER_MODEL")
    
    # File paths
    temp_folder: Path = Field(default=Path("./temp"), env="TEMP_FOLDER")
    output_folder: Path = Field(default=Path("./output"), env="OUTPUT_FOLDER")
    
    # Processing settings
    max_file_size_mb: int = Field(default=500, env="MAX_FILE_SIZE_MB")
    audio_quality: str = Field(default="best", env="AUDIO_QUALITY")
    video_quality: str = Field(default="best", env="VIDEO_QUALITY")
    
    # Language settings
    default_language: str = Field(default="en", env="DEFAULT_LANGUAGE")
    supported_languages: List[str] = Field(
        default=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
        env="SUPPORTED_LANGUAGES"
    )