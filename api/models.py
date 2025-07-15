from pydantic import BaseModel, Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class LanguageCode(str, Enum):
    EN = "en"
    ES = "es"
    FR = "fr"
    DE = "de"
    IT = "it"
    PT = "pt"
    RU = "ru"
    JA = "ja"
    KO = "ko"
    ZH = "zh"

class ProcessingRequest(BaseModel):
    """Request model for video processing"""
    youtube_url: HttpUrl = Field(..., description="YouTube video URL")
    operations: List[str] = Field(..., description="List of operations to perform")
    target_languages: List[LanguageCode] = Field(default=[], description="Target languages for translation")
    priority: int = Field(default=1, ge=1, le=5, description="Processing priority (1-5)")
    webhook_url: Optional[HttpUrl] = Field(None, description="Webhook URL for notifications")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class ProcessingResponse(BaseModel):
    """Response model for processing requests"""
    job_id: str = Field(..., description="Unique job identifier")
    status: ProcessingStatus = Field(..., description="Current processing status")
    created_at: datetime = Field(..., description="Job creation timestamp")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    progress: float = Field(default=0.0, ge=0.0, le=100.0, description="Progress percentage")

class JobStatus(BaseModel):
    """Job status model"""
    job_id: str
    status: ProcessingStatus
    progress: float
    current_stage: str
    stages_completed: List[str]
    stages_remaining: List[str]
    created_at: datetime
    updated_at: datetime
    completion_time: Optional[datetime]
    error_message: Optional[str]
    results: Dict[str, Any]

class BatchProcessingRequest(BaseModel):
    """Batch processing request model"""
    requests: List[ProcessingRequest] = Field(..., max_length=10)
    batch_name: Optional[str] = Field(None, description="Name for the batch")
    process_sequentially: bool = Field(False, description="Process items one by one")
