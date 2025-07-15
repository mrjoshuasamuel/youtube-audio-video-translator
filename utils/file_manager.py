# utils/file_manager.py
import os
import shutil
import tempfile
import aiofiles
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class FileManager:
    """Manages file operations for the workflow"""
    
    def __init__(self, temp_folder: Path, output_folder: Path):
        self.temp_folder = Path(temp_folder)
        self.output_folder = Path(output_folder)
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories"""
        self.temp_folder.mkdir(parents=True, exist_ok=True)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.temp_folder / "audio").mkdir(exist_ok=True)
        (self.temp_folder / "video").mkdir(exist_ok=True)
        (self.temp_folder / "transcripts").mkdir(exist_ok=True)
        (self.output_folder / "final").mkdir(exist_ok=True)
    
    def get_temp_path(self, filename: str, subfolder: str = "") -> Path:
        """Get temporary file path"""
        if subfolder:
            path = self.temp_folder / subfolder / filename
        else:
            path = self.temp_folder / filename
        return path
    
    def get_output_path(self, filename: str) -> Path:
        """Get output file path"""
        return self.output_folder / "final" / filename
    
    async def save_file(self, content: bytes, filepath: Path) -> bool:
        """Save file asynchronously"""
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(filepath, 'wb') as f:
                await f.write(content)
            return True
        except Exception as e:
            logger.error(f"Failed to save file {filepath}: {e}")
            return False
    
    async def read_file(self, filepath: Path) -> Optional[bytes]:
        """Read file asynchronously"""
        try:
            async with aiofiles.open(filepath, 'rb') as f:
                return await f.read()
        except Exception as e:
            logger.error(f"Failed to read file {filepath}: {e}")
            return None
    
    def cleanup_temp_files(self, keep_final: bool = True):
        """Clean up temporary files"""
        try:
            if self.temp_folder.exists():
                shutil.rmtree(self.temp_folder)
                self._setup_directories()
            logger.info("Temporary files cleaned up")
        except Exception as e:
            logger.error(f"Failed to clean up temp files: {e}")
    
    def get_file_size_mb(self, filepath: Path) -> float:
        """Get file size in MB"""
        if filepath.exists():
            return filepath.stat().st_size / (1024 * 1024)
        return 0.0