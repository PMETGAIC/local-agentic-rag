import os
from pydantic_settings import BaseSettings

class Config(BaseSettings):
    PROFILE: str = os.getenv("VRAM_PROFILE", "4gb")
    
    PROFILES: dict = {
        "4gb": {
            "llm_path": "models/qwen2.5-3b-instruct-q4_k_m.gguf",
            "embed_path": "models/nomic-embed-text-v1.5.f16.gguf",
            "n_ctx": 2048,
            "n_batch": 256
        },
        "12gb": {
            "llm_path": "models/Qwen_Qwen3.5-9B-Q4_K_M.gguf",
            "embed_path": "models/nomic-embed-text-v1.5.f16.gguf",
            "n_ctx": 8192,
            "n_batch": 512
        }
    }
    
    @property
    def active(self) -> dict:
        return self.PROFILES.get(self.PROFILE, self.PROFILES["4gb"])

cfg = Config()