from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    APP_NAME: str = "GameVerse API"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # DEBUG dinâmico baseado no ambiente (obrigatório False em produção per Checklist)
    DEBUG: bool = os.getenv("ENVIRONMENT", "development") == "development"
    
    # CORS (Restrição Específica)
    ALLOWED_ORIGINS_STR: str = os.getenv(
        "ALLOWED_ORIGINS", 
        "http://localhost:3000,http://localhost:8000"
    )

    @property
    def ALLOWED_ORIGINS(self) -> list[str]:
        return [origin.strip() for origin in self.ALLOWED_ORIGINS_STR.split(",") if origin.strip()]

    # Supabase / PostgreSQL
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/steam_db",
    )

    # FIXED: adicionar modo de SSL configurável para conexão com o banco (segurança dev vs prod)
    # Valores possíveis:
    #   - require  -> SSL com verificação de certificado (default e recomendado em produção)
    #   - insecure -> SSL sem verificação (apenas para debug local pontual)
    #   - disable  -> sem SSL (ex.: Postgres local em desenvolvimento)
    DB_SSL_MODE: str = os.getenv("DB_SSL_MODE", "require")

    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "super-secret-key-321")

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> "Settings":
    return Settings()


settings = get_settings()
