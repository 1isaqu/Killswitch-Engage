import asyncio
import asyncpg
import logging
import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FIXED: remover credenciais hard-coded e usar variáveis de ambiente para teste de conexão
load_dotenv()


async def test_conn() -> None:
    """
    Testa a autenticação com o Supabase/Postgres usando variáveis de ambiente.

    Requer as variáveis:
      - TEST_DB_HOST
      - TEST_DB_PORT (opcional, default 6543)
      - TEST_DB_USER
      - TEST_DB_PASSWORD
      - TEST_DB_NAME (opcional, default 'postgres')
    """
    host = os.getenv("TEST_DB_HOST")
    port = int(os.getenv("TEST_DB_PORT", "6543"))
    user = os.getenv("TEST_DB_USER")
    password = os.getenv("TEST_DB_PASSWORD")
    database = os.getenv("TEST_DB_NAME", "postgres")

    if not all([host, user, password]):
        logger.error(
            "Variáveis de ambiente de teste não configuradas. "
            "Defina TEST_DB_HOST, TEST_DB_USER e TEST_DB_PASSWORD no .env."
        )
        return

    try:
        conn = await asyncpg.connect(
            user=user,
            password=password,
            host=host,
            port=port,
            database=database,
            ssl="require",
        )
        logger.info(f"Conexão de teste bem-sucedida com {user}@{host}:{port}/{database}")
        await conn.close()
    except Exception as e:  # noqa: BLE001
        logger.error(f"Falha ao conectar para teste de autenticação: {e}")


if __name__ == "__main__":
    asyncio.run(test_conn())
