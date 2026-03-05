import asyncpg
import ssl
from app.config import settings
from app.utils.logging import setup_logging

logger = setup_logging()


class Database:
    def __init__(self) -> None:
        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        if self.pool:
            return

        try:
            ssl_param: ssl.SSLContext | None

            # FIXED: SSL condicional - seguro por padrão, com modos explícitos para dev/local
            if settings.DB_SSL_MODE == "disable":
                ssl_param = None
                logger.info("DB_SSL_MODE=disable - conexão sem SSL (uso apenas em desenvolvimento local).")
            elif settings.DB_SSL_MODE == "insecure":
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                ssl_param = ctx
                logger.warning(
                    "DB_SSL_MODE=insecure - SSL sem verificação de certificado habilitado "
                    "(permitido apenas para depuração local)."
                )
            else:
                # require (default): SSL com verificação padrão de certificado
                ssl_param = ssl.create_default_context()
                logger.info("DB_SSL_MODE=require - SSL com verificação de certificado habilitado.")

            self.pool = await asyncpg.create_pool(
                settings.DATABASE_URL,
                min_size=1,
                max_size=10,
                statement_cache_size=0,
                ssl=ssl_param,
            )
            logger.info("Conexão com o banco de dados estabelecida via pool (asyncpg).")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Erro ao conectar ao banco de dados: {e}")
            raise e

    async def disconnect(self) -> None:
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("Pool de conexões com o banco de dados fechado.")

    async def fetch(self, query: str, *args):
        if not self.pool:
            await self.connect()
        async with self.pool.acquire() as connection:
            return await connection.fetch(query, *args)

    async def fetchrow(self, query: str, *args):
        if not self.pool:
            await self.connect()
        async with self.pool.acquire() as connection:
            return await connection.fetchrow(query, *args)

    async def execute(self, query: str, *args):
        if not self.pool:
            await self.connect()
        async with self.pool.acquire() as connection:
            return await connection.execute(query, *args)


db = Database()


async def get_db() -> Database:
    return db
