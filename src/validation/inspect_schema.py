import asyncio
import os

import asyncpg
from dotenv import load_dotenv

load_dotenv()


async def inspect():
    host = os.getenv("DB_HOST")
    port = int(os.getenv("DB_PORT", 6543))
    user = os.getenv("DB_USER")
    database = os.getenv("DB_NAME")
    password = os.getenv("DB_PASSWORD")

    # Desativar statement_cache para pgbouncer transaction mode
    conn = await asyncpg.connect(
        user=user,
        password=password,
        database=database,
        host=host,
        port=port,
        ssl="require",
        statement_cache_size=0,
    )

    print("Colunas da tabela 'jogos':")
    columns = await conn.fetch(
        "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'jogos'"
    )
    for col in columns:
        print(f"- {col['column_name']} ({col['data_type']})")

    print("\nOutras tabelas:")
    tables = await conn.fetch(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
    )
    for table in tables:
        print(f"- {table['table_name']}")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(inspect())
