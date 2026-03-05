import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded

from app.routes import jogos, usuarios, analiticos, recomendacoes
from app.models.database import db
from app.utils.logging import setup_logging
from app.config import settings
from app.utils.security import limiter, SecurityHeadersMiddleware, _rate_limit_exceeded_handler

# Setup Logging
logger = setup_logging()

app = FastAPI(
    title=settings.APP_NAME,
    description="API para exploração e recomendação de jogos da Steam",
    version=settings.APP_VERSION,
    debug=settings.DEBUG
)

# Registra Limitador SlowAPI
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Adiciona Security Headers customizados (HSTS, CSP, XFO, XCTO)
app.add_middleware(SecurityHeadersMiddleware)

# CORS Específico
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware para logging de requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    logger.info(f"{request.method} {request.url.path} - {response.status_code} ({process_time:.2f}ms)")
    return response

# Eventos de Ciclo de Vida
@app.on_event("startup")
async def startup():
    await db.connect()

@app.on_event("shutdown")
async def shutdown():
    await db.disconnect()

# Rotas
app.include_router(jogos.router, prefix="/api/v1/jogos", tags=["Jogos"])
app.include_router(usuarios.router, prefix="/api/v1/usuarios", tags=["Usuários"])
app.include_router(analiticos.router, prefix="/api/v1/analiticos", tags=["Analíticos"])
app.include_router(recomendacoes.router, prefix="/api/v1/recomendacoes", tags=["Recomendações"])

@app.get("/")
async def root():
    return {
        "app": "GameVerse API",
        "docs": "/docs",
        "status": "online"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": time.time()}
