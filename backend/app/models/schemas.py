from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from datetime import datetime

from uuid import UUID

class JogoBase(BaseModel):
    id: UUID
    titulo: str
    preco_base: float
    avaliacao_media: float
    total_avaliacoes: int
    idade_requerida: int

class JogoDetalhes(JogoBase):
    desenvolvedor_nome: Optional[str] = None
    desenvolvedor_pais: Optional[str] = None
    categorias: List[str] = []
    
    model_config = ConfigDict(from_attributes=True)

class SimilarJogo(BaseModel):
    id: UUID
    titulo: str
    avaliacao_media: float
    categorias_comuns: int

class AchievementTier(BaseModel):
    achievements: int
    tier: str

class UserLibraryItem(JogoBase):
    horas_jogadas: Optional[float] = None
    ultima_sessao: Optional[datetime] = None

class UserStats(BaseModel):
    total_jogos: int
    total_horas: float
    categoria_preferida: Optional[str] = None

class AnaliticoPreco(BaseModel):
    nome: str
    preco_medio: float
    desvio_padrao: float
    total_jogos: int

class AnaliticoTempo(BaseModel):
    genero: str
    mediana_horas: float
    media_horas: float
    total_jogadores: int

class TopDev(BaseModel):
    id: UUID
    nome: str
    nota_media: float
    consistencia: float
    total_jogos: int
