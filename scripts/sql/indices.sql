-- FIXED: índices recomendados com base nas queries do backend e REVISAO_TECNICA.md

-- Índice para acelerar consultas por usuário/jogo em sessoes_jogo
CREATE INDEX IF NOT EXISTS idx_sessoes_usuario_jogo
    ON sessoes_jogo(usuario_id, jogo_id);

-- Índice incluindo fim para filtros temporais em tendências
CREATE INDEX IF NOT EXISTS idx_sessoes_fim
    ON sessoes_jogo(fim);

-- Índices para relações jogo <-> categoria
CREATE INDEX IF NOT EXISTS idx_jogos_categorias_jogo
    ON jogos_categorias(jogo_id, categoria_id);

CREATE INDEX IF NOT EXISTS idx_jogos_categorias_categoria
    ON jogos_categorias(categoria_id, jogo_id);

-- Índice para buscas por nome de categoria (ILIKE)
CREATE INDEX IF NOT EXISTS idx_categorias_nome
    ON categorias(LOWER(nome));

