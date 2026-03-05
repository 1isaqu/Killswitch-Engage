# queries.py - Consultas centralizadas

JOGOS_BY_CATEGORIA = """
    SELECT j.* FROM jogos j
    JOIN jogos_categorias jc ON j.id = jc.jogo_id
    JOIN categorias c ON jc.categoria_id = c.id
    WHERE c.nome ILIKE $1
    ORDER BY j.avaliacao_media DESC
    LIMIT $2 OFFSET $3
"""

JOGO_DETALHES = """
    SELECT 
        j.*,
        d.nome as desenvolvedor_nome,
        d.pais as desenvolvedor_pais,
        (
            SELECT array_agg(c.nome)
            FROM jogos_categorias jc
            JOIN categorias c ON jc.categoria_id = c.id
            WHERE jc.jogo_id = j.id
        ) as categorias
    FROM jogos j
    LEFT JOIN desenvolvedores d ON j.desenvolvedor_id = d.id
    WHERE j.id = $1
"""

JOGOS_SIMILARES = """
    WITH categorias_jogo AS (
        SELECT categoria_id
        FROM jogos_categorias
        WHERE jogo_id = $1
    )
    SELECT 
        j.id,
        j.titulo,
        j.avaliacao_media,
        COUNT(DISTINCT jc.categoria_id) as categorias_comuns
    FROM jogos j
    JOIN jogos_categorias jc ON j.id = jc.jogo_id
    WHERE jc.categoria_id IN (SELECT categoria_id FROM categorias_jogo)
        AND j.id != $1
    GROUP BY j.id, j.titulo, j.avaliacao_media
    ORDER BY categorias_comuns DESC, j.avaliacao_media DESC
    LIMIT $2
"""

ACHIEVEMENT_TIER = """
    SELECT 
        COUNT(id) as achievements,
        CASE 
            WHEN COUNT(id) <= 1 THEN 'baixo'
            WHEN COUNT(id) <= 3 THEN 'médio'
            WHEN COUNT(id) <= 5 THEN 'alto'
            ELSE 'extremo'
        END as tier
    FROM conquistas
    WHERE jogo_id = $1
"""

PRECO_MEDIO_POR_CATEGORIA = """
    SELECT 
        c.nome,
        AVG(j.preco_base) as preco_medio,
        STDDEV(j.preco_base) as desvio_padrao,
        COUNT(j.id) as total_jogos
    FROM jogos j
    JOIN jogos_categorias jc ON j.id = jc.jogo_id
    JOIN categorias c ON jc.categoria_id = c.id
    GROUP BY c.nome
    ORDER BY preco_medio DESC
"""

TEMPO_MEDIO_POR_GENERO = """
    SELECT 
        c.nome as genero,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY s.horas_jogadas) as mediana_horas,
        AVG(s.horas_jogadas) as media_horas,
        COUNT(DISTINCT s.usuario_id) as total_jogadores
    FROM sessoes_jogo s
    JOIN jogos j ON s.jogo_id = j.id
    JOIN jogos_categorias jc ON j.id = jc.jogo_id
    JOIN categorias c ON jc.categoria_id = c.id
    WHERE c.nome IN ('Massively Multiplayer', 'RPG', 'Strategy', 'Simulation', 'Action', 'Adventure')
    GROUP BY c.nome
    ORDER BY mediana_horas DESC
"""

TOP_DESENVOLVEDORES = """
    SELECT 
        d.id,
        d.nome,
        AVG(j.avaliacao_media) as nota_media,
        STDDEV(j.avaliacao_media) as consistencia,
        COUNT(j.id) as total_jogos
    FROM desenvolvedores d
    JOIN jogos j ON d.id = j.desenvolvedor_id
    GROUP BY d.id, d.nome
    HAVING COUNT(j.id) >= 5
    ORDER BY nota_media DESC
    LIMIT 10
"""

USER_LIBRARY = """
    SELECT 
        j.*, 
        COALESCE(SUM(s.horas_jogadas), 0) as horas_jogadas, 
        MAX(s.fim) as ultima_sessao
    FROM biblioteca b
    JOIN jogos j ON b.jogo_id = j.id
    LEFT JOIN sessoes_jogo s ON (s.jogo_id = j.id AND s.usuario_id = b.usuario_id)
    WHERE b.usuario_id = $1
    GROUP BY j.id, j.titulo, j.descricao, j.data_lancamento, j.preco_base, 
             j.desenvolvedor_id, j.avaliacao_media, j.total_avaliacoes, 
             j.metacritic_score, j.idade_requerida, j.created_at, j.updated_at
"""

USER_STATS = """
    SELECT 
        (SELECT COUNT(*) FROM biblioteca WHERE usuario_id = $1) as total_jogos,
        COALESCE(SUM(horas_jogadas), 0) as total_horas,
        (
            SELECT c.nome 
            FROM jogos_categorias jc 
            JOIN categorias c ON jc.categoria_id = c.id 
            WHERE jc.jogo_id IN (SELECT jogo_id FROM biblioteca WHERE usuario_id = $1)
            GROUP BY c.id 
            ORDER BY COUNT(*) DESC 
            LIMIT 1
        ) as categoria_preferida
    FROM sessoes_jogo
    WHERE usuario_id = $1
"""
