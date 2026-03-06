# Decisões Técnicas — Killswitch Engage

## Layer 4: Meta-Observer (cGAN)

### Motivação
Modelos de ranking tradicionais (Layer 3) costumam retornar scores com escalas variadas para usuários diferentes. Um threshold fixo (ex: 0.5) pode ser muito restritivo para um usuário "Casual" e muito permissivo para um "Hardcore".

### Implementação (Produção)
- **Arquitetura**: Conditional GAN (cGAN) onde o Gerador recebe o perfil do usuário (147 features) + ruído latente (32 dim) para prever o threshold ideal.
- **Integração**: O cGAN foi definido como o modo padrão (`meta`) do sistema.
- **Inferência**: Realizada em tempo real via PyTorch (CPU) no momento da recomendação.
- **Validação**: O modelo foi validado em 1.000 perfis, demonstrando ajuste automático para perfis mais ativos (thresholds seletivos) vs esporádicos.

## Fluxo de Recomendação
1. **RF Filter**: Baseado em tags e preço.
2. **KMeans Archetype**: Agrupamento por comportamento.
3. **LightFM Ranker**: Matriz de fatoração híbrida.
4. **cGAN Threshold**: Calibração meta-heurística do corte de score.

## Segurança
- Todos os segredos em `.env`.
- Modelos carregados via caminhos controlados em `src/config/paths.py`.
- Sanitização de logs para evitar PII.
