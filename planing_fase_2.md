# planning.md — Sincronização Bidirecional Tabuleiro Físico ↔ Lichess

## 🎯 Objetivo
Criar um sistema confiável que:
- Leia movimentos do tabuleiro físico via visão computacional
- Valide movimentos com python-chess
- Envie movimentos válidos para o Lichess via API oficial
- Receba movimentos do Lichess em tempo real
- Mantenha um **único estado de verdade (FEN)** sem estados fantasmas
- Seja robusto a ruído visual (mão do jogador, oclusões, iluminação)

---

## 🧠 Arquitetura Geral

### Fonte de Verdade
- `python-chess.Board` → **estado canônico**
- `last_committed_fen` → único estado aceito
- Nenhuma decisão é tomada apenas pela visão

### Componentes
- Vision Engine (OpenCV)
- Board Stabilizer / Noise Gate
- Move Resolver (python-chess)
- Lichess Sync Engine (API)
- UI / Overlay (debug e feedback visual)

---

## 🧩 FASE 1 — Detecção Visual Robusta (sem IA) ✅

> **Status: IMPLEMENTADA**

### ✅ 1.1 Calibração do Tabuleiro
- [x] Interface para o usuário definir:
  - 4 vértices do tabuleiro
  - lado das brancas / pretas
  - orientação (jogador joga de brancas ou pretas)
- [x] Renderizar notação correta (a1–h8) no overlay

### ✅ 1.2 Classificação de Casas
- [x] Diferenciar:
  - casa vazia
  - casa ocupada
- [x] Usar:
  - background subtraction por casa
  - energia de gradiente (Sobel)
  - variação de cor relativa à casa

### ✅ 1.3 Detector de Mudanças
- [x] `ChangeDetector` com sensibilidade configurável
- [x] Script de calibração interativo
- [x] Persistência de configurações

---

## 🧩 FASE 2 — Estado e Regras (python-chess)

> **Status: EM PROGRESSO**

### ✅ 2.1 Inicialização do Estado
- [x] Inicializar `python-chess.Board()` no início
- [x] Gerar FEN inicial confirmado
- [ ] Confirmar posição inicial com visão

### ✅ 2.2 Turnos
- [x] Travar lógica:
  - Brancas jogam primeiro
  - Depois pretas
- [x] Ignorar detecções fora do turno correto

### ✅ 2.3 Validação de Jogadas
- [x] Gerar `legal_moves` a partir do board atual
- [x] Usar essas jogadas como **filtro semântico**
- [x] Nunca aceitar jogada fora de `legal_moves`

### 🔄 2.4 Jogadas Especiais
- [ ] Roque (kingside/queenside)
- [ ] En passant
- [ ] Promoção de peão

---

## 🧩 FASE 3 — Noise Handling (Mão do Jogador) ✅

> **Status: IMPLEMENTADA**

### ✅ 3.1 Detecção de NOISE
- [x] Detectar ruído quando >3 casas mudam
- [x] Entrar em estado `NOISE_ACTIVE`

### ✅ 3.2 Lock de Identidade
- [x] Bloquear processamento de jogadas durante NOISE
- [x] Nenhuma jogada é validada durante ruído

### ✅ 3.3 Highlight Visual
- [x] Overlay vermelho durante NOISE
- [x] Indicador de progresso de estabilização

### ✅ 3.4 Saída do NOISE
- [x] Aguardar N frames estáveis (COOLDOWN_FRAMES=5)
- [x] Transição para IDLE ou MOVE_PENDING

---

## 🧩 FASE 4 — Resolução de Movimento ✅

> **Status: IMPLEMENTADA**

### ✅ 4.1 Preservação de Identidade
- [x] Identidade gerenciada via `python-chess`
- [x] Nunca gera peça nova
- [x] Capturas validadas pelas regras

### ✅ 4.2 Algoritmo de Resolução
- [x] `GameState.process_occupancy_change()` resolve movimentos
- [x] Suporta todos os padrões visuais:
  - 1v/1a = movimento normal
  - 1v/0a = captura
  - 2v/2a = roque
  - 2v/1a = en passant

### ✅ 4.3 Commit de Estado
- [x] Atualiza `python-chess.Board` após confirmação
- [x] Atualiza referência visual após movimento

---

## 🧩 FASE 5 — Integração com Lichess API ✅

> **Status: IMPLEMENTADA**

### ✅ 5.1 Autenticação
- [x] Token OAuth configurado (.env)
- [x] Escopos: `board:play`
- [x] Cliente HTTP direto (compatível Python 3.13)

### ✅ 5.2 Leitura de jogo (stream)
- [x] `stream_game()` com NDJSON
- [x] Detecta cor do jogador automaticamente
- [x] Retorna eventos em tempo real

### ✅ 5.3 Envio de jogadas
- [x] `make_move(uci)` envia para API
- [x] Tratamento de erros
- [x] Resign e seek game

---

## 🧩 FASE 6 — Assincronismo ✅

> **Status: IMPLEMENTADA**

### ✅ 6.1 Threads
- [x] Thread principal: visão + UI
- [x] Thread secundária: Lichess stream
- [x] Lock de turno sincronizado

### ✅ 6.2 Conflitos
- [x] Bloqueia input físico quando aguarda oponente
- [x] Sync automático de moves do Lichess
- [x] Rollback se envio falhar

---

## 🧪 FASE 7 — Testes e Debug ✅

> **Status: IMPLEMENTADA**

### ✅ 7.1 Logs
- [x] `logger.py` centralizado
- [x] Log de moves, noise, API

### ✅ 7.2 Testes
- [x] 26 testes unitários
- [x] GameState, NoiseHandler, LichessClient

---

## ✅ Critério de Sucesso

- [ ] Nenhuma peça "fantasma"
- [ ] Nenhuma troca de identidade
- [ ] Nenhuma jogada ilegal enviada
- [ ] Sincronização perfeita físico ↔ Lichess
- [ ] Sistema robusto à mão cobrindo o tabuleiro

---

## 📚 Referências Técnicas
- [Lichess Board API](https://lichess.org/api#tag/Board)
- [berserk (Python wrapper)](https://berserk.readthedocs.io/)
- python-chess
- Background Subtraction por região
- Sobel / Gradiente estrutural
- State machines para visão computacional
