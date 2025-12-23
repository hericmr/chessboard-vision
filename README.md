# ChessVision: Sistema de Digitalização de Tabuleiro

Este projeto é uma ferramenta de visão computacional projetada para digitalizar um tabuleiro de xadrez físico em tempo real usando uma webcam comum (ou smartphone via Iriun Webcam).

O foco atual é na **robustez, estabilidade e desempenho**, utilizando técnicas clássicas de visão computacional com OpenCV, deliberadamente evitando Deep Learning para a detecção do tabuleiro a fim de garantir zero jitter e baixo consumo de recursos.

## Funcionalidades Principais

*   **Seleção Manual de ROI (Região de Interesse)**:
    *   O operador desenha o retângulo do tabuleiro manualmente.
    *   **Benefício**: Garante coordenadas 100% estáveis (zero tremedeira/jitter), essenciais para evitar "alucinações" de movimento de peças.
    *   Ideal para setups com câmera fixa (tripé ou braço).
    
*   **Pipeline de Melhoria de Imagem (`frame_enhancer.py`)**:
    *   Processamento em tempo real para tratar imagens de webcam.
    *   **Correção de Luz**: CLAHE aplicado no canal de Luminosidade (LAB).
    *   **Redução de Ruído**: Filtro Bilateral para suavizar texturas preservando bordas das peças.
    *   **Sharpening**: Kernel de nitidez para destacar limites entre casas.
    *   **Análise**: Geração de máscaras binárias (Otsu) para facilitar segmentação.

*   **Suporte a Iriun Webcam**:
    *   Documentado e otimizado para uso com smartphones como câmeras de alta fidelidade via Wi-Fi/USB.

## Stack Tecnológico

*   **Linguagem**: Python 3.x
*   **Bibliotecas**: OpenCV (`opencv-python`), NumPy.
*   **Deep Learning?**: **NÃO utilizado** para detecção de bordas/tabuleiro (substituído por input manual determinístico).

## Pré-requisitos

1.  Python 3 instalado.
2.  Webcam conectada ou Smartphone + [Iriun Webcam](https://iriun.com/).
3.  Bibliotecas:
    ```bash
    pip install -r requirements.txt
    ```

## Como Usar

### 1. Testar Qualidade da Imagem
Execute o script de melhoria para ajustar iluminação e foco:
```bash
python frame_enhancer.py
```
*   Verifique as janelas "Original" vs "Enhanced".
*   Assegure que o tabuleiro esteja bem iluminado e focado.

### 2. Executar o Sistema Principal
(Em desenvolvimento - futura integração do ROI manual)
```bash
python main.py
```
*   **Passo 1**: Uma janela mostrará o feed da câmera.
*   **Passo 2**: Clique e arraste o mouse para desenhar um retângulo cobrindo EXATAMENTE o tabuleiro.
*   **Passo 3**: Solte para confirmar. O sistema travará nessas coordenadas.

## Decisões Técnicas

| Decisão | Motivo |
| :--- | :--- |
| **ROI Manual vs Automático** | Algoritmos automáticos recalculam vértices a cada frame, causando oscilação ("jitter") nas casas. O ROI manual é imutável e computacionalmente gratuito. |
| **OpenCV Puro vs YOLO/CNN** | Reduz latência e remove dependência de GPU dedicada para a etapa de segmentação do grid. |
| **Filtro Bilateral** | Escolhido ao invés de Gaussian Blur por preservar as "quinas" das casas e peças, cruciais para a lógica de detecção. |

## Contribuição

Este projeto é focado em simplicidade e eficácia para digitalização de partidas físicas. Sugestões de melhoria em algoritmos de segmentação de cor/peça são bem-vindas.
