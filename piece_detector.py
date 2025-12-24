"""
Piece Detector - Detecta peças circulares em casas do tabuleiro

Em vez de detectar MUDANÇAS, detecta diretamente a PRESENÇA de peça
analisando se há um padrão circular em cada casa.

Técnicas:
1. Detecção de bordas (Canny)
2. Hough Circles ou análise de contornos
3. Verificar simetria radial
"""

import cv2
import numpy as np


class PieceDetector:
    """
    Detecta peças redondas em casas do tabuleiro.
    
    Não depende de referência/calibração - analisa diretamente
    se há forma circular em cada casa.
    
    Inclui suavização temporal para estabilizar detecções.
    """
    
    def __init__(self):
        # Parâmetros de detecção
        self.min_radius_ratio = 0.20  # Raio mínimo = 20% do tamanho da casa
        self.max_radius_ratio = 0.55  # Raio máximo = 55% do tamanho da casa (bases maiores)
        self.edge_threshold = 50      # Threshold para detecção de borda
        self.circle_threshold = 0.6   # Mínimo de "circularidade" (0-1)
        
        # Suavização temporal
        self.history_size = 5         # Guardar últimos N frames
        self.min_presence = 0.6       # 60% presença para confirmar peça
        self.detection_history = {}   # {(file, rank): [bool, bool, ...]}
        
        # DELTA DETECTION - referência para detectar mudanças
        self.reference_squares = {}   # {(file, rank): grayscale_img}
        self.cached_results = {}      # {(file, rank): detection_result}
        self.change_threshold = 25    # Diferença média de pixels para considerar mudança
    
    def calibrate_reference(self, squares_dict):
        """Salva referência das imagens para detecção delta."""
        self.reference_squares.clear()
        self.cached_results.clear()
        
        for pos, img in squares_dict.items():
            gray = self._preprocess_square(img)
            self.reference_squares[pos] = gray.copy()
            # Detectar estado inicial
            result = self.detect_piece(img, pos)
            self.cached_results[pos] = result
    
    def _has_changed(self, pos, current_gray):
        """Verifica se a casa mudou em relação à referência."""
        if pos not in self.reference_squares:
            return True  # Sem referência = processar
        
        ref = self.reference_squares[pos]
        
        # Calcular diferença média de intensidade
        diff = cv2.absdiff(current_gray, ref)
        mean_diff = np.mean(diff)
        
        return mean_diff > self.change_threshold
    
    def _update_reference(self, pos, gray):
        """Atualiza referência de uma casa."""
        self.reference_squares[pos] = gray.copy()
    
    def _update_history(self, pos, has_piece):
        """Atualiza histórico de detecção para uma casa."""
        if pos not in self.detection_history:
            self.detection_history[pos] = []
        
        history = self.detection_history[pos]
        history.append(has_piece)
        
        # Manter apenas últimos N frames
        if len(history) > self.history_size:
            history.pop(0)
    
    def _get_stable_detection(self, pos):
        """Retorna detecção estável baseada no histórico."""
        if pos not in self.detection_history:
            return False
        
        history = self.detection_history[pos]
        if len(history) < 3:  # Precisa de pelo menos 3 frames
            return history[-1] if history else False
        
        # Calcular presença média
        presence = sum(history) / len(history)
        return presence >= self.min_presence
        
    def _preprocess_square(self, square_img):
        """Pré-processa imagem da casa para detecção."""
        # Converter para escala de cinza
        if len(square_img.shape) == 3:
            gray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = square_img.copy()
        
        # Blur para reduzir ruído
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return blurred
    

    

    
    def _analyze_radial_symmetry(self, gray):
        """
        Analisa simetria radial - peças redondas têm alta simetria.
        
        Returns:
            score: 0-1, quanto maior mais circular
        """
        h, w = gray.shape
        center_y, center_x = h // 2, w // 2
        
        # Criar anéis concêntricos e comparar intensidade média
        radii = [min(h, w) * r for r in [0.15, 0.25, 0.35, 0.45]]
        ring_means = []
        
        for r in radii:
            # Criar máscara de anel
            y_coords, x_coords = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            ring_mask = (dist_from_center >= r - 5) & (dist_from_center <= r + 5)
            
            if np.sum(ring_mask) > 0:
                ring_mean = np.mean(gray[ring_mask])
                ring_means.append(ring_mean)
        
        if len(ring_means) < 2:
            return 0.0
        
        # Calcular variância das médias dos anéis
        # Alta variância = borda de círculo (mudança abrupta)
        variance = np.var(ring_means)
        
        # Normalizar para 0-1
        score = min(1.0, variance / 500)  # 500 é um valor de normalização
        
        return score
    
    def _detect_center_vs_border(self, gray):
        """
        Compara intensidade do centro vs borda para detectar peça.
        
        Peça: Centro diferente da borda
        Vazio: Centro similar à borda
        """
        h, w = gray.shape
        center_y, center_x = h // 2, w // 2
        radius = min(h, w) // 4
        
        # Máscara do centro (círculo interno)
        y_coords, x_coords = np.ogrid[:h, :w]
        center_mask = ((x_coords - center_x)**2 + (y_coords - center_y)**2) <= radius**2
        
        # Máscara da borda (cantos)
        corner_size = min(h, w) // 4
        border_mask = np.zeros((h, w), dtype=bool)
        border_mask[:corner_size, :corner_size] = True
        border_mask[:corner_size, -corner_size:] = True
        border_mask[-corner_size:, :corner_size] = True
        border_mask[-corner_size:, -corner_size:] = True
        
        center_mean = np.mean(gray[center_mask])
        border_mean = np.mean(gray[border_mask])
        
        # Diferença entre centro e borda
        diff = abs(center_mean - border_mean)
        
        # Se diferença alta, provavelmente há peça
        return diff, center_mean, border_mean
    
    
    def _detect_circle_unified(self, gray):
        """
        Detecção unificada de círculos (Grandes e Pequenos).
        
        Substitui _detect_circle_hough (peça inteira) e _detect_small_circle (topo).
        Usa um range de raio mais amplo (12% a 55%) e seleciona o melhor candidato.
        
        Returns:
            (found, center, radius, type)
            type: 'hough' (grande) ou 'tower_top' (pequeno)
        """
        h, w = gray.shape
        min_dim = min(h, w)
        
        # Range unificado: 12% (tower top) a 55% (peça cheia)
        min_radius = int(min_dim * 0.12)
        max_radius = int(min_dim * 0.55)
        
        # Param2=25 (mais sensível para pegar ambos)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dim // 3,
            param1=100,
            param2=25,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        if circles is not None and len(circles[0]) > 0:
            center_x, center_y = w // 2, h // 2
            
            best_circle = None
            best_dist = float('inf')
            
            # Limite de centralização (30% do tamanho)
            max_offset = min_dim * 0.3
            
            for circle in circles[0]:
                cx, cy, r = circle
                dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                
                if dist < max_offset and dist < best_dist:
                    best_dist = dist
                    best_circle = circle
            
            if best_circle is not None:
                r = int(best_circle[2])
                
                # Classificar pelo tamanho
                # < 20% -> Small (Tower Top)
                if r < min_dim * 0.20:
                     return True, (int(best_circle[0]), int(best_circle[1])), r, 'tower_top'
                else:
                     return True, (int(best_circle[0]), int(best_circle[1])), r, 'hough'
        
        return False, None, None, None

    def detect_piece(self, square_img, pos=None):
        """
        Detecta se há peça circular/oval na casa.
        
        Args:
            square_img: Imagem da casa (BGR ou grayscale)
            pos: (file, rank) para ajustar tolerância nas bordas
            
        Returns:
            dict com:
                - has_piece: bool
                - confidence: 0-1
                - center: (x, y) se encontrado
                - radius: int se encontrado
                - method: qual método detectou
                - is_ellipse: True se detectou forma oval
        """
        gray = self._preprocess_square(square_img)
        h, w = gray.shape
        
        result = {
            'has_piece': False,
            'confidence': 0.0,
            'center': None,
            'radius': None,
            'method': None,
            'center_border_diff': 0,
            'is_ellipse': False,
            'axes': None,
        }
        
        # Pré-filtro: se a imagem for muito uniforme, não tem peça
        # (filtra casa vazia lisa ou mão cobrindo tudo)
        std_dev = np.std(gray)
        if std_dev < 15:  # Threshold de uniformidade
            return result
        
        # Método 1: Hough Circles Unificado (Grande e Pequeno)
        found, center, radius, type_detected = self._detect_circle_unified(gray)
        
        if found:
            result['has_piece'] = True
            result['center'] = center
            result['radius'] = radius
            result['method'] = type_detected
            result['confidence'] = 0.9 if type_detected == 'hough' else 0.75
            return result

        
        # Método 4: Centro vs Borda
        diff, center_mean, border_mean = self._detect_center_vs_border(gray)
        result['center_border_diff'] = diff
        
        # Se diferença significativa entre centro e borda
        if diff > 40:  # Threshold aumentado (era 25) para evitar falso positivo com sombra
            result['has_piece'] = True
            result['center'] = (w // 2, h // 2)
            result['radius'] = min(h, w) // 3
            result['method'] = 'center_diff'
            result['confidence'] = min(1.0, diff / 80)
            return result
        
        # Método 5: Simetria radial
        symmetry = self._analyze_radial_symmetry(gray)
        
        if symmetry > self.circle_threshold:
            result['has_piece'] = True
            result['center'] = (w // 2, h // 2)
            result['radius'] = min(h, w) // 3
            result['method'] = 'symmetry'
            result['confidence'] = symmetry
            return result
        
        return result
    
    
    def detect_all_pieces(self, squares_dict, use_smoothing=True, use_delta=True, squares_to_check=None):
        """
        Detecta peças em todas as casas.
        
        Args:
            squares_dict: {(file, rank): square_img}
            use_smoothing: usar suavização temporal
            use_delta: só processar casas que mudaram (performance)
            squares_to_check: set of (file, rank) - se fornecido, força verificação nestas casas
            
        Returns:
            (results, visual_changes)
            results: dict {(file, rank): detection_result}
            visual_changes: set {(file, rank)} das casas que mudaram visualmente
        """
        results = {}
        visual_changes = set()
        processed_count = 0
        
        for pos, img in squares_dict.items():
            gray = self._preprocess_square(img)
            
            # Decidir se deve processar esta casa
            should_process = False
            has_changed_visual = False
            
            # Verificar mudança visual (Delta)
            # Calculamos sempre para alimentar o NoiseHandler, mesmo se squares_to_check forçar processamento
            if self._has_changed(pos, gray):
                has_changed_visual = True
                visual_changes.add(pos)
            
            # 1. Prioridade: Se está na lista de verificação obrigatória (regras)
            if squares_to_check is not None:
                if pos in squares_to_check:
                    should_process = True
            
            # 2. Se não foi forçado, verificar delta (visual)
            if not should_process:
                if squares_to_check is None or use_delta: # Se delta ativo, verifica mudança
                    if pos not in self.cached_results or has_changed_visual:
                        should_process = True
            
            # Se a imagem mudou, processar novamente
            if should_process:
                processed_count += 1
                raw_result = self.detect_piece(img, pos)
                
                # Armazenar resultado BRUTO no cache (antes da estabilização)
                self.cached_results[pos] = raw_result.copy()
                # NOTA: _update_reference movido para o final (só se estável)
                
                raw_has_piece = raw_result['has_piece']
            else:
                # Usar resultado bruto do cache
                if pos in self.cached_results:
                     raw_result = self.cached_results[pos].copy()
                else:
                     # Fallback seguro se não tiver cache
                     raw_result = self.detect_piece(img, pos)
                     self.cached_results[pos] = raw_result.copy()
                
                raw_has_piece = raw_result['has_piece']
            
            # SEMPRE atualizar histórico (mesmo se veio do cache)
            self._update_history(pos, raw_has_piece)
            
            # Verificar estabilidade
            is_stable_update = True
            if use_smoothing:
                # Recalcular estabilidade com histórico atualizado
                stable_detection = self._get_stable_detection(pos)
                raw_result['has_piece'] = stable_detection
                
                # Adicionar info de debug
                if 'confidence' in raw_result:
                    raw_result['confidence'] = raw_result['confidence']
                
                # Se o que vemos agora (raw) difere do estável, a situação é instável (transição)
                # Não devemos atualizar a referência visual nesses casos
                if raw_has_piece != stable_detection:
                    is_stable_update = False
            
            # Atualizar referência visual APENAS se:
            # 1. Processamos uma imagem nova (should_process)
            # 2. A detecção é estável (raw == stable)
            # Isso evita salvar referência de uma "mão" passando
            if should_process and is_stable_update:
                self._update_reference(pos, gray)
            
            results[pos] = raw_result
        
        return results, visual_changes
    
    def get_occupied_squares(self, squares_dict, use_smoothing=True):
        # Retorna conjunto de casas ocupadas.
        results, _ = self.detect_all_pieces(squares_dict, use_smoothing)
        return {pos for pos, info in results.items() if info['has_piece']}
    
    def update_references(self, squares_dict):
        """Força atualização de todas as referências visuais e limpa cache."""
        for pos, img in squares_dict.items():
            gray = self._preprocess_square(img)
            self._update_reference(pos, gray)
        # Limpar cache para forçar reprocessamento delta na próxima vez
        self.cached_results.clear()
