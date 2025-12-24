"""
Change Detector with Single Gaussian Background Model

Detects CHANGES from a reference state using adaptive Gaussian model.
Each square maintains its own mean (μ) and variance (σ²).
Detection uses Z-score instead of fixed threshold.

Based on: Running Gaussian Average method
Reference: Background-subtraction/single_gaussian.py
"""

import cv2
import numpy as np
import json
import os

SETTINGS_FILE = "sensitivity_settings.json"
DEFAULT_SETTINGS = {
    "sensitivity": 25,      # Fallback for simple mode
    "blur_kernel": 5,
    "stable_frames": 15,
    "z_threshold": 2.5,     # Z-score threshold (standard deviations)
    "alpha": 0.15,          # Learning rate for Gaussian update
    "initial_variance": 400, # Initial variance for new squares
    "use_gaussian": True,   # Use Gaussian model vs simple diff
}


class ChangeDetector:
    """
    Detects CHANGES in squares using Single Gaussian Background Model.
    
    Each square has:
    - mean (μ): expected pixel intensity
    - variance (σ²): how much variation is normal
    
    A square is "changed" if current intensity is > z_threshold standard deviations from mean.
    """
    
    def __init__(self):
        """Load settings from file or use defaults."""
        self.settings = self._load_settings()
        
        # Basic settings
        self.sensitivity = self.settings["sensitivity"]
        self.blur_kernel = self.settings["blur_kernel"]
        self.stable_frames = self.settings["stable_frames"]
        self._kernel = max(1, self.blur_kernel | 1)
        
        # Gaussian model settings
        self.z_threshold = self.settings.get("z_threshold", 2.5)
        self.alpha = self.settings.get("alpha", 0.15)
        self.initial_variance = self.settings.get("initial_variance", 400)
        self.use_gaussian = self.settings.get("use_gaussian", True)
        
        # Per-square Gaussian parameters
        # Each stores: {'mean': np.array, 'variance': np.array}
        self.gaussian_models = {}
        
        # Fallback for compatibility
        self.reference_squares = {}
        self.is_calibrated = False
        
        # Focus squares (radar) - when set, only these squares are monitored
        # This optimizes processing when we know legal move destinations
        self.focus_squares = None  # None = monitor all, set() = monitor specific
        
        mode = "Gaussian" if self.use_gaussian else "Simple"
        print(f"[ChangeDetector] Mode: {mode}")
        print(f"  Z-threshold: {self.z_threshold}, Alpha: {self.alpha}")
        print(f"  Variance: {self.initial_variance}, Blur: {self.blur_kernel}")
    
    def _load_settings(self):
        """Load all settings from file."""
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, 'r') as f:
                    data = json.load(f)
                    # Merge with defaults
                    result = DEFAULT_SETTINGS.copy()
                    result.update(data)
                    return result
            except:
                pass
        return DEFAULT_SETTINGS.copy()
    
    def _preprocess(self, img):
        """Convert to grayscale and apply blur."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (self._kernel, self._kernel), 0).astype(np.float64)
    
    def calibrate(self, squares_dict):
        """
        Initialize Gaussian model for each square.
        
        Sets mean = current intensity, variance = initial_variance
        """
        self.gaussian_models = {}
        self.reference_squares = {}
        
        for pos, img in squares_dict.items():
            processed = self._preprocess(img)
            
            # Initialize Gaussian model
            self.gaussian_models[pos] = {
                'mean': processed.copy(),
                'variance': np.full_like(processed, self.initial_variance, dtype=np.float64)
            }
            
            # Also store reference for compatibility
            self.reference_squares[pos] = processed.astype(np.uint8)
        
        self.is_calibrated = True
        print(f"[ChangeDetector] Gaussian model calibrated for {len(self.gaussian_models)} squares")
    
    def _calculate_zscore(self, current_processed, model):
        """
        Calculate Z-score for each pixel.
        
        Z = |current - mean| / sqrt(variance)
        
        Returns mean Z-score across all pixels.
        """
        mean = model['mean']
        variance = model['variance']
        
        # Ensure shapes match
        if current_processed.shape != mean.shape:
            current_processed = cv2.resize(
                current_processed, 
                (mean.shape[1], mean.shape[0])
            ).astype(np.float64)
        
        # Prevent division by zero
        sigma = np.sqrt(np.maximum(variance, 1.0))
        
        # Z-score per pixel
        diff = np.abs(current_processed - mean)
        z_scores = diff / sigma
        
        # Return mean Z-score (could also use max, or % above threshold)
        return np.mean(z_scores)
    def _calculate_change_intensity(self, current_processed, model):
        """
        Calcula análise fragmentada de mudança por casa.
        
        Returns:
            dict com:
                - z_score: média do Z-score
                - pct_changed: percentual de pixels significativamente alterados
                - intensity: 'TOTAL' (>85%), 'PARCIAL' (50-85%), 'LEVE' (15-50%)
                - is_circular: True se padrão de mudança é circular (peça)
                - center_ratio: proporção de mudança no centro vs borda
        """
        mean = model['mean']
        variance = model['variance']
        
        # Ensure shapes match
        if current_processed.shape != mean.shape:
            current_processed = cv2.resize(
                current_processed, 
                (mean.shape[1], mean.shape[0])
            ).astype(np.float64)
        
        # Prevent division by zero
        sigma = np.sqrt(np.maximum(variance, 1.0))
        
        # Z-score per pixel
        diff = np.abs(current_processed - mean)
        z_scores = diff / sigma
        
        # Máscara de pixels alterados
        changed_mask = z_scores > self.z_threshold
        
        # Calcular percentual de pixels acima do threshold
        total_pixels = z_scores.size
        pixels_changed = np.sum(changed_mask)
        pct_changed = (pixels_changed / total_pixels) * 100
        
        # === ANÁLISE CENTRO VS BORDA ===
        h, w = z_scores.shape
        center_y, center_x = h // 2, w // 2
        radius = min(h, w) // 4  # Raio do círculo central (50% do tamanho)
        
        # Criar máscara circular para o centro
        y_coords, x_coords = np.ogrid[:h, :w]
        center_mask = ((x_coords - center_x)**2 + (y_coords - center_y)**2) <= radius**2
        border_mask = ~center_mask
        
        # Contar pixels alterados no centro e na borda
        center_changed = np.sum(changed_mask & center_mask)
        border_changed = np.sum(changed_mask & border_mask)
        
        center_total = np.sum(center_mask)
        border_total = np.sum(border_mask)
        
        # Proporção de mudança (normalizada pela área)
        center_pct = (center_changed / center_total * 100) if center_total > 0 else 0
        border_pct = (border_changed / border_total * 100) if border_total > 0 else 0
        
        # Razão centro/borda: >1 = mais no centro (circular), <1 = mais na borda (mão)
        if border_pct > 0:
            center_ratio = center_pct / border_pct
        else:
            center_ratio = 10.0 if center_pct > 0 else 1.0
        
        # Peça circular: mudança concentrada no centro
        # Limite ajustado: peça redonda tem ~1.5x mais mudança no centro
        is_circular = center_ratio > 1.2 and pct_changed > 30
        
        # Classificar intensidade
        if pct_changed > 85:
            intensity = 'TOTAL'      # Mão tampando casa VAZIA completamente
        elif pct_changed > 50:
            intensity = 'PARCIAL'    # Peça movida (~70%) ou braço passando
        elif pct_changed > 15:
            intensity = 'LEVE'       # Sombra / borda de mão / mão sobre peça clara
        else:
            intensity = 'NENHUMA'    # Sem mudança significativa
        
        return {
            'z_score': np.mean(z_scores),
            'pct_changed': pct_changed,
            'intensity': intensity,
            'is_circular': is_circular,
            'center_ratio': center_ratio,
            'center_pct': center_pct,
            'border_pct': border_pct
        }
    
    def _update_gaussian(self, pos, current_processed, is_background=True):
        """
        Update Gaussian model for a square.
        
        If is_background: update mean and variance (pixel matched background)
        If not: don't update (pixel is foreground/changing)
        
        Using running average:
        μ_new = (1-α)·μ + α·current
        σ²_new = (1-α)·σ² + α·(current - μ)²
        """
        if pos not in self.gaussian_models:
            return
        
        model = self.gaussian_models[pos]
        mean = model['mean']
        variance = model['variance']
        
        # Ensure shapes match
        if current_processed.shape != mean.shape:
            current_processed = cv2.resize(
                current_processed,
                (mean.shape[1], mean.shape[0])
            ).astype(np.float64)
        
        if is_background:
            # Update mean
            new_mean = (1 - self.alpha) * mean + self.alpha * current_processed
            
            # Update variance
            diff_sq = (current_processed - mean) ** 2
            new_variance = (1 - self.alpha) * variance + self.alpha * diff_sq
            
            # Minimum variance to prevent instability
            new_variance = np.maximum(new_variance, 1.0)
            
            model['mean'] = new_mean
            model['variance'] = new_variance
    
    def set_focus_squares(self, squares_set):
        """
        Set which squares to monitor (radar optimization).
        
        When focus is set, detect_changes() only processes these squares,
        saving ~80% processing when we know legal move destinations.
        
        Args:
            squares_set: set of (file, rank) tuples to monitor
        """
        self.focus_squares = set(squares_set) if squares_set else None
    
    def clear_focus(self):
        """Clear focus - monitor all squares again."""
        self.focus_squares = None
    
    def get_focus_count(self):
        """Return number of focused squares, or 64 if no focus."""
        return len(self.focus_squares) if self.focus_squares else 64
    
    def detect_changes(self, squares_dict):
        """
        Find which squares changed from the background model.
        
        Uses Z-score: a square is changed if mean Z-score > z_threshold
        
        If focus_squares is set, only those squares are checked (optimization).
        
        Returns:
            dict: {(f,r): z_score} for squares that changed
        """
        if not self.is_calibrated:
            return {}
        
        changed = {}
        
        # Determine which squares to check
        squares_to_check = self.focus_squares if self.focus_squares else squares_dict.keys()
        
        for pos in squares_to_check:
            if pos not in squares_dict or pos not in self.gaussian_models:
                continue
            
            current_img = squares_dict[pos]
            current_processed = self._preprocess(current_img)
            model = self.gaussian_models[pos]
            
            if self.use_gaussian:
                # Gaussian mode: use Z-score
                z_score = self._calculate_zscore(current_processed, model)
                
                if z_score > self.z_threshold:
                    changed[pos] = z_score
                else:
                    # Square is stable, update background model
                    self._update_gaussian(pos, current_processed, is_background=True)
            else:
                # Fallback: simple difference
                diff = cv2.absdiff(
                    current_processed.astype(np.uint8),
                    model['mean'].astype(np.uint8)
                )
                mean_diff = np.mean(diff)
                
                if mean_diff > self.sensitivity:
                    changed[pos] = mean_diff
        
        return changed
    
    def detect_changes_detailed(self, squares_dict):
        """
        Análise fragmentada: detecta mudanças com detalhe de intensidade.
        
        Retorna informação sobre o TIPO de mudança em cada casa:
        - TOTAL (>80% pixels): mão cobrindo completamente
        - PARCIAL (30-80%): movimento de peça ou braço passando
        - LEVE (10-30%): sombra ou borda de mão
        - NENHUMA (<10%): sem mudança
        
        Returns:
            dict: {(f,r): {'z_score', 'pct_changed', 'intensity'}}
        """
        if not self.is_calibrated:
            return {}
        
        detailed = {}
        
        squares_to_check = self.focus_squares if self.focus_squares else squares_dict.keys()
        
        for pos in squares_to_check:
            if pos not in squares_dict or pos not in self.gaussian_models:
                continue
            
            current_img = squares_dict[pos]
            current_processed = self._preprocess(current_img)
            model = self.gaussian_models[pos]
            
            info = self._calculate_change_intensity(current_processed, model)
            
            # Só incluir se houve mudança significativa
            if info['intensity'] != 'NENHUMA':
                detailed[pos] = info
            else:
                # Atualizar modelo de background para casas estáveis
                self._update_gaussian(pos, current_processed, is_background=True)
        
        return detailed
    
    def classify_hand_pattern(self, detailed_changes):
        """
        Classifica o padrão de mudanças para identificar mão vs movimento.
        
        Usa is_circular para identificar peças redondas vs mãos.
        
        Padrões:
        - Casas PARCIAL circulares = peça movendo
        - Casas PARCIAL não-circulares = mão/braço
        - TOTAL = mão tampando completamente
        
        Returns:
            dict com:
                - is_hand: True se parece ser mão
                - is_move: True se parece movimento válido
                - hand_squares: casas com mão detectada
                - move_candidates: casas com peça (circulares)
        """
        total = [pos for pos, info in detailed_changes.items() if info['intensity'] == 'TOTAL']
        parcial = [pos for pos, info in detailed_changes.items() if info['intensity'] == 'PARCIAL']
        leve = [pos for pos, info in detailed_changes.items() if info['intensity'] == 'LEVE']
        
        # Separar PARCIAL em circular (peça) vs não-circular (mão)
        circular = [pos for pos, info in detailed_changes.items() 
                    if info['intensity'] == 'PARCIAL' and info.get('is_circular', False)]
        non_circular = [pos for pos, info in detailed_changes.items() 
                        if info['intensity'] == 'PARCIAL' and not info.get('is_circular', False)]
        
        result = {
            'is_hand': False,
            'is_move': False,
            'hand_squares': set(total + non_circular),  # Mão = TOTAL + PARCIAL não-circular
            'move_candidates': set(circular),            # Peça = PARCIAL circular
            'shadow_squares': set(leve),
        }
        
        # Muitas casas TOTAL = mão sobre tabuleiro
        if len(total) >= 3:
            result['is_hand'] = True
            return result
        
        # PARCIAL não-circulares + sombras = mão passando
        if len(non_circular) >= 2 and len(circular) == 0:
            result['is_hand'] = True
            return result
        
        # 2 PARCIAL circulares = movimento válido de peça!
        if len(circular) == 2:
            result['is_move'] = True
            return result
        
        # 1 PARCIAL circular = peça levantada
        if len(circular) == 1:
            result['move_candidates'] = set(circular)
            return result
        
        # 1 TOTAL + 1 circular = mão segurando peça, origem provável
        if len(total) == 1 and len(circular) >= 1:
            result['is_hand'] = True
            result['move_candidates'] = set(circular)
            return result
        
        return result
    
    def update_reference(self, pos, img):
        """
        Force update reference for a single square.
        Resets the Gaussian model to current state.
        """
        processed = self._preprocess(img)
        
        self.gaussian_models[pos] = {
            'mean': processed.copy(),
            'variance': np.full_like(processed, self.initial_variance, dtype=np.float64)
        }
        
        self.reference_squares[pos] = processed.astype(np.uint8)
    
    def update_all_references(self, squares_dict):
        """
        Force update all references after confirmed move.
        Resets Gaussian models to current state.
        """
        for pos, img in squares_dict.items():
            self.update_reference(pos, img)
    
    def get_debug_info(self, pos):
        """
        Get debug information for a specific square.
        
        Returns:
            dict with mean intensity, variance, sigma
        """
        if pos not in self.gaussian_models:
            return None
        
        model = self.gaussian_models[pos]
        return {
            'mean_intensity': np.mean(model['mean']),
            'mean_variance': np.mean(model['variance']),
            'mean_sigma': np.mean(np.sqrt(model['variance'])),
        }
    
    # ========== Compatibility methods ==========
    
    def train(self, squares_dict):
        """Alias for calibrate."""
        self.calibrate(squares_dict)
    
    def predict(self, square_img, pos=None):
        """
        For compatibility - returns if square changed from reference.
        """
        if not self.is_calibrated or pos is None or pos not in self.gaussian_models:
            return {'label': 'unknown', 'occupied': False, 'changed': False}
        
        current_processed = self._preprocess(square_img)
        model = self.gaussian_models[pos]
        
        z_score = self._calculate_zscore(current_processed, model)
        changed = z_score > self.z_threshold
        
        return {
            'label': 'changed' if changed else 'unchanged',
            'occupied': None,
            'changed': changed,
            'z_score': z_score,
            'diff': z_score  # For compatibility
        }
    
    def update_background(self, pos, square_img):
        """Alias for update_reference."""
        self.update_reference(pos, square_img)
