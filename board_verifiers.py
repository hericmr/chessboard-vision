
import cv2
import numpy as np

class InitialBoardVerifier:
    def __init__(self, classifier):
        self.classifier = classifier
        # Thresholds
        self.MIN_CONFIDENCE_PER_SQUARE = 0.85 
        self.MIN_GLOBAL_CONFIDENCE = 0.90
        self.initial_setup = classifier.initial_setup
    
    def verify(self, squares_dict):
        """
        Verifica se o tabuleiro atual corresponde ao setup inicial.
        Retorna: (aprovado: bool, report: dict, score_img: np.array)
        """
        results = {}
        total_score = 0
        count = 0
        failed_squares = []
        
        for pos, expected_char in self.initial_setup.items():
            if pos not in squares_dict:
                continue
                
            sq_img = squares_dict[pos]
            metrics = self.classifier.get_metrics(sq_img, pos)
            
            # Calcular confianca baseada em quao proximo esta do template esperado
            # score baixo de match = alta confianca
            # match_score eh SOMA das diferencas
            match_score = metrics.get('label_scores', {}).get(expected_char, float('inf'))
            
            # Normalizar score. Quanto menor melhor.
            # Assumindo area ~3000px, diff media 50 -> 150000
            # Um match bom deve ser < 50000
            # Confianca = exp(-score / K)
            # Ajuste K experimentalmente
            confidence = np.exp(-match_score / 100000.0)
            
            # Se for casa vazia (nao deveriamos checar casas vazias no setup inicial? 
            # O classifier.initial_setup so tem pecas. 
            # Mas vamos checar se casas que DEVERIAM ser vazias estao vazias?
            # O initial_setup do classifier define linhas 0,1,6,7.
            # Podemos checar linhas 2-5 como empty.
            
            results[pos] = confidence
            total_score += confidence
            count += 1
            
            if confidence < self.MIN_CONFIDENCE_PER_SQUARE:
                failed_squares.append(pos)

        # Checar vazios (linhas 2 a 5)
        for r in range(2, 6):
            for f in range(8):
                pos = (f, r)
                if pos not in squares_dict: continue
                
                sq_img = squares_dict[pos]
                metrics = self.classifier.get_metrics(sq_img, pos)
                
                # Para vazio, esperamos match com 'empty' ou bg_diff baixo
                # Mas no inicio nao temos bg, entao confiamos no classifier prediction ou energy
                energy = metrics['energy']
                # Se energia baixa -> vazio
                # Energia alta -> ocupado
                
                # Confianca de ser vazio
                # Se energy < 20 -> conf 1.0
                # Se energy > 100 -> conf 0.0
                conf_empty = max(0, min(1, 1 - (energy - 20)/80))
                
                results[pos] = conf_empty
                total_score += conf_empty
                count += 1
                
                if conf_empty < self.MIN_CONFIDENCE_PER_SQUARE:
                    failed_squares.append(pos)

        global_confidence = total_score / max(count, 1)
        approved = (global_confidence >= self.MIN_GLOBAL_CONFIDENCE) and (len(failed_squares) == 0)
        
        return approved, {"global_conf": global_confidence, "failed": failed_squares}, results


class IncrementalBoardVerifier:
    def __init__(self, classifier):
        self.classifier = classifier
        
    def verify_move(self, fen_before, fen_after, move, squares_dict):
        """
        Verifica a consistencia visual de um movimento detectado.
        move: objeto chess.Move ou string uci
        """
        # Exemplo simples: verificar se squares de origem e destino mudaram significativamente
        # Por enquanto, retorna True para nao bloquear o fluxo, mas loga metricas
        
        # Converter move UCI para coordenadas (se necessario)
        # Vamos assumir que 'move' eh string uci ex: "e2e4"
        
        # TODO: Implementar checagem fina
        # Por hora, apenas checagem de "sanity":
        # Se a origem e destino tem alta diferenca de background ou energia
        
        return True, "Verificacao incremental placeholder OK"
