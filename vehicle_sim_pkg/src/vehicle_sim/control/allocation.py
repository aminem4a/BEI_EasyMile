import numpy as np

class TorqueAllocator:
    def __init__(self, loader):
        # Le loader sert uniquement de backup ou pour info, 
        # car maintenant tout le monde utilise les polynômes.
        self.loader = loader
        
        # --- 1. TES COEFFICIENTS (MODÈLE DE PERTES) ---
        # Formule : Loss = A*T^2 + B*w^2 + G*T*w + D*T + E*w + F
        
        # Cas Couple POSITIF (T >= 0)
        self.COEFFS_POS = {
            'A': -0.000125338082179,
            'B': -7.33783997958e-08,
            'D': 0.0272621224832,
            'E': 0.000657907533808,
            'F': -0.634473528053,
            'G': -5.31545729369e-06
        }
        
        # Cas Couple NEGATIF (T < 0)
        # Attention : le fit a été fait sur -T_data, donc on adapte les signes
        self.COEFFS_NEG = {
            'A': -0.000125338082179,
            'B': -7.33783997958e-08,
            'D': -0.0272621224832,  # Signe inversé car terme en T
            'E': 0.000657907533808,
            'F': -0.634473528053,
            'G': 5.31545729369e-06  # Signe inversé car terme en T*w
        }

    def get_poly_loss(self, T, rpm):
        """
        Calcule les pertes selon le modèle polynomial.
        Utilisé par TOUTES les stratégies.
        """
        # Sélection des coeffs selon le signe du couple
        if T >= 0:
            c = self.COEFFS_POS
        else:
            c = self.COEFFS_NEG
            
        w = abs(rpm) # Vitesse toujours positive dans le modèle (ou symétrique)
        
        # Application de la formule complète
        # L = A*T^2 + B*w^2 + G*T*w + D*T + E*w + F
        loss = (c['A'] * T**2 + 
                c['B'] * w**2 + 
                c['G'] * T * w + 
                c['D'] * T + 
                c['E'] * w + 
                c['F'])
        
        # On évite les pertes négatives (aberration du modèle à très faible charge)
        # On met un plancher à 0 ou une petite valeur de frottement
        if loss < 0:
            loss = 0.0
            
        return loss

    def optimize(self, strategy_name, T_req, rpm, prev_front_ratio=0.5):
        if abs(rpm) < 1.0: rpm = 1.0
        strat = strategy_name.lower() if strategy_name else "inverse"
        
        if "inverse" in strat:
            return self._strat_inverse(T_req, rpm)
        elif "piecewise" in strat:
            # Grid Search utilisant le polynôme
            return self._strat_search(T_req, rpm, use_smooth=False)
        elif "smooth" in strat:
            # Grid Search lissé utilisant le polynôme
            return self._strat_search(T_req, rpm, use_smooth=True, prev_ratio=prev_front_ratio)
        elif "quadratic" in strat:
            # Résolution analytique exacte des coefficients
            return self._strat_polynomial_analytical(T_req, rpm)
        else:
            return self._strat_inverse(T_req, rpm)

    def _strat_inverse(self, T_req, rpm):
        # 50/50
        return self._build(T_req * 0.5, T_req * 0.5, rpm)

    def _strat_search(self, T_req, rpm, use_smooth=False, prev_ratio=0.5):
        """
        Cherche le meilleur ratio en testant plein de combinaisons
        et en calculant le coût via get_poly_loss.
        """
        ratios = np.linspace(0, 1, 41) # Plus précis (pas de 2.5%)
        best_cost = float('inf')
        best_r = 0.5
        
        alpha = 200.0 if use_smooth else 0.0
        
        for r in ratios:
            Tf = T_req * r
            Tr = T_req * (1 - r)
            
            # ICI : On utilise bien le polynôme pour évaluer la performance
            loss = self.get_poly_loss(Tf, rpm) + self.get_poly_loss(Tr, rpm)
            
            cost = loss + alpha * (r - prev_ratio)**2
            
            if cost < best_cost:
                best_cost = cost
                best_r = r
                
        return self._build(T_req * best_r, T_req * (1 - best_r), rpm)

    def _strat_polynomial_analytical(self, T_req, rpm):
        """
        Résolution directe.
        Minimiser a*Tf^2 + b*Tf + ...
        """
        if T_req >= 0:
            coeffs = self.COEFFS_POS
        else:
            coeffs = self.COEFFS_NEG
            
        a = coeffs['A']
        # Le terme linéaire total dépend aussi de G*w et D
        # Loss(T) = A*T^2 + (D + G*w)*T + (Cstes en w)
        # On minimise Loss(Tf) + Loss(Tr) avec Tr = T_req - Tf
        
        # Si a < 0 (Concave, ce qui est ton cas avec -0.000125)
        # Le minimum est sur les bords (0 ou T_req)
        if a < 0:
            # On compare les deux extrêmes pour être sûr
            loss_tout_avant = self.get_poly_loss(T_req, rpm) + self.get_poly_loss(0, rpm)
            loss_tout_arriere = self.get_poly_loss(0, rpm) + self.get_poly_loss(T_req, rpm)
            
            if loss_tout_avant < loss_tout_arriere:
                Tf = T_req
            else:
                Tf = 0.0
                
            # Si T_req est très petit, on reste au milieu pour la stabilité
            if abs(T_req) < 1.0:
                Tf = T_req * 0.5
                
        else:
            # Cas Convexe (Classique) : Répartition équilibrée 50/50 car moteurs identiques
            Tf = T_req * 0.5

        Tr = T_req - Tf
        return self._build(Tf, Tr, rpm)

    def _build(self, Tf, Tr, rpm):
        """
        Construit la réponse finale.
        IMPORTANT : On renvoie ici la perte polynomiale pour l'affichage graphique.
        """
        lf = self.get_poly_loss(Tf, rpm)
        lr = self.get_poly_loss(Tr, rpm)
        return {
            'T_front': Tf,
            'T_rear': Tr,
            'P_loss': lf + lr
        }