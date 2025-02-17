import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt


# Classe des systèmes flous
class SystemeFlou:
    
    # fonction à appliquer selon la t-norme choisie
    functable = {"min": np.fmin.reduce,
                 "proba": np.prod}
    
    
    # initialisation du systeme flou, la t-norme par défaut est min
    def __init__(self, entrees:list, regles:dict, t_norme:str="min"):
        
        for entree in entrees:
            if (not isinstance(entree, Entree_nette)) and (not isinstance(entree, Entree_floue)):
                raise TypeError("La liste des entrées doit contenir des objets de type entrées")
        
        nombre_de_regles_necessaires = 1
        for i in entrees:
            nombre_de_regles_necessaires *= len(i.partition)
        
        if len(regles) != nombre_de_regles_necessaires:
            print(len(regles), nombre_de_regles_necessaires)
            raise ValueError("Le nombre de règles de votre système ne correspond pas à la partition de vos variables.")
        
        self.regles = regles
        self.t_norme = self.functable[t_norme]
        
        # dictionnaire regroupant toutes les entrées fuzzifiées du systeme
        self.entrees_floues = {variable.nom: variable.entree_floue for variable in entrees}
        
        
        
    # calcule les degrés d'activation des regles du systeme flou
    def activation_regles(self):
        
        # initialisation du dictionnaire d'activation des différentes conclusions possibles
        activations = {conclusion: 0 for conclusion in self.regles.values()}
        
        # pour chaque regle dans le dictionnaire de règles
        for conditions, conclusion in self.regles.items():
            
            # calcule les degrés d'appartenance pour toutes les conditions de la règle
            #initialisation de la table contenant les degrés d'appartenance des classes floues des conditions
            degres = []
            # pour chaque condition de la regles (minimum 2) de la forme (entrée, classe floue de l'entrée)
            for entree, classe_floue in conditions:
                # on ajoute le degré d'appartenance de la condition à la table des degrés
                degres.append(self.entrees_floues[entree][classe_floue])
            
            # calcul du degré d'activation de la regle avec la t-norme du systeme flou
            activation = self.t_norme(degres)
            
            # max entre la valeur précédente d'activation de la conclusion et la nouvelle ~= la max-union des conclusions floues
            activations[conclusion] = max(activations[conclusion], activation)
        
        # retourne l'activation de chaque conclusion possible aux regles dans un dictionnaire {conclusion: degré d'activation}
        return activations
    
    def sortie_floue_non_normalisée(self, nom:str):
        sortie_initiale = self.activation_regles()
        partition = [classe_floue for classe_floue in sortie_initiale.keys()]
        valeurs = [degre for degre in sortie_initiale.values()]
        sortie_finale = Entree_floue(nom, partition, valeurs)
        return sortie_finale

    def sortie_floue_normalisée(self, nom:str):
        sortie_initiale = self.activation_regles()
        partition = [classe_floue for classe_floue in sortie_initiale.keys()]
        valeurs = [degre for degre in sortie_initiale.values()]
        sortie_finale = Entree_floue(nom, partition, valeurs)
        sortie_finale.normaliser()
        return sortie_finale
    
    def sortie_defuzzifiee(self, nom:str, valeurs_regression:list, gamma:int=1):
        sortie_floue = self.sortie_floue_normalisée(nom)
        return sortie_floue.defuzzification(valeurs_regression, gamma)



#######################################################################################################################################



# Classe pour les entrées nettes qu'on va fuzzifier
class Entree_nette:
    
    # univers de la forme (x1, x2, pas)
    # partition est un dictonnaire de la forme {label: [x1, x2, x3, x4]} où les x sont les coordonnées des trapezes en notation de Kaufmann
    # valeur pas nécessairement donnée en initialisation
    def __init__(self, nom:str, univers:list, partition:dict, valeur=None):
        self.nom = nom
        self.univers = np.linspace(*univers)
        
        # partition floue de l'univers de la variable
        self.partition = {}
        for label in partition.keys():
            self.partition[str(label)] = fuzz.trapmf(self.univers, partition[label])
        
        # En créant une entrée on est pas nécessairement obligé de donner directement la valeur qui correspond,
        # on peut la définir plus tard en écrivant nom de l'entrée.entree_nette = valeur voulue
        if valeur is not None:
            self.entree_nette = valeur
            
    @property
    def entree_nette(self):
        return self._entree_nette
    
    @property
    def entree_floue(self):
        return self._entree_floue
    
    @entree_nette.setter
    def entree_nette(self, valeur):
        self._entree_nette = valeur
        
        # Met à jour l'entrée fuzzifiée automatiquement pour qu'elle corresponde toujours à l'entrée nette
        self.entree_floue = valeur
    
    # prend une valeur en entrée et retourne le dictionnaire donnant le degré d'appartenance de la valeur à chaque classe floue
    @entree_floue.setter
    def entree_floue(self, valeur):
        self._entree_floue = {}
        for label, fonction_appartenance in self.partition.items():
            # fuzzifie la valeur sur la partition floue de la variable
            self._entree_floue[label] = fuzz.interp_membership(self.univers, fonction_appartenance, valeur)
    
    # Pour afficher les fonctions d'appartenance avec matplotlib
    def afficher_fonctions_appartenance(self, titre:str="", label_x:str="", label_y:str=""):
        # créé la figure
        plt.figure(figsize=(8, 5))
        
        # trace les fonctions d'appartenance
        for label in self.partition.keys():
            plt.plot(self.univers, self.partition[label], label=str(label))
        
        # ajoute des titres et légendes
        plt.title(titre)
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.legend(loc="lower right")
        plt.grid()

        # affiche le graphique
        plt.show()

##########################################################################################################################################

# Classe pour les entrées déjà floues qu'on a pas besoin de fuzzifier
# on l'utilise aussi pour les sorties de systeme flou mais vas y
class Entree_floue:
    
    def __init__(self, nom:str, partition:list, valeur:list=None):
        self.nom = nom
        
        # contrairement à entrée nette on a seulement besoin des label des classes floues pour la partition car on ne fuzzifie pas
        self.partition = {str(classe_floue): 0 for classe_floue in partition}
        self._entree_floue = self.partition
        
        # meme chose que Entree_nette on n'a pas besoin de definir la valeur à l'initialisation de l'entrée
        if valeur is not None:
            self.entree_floue = valeur
    
    @property
    def entree_floue(self):
        return self._entree_floue
    
    # faut donner les degrés d'appartenance dans le meme ordre que la definition de la partition floue!
    @entree_floue.setter
    def entree_floue(self, valeur:list):
        i = 0
        for classe_floue in self.entree_floue:
            self._entree_floue[classe_floue] = valeur[i]
            i += 1
    
    # normalise l'entrée floue pour l'algorithme de Zalila généralisé ou jsp quoi
    def normaliser(self):
        # on met toutes les hauteurs des classes floues activées dans une liste pour avoir la hauteur max
        valeurs = [degre_appartenance for degre_appartenance in self._entree_floue.values()]
        hauteur_max = max(valeurs)
        # On évite la division par zéro
        if hauteur_max > 0:
            for classe_floue in self._entree_floue:
                # on divise chaque degré d'appartenance par la hauteur max pour normaliser
                self._entree_floue[classe_floue] /= hauteur_max
        else:
            raise ValueError("Attention : Les valeurs des degrés d'appartenance sont toutes nulles. Aucune normalisation effectuée.")
    
    # defuzzification par methode barycentrique ZZ-gamma
    # donner les valeurs de régression dans le meme ordre que la définition de la partition!
    def defuzzification(self, valeurs_regression:list, gamma:int=1):
        
        numerateur = 0
        denominateur = 0
        i = 0
        
        # c'est la formule
        for classe_floue, degre_appartenance in self._entree_floue.items():
            print(classe_floue, degre_appartenance, valeurs_regression[i])
            numerateur += valeurs_regression[i] * degre_appartenance ** gamma
            denominateur += degre_appartenance ** gamma
            i += 1
        print()
        
        if denominateur > 0:
            return numerateur / denominateur
        else:
            raise ValueError("Attention : Les valeurs des degrés d'appartenance sont toutes nulles. Aucune normalisation effectuée.")
    
    
    
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################



#calcul de la maintenance grace à la formule de harris benedict
def calcul_maintenance(taille,poid,age,sexe,activite):
    if sexe == "M" :
        bmr = (66.5 + (13.75*poid) + (5.003*taille) - (6.75*age))
    elif sexe == "F" :
        bmr = (655.1 + (9.563*poid) + (1.850*taille) - (4.676*age))
    if activite == 1 :
        bmr = bmr*1.2
    elif activite == 2 :
        bmr = bmr*1.375
    elif activite == 3 :
        bmr = bmr*1.55
    elif activite == 4 :
        bmr = bmr*1.725
    return bmr

def trouver_maximum_prioritaire_alpha(fuzzifications, ordre_priorite, alpha=0.3):
    """
    Trouve la catégorie avec la priorité la plus élevée parmi celles activées, en appliquant une alpha-coupe.
    Si aucune catégorie ne dépasse l'alpha, revient au protocole précédent.
    
    Args:
        fuzzifications (dict): Fuzzifications des parties du corps.
        ordre_priorite (list): Liste des catégories dans l'ordre de priorité.
        alpha (float): Seuil minimum pour qu'une catégorie soit considérée.
        
    Returns:
        str: La catégorie avec la priorité la plus élevée activée.
    """
    # Étape 1 : Identifier les catégories activées au-delà de l'alpha-coupe
    categories_valides = set()  # Contiendra les catégories activées au-dessus de l'alpha
    for partie, fuzzification in fuzzifications.items():
        for label, activation in fuzzification.items():
            if activation >= alpha:  # Si activation dépasse alpha
                categories_valides.add(label)

    # Étape 2 : Si aucune catégorie ne dépasse l'alpha, utiliser l'ancien protocole
    if not categories_valides:
        print(f"\nAlpha-coupe {alpha} : aucune catégorie activée au-delà de {alpha}")
        # Protocole précédent : maximum selon l'ordre de priorité
        for categorie in ordre_priorite:
            for partie, fuzzification in fuzzifications.items():
                if fuzzification.get(categorie, 0) > 0:  # Si activé
                    return categorie

    # Étape 3 : Trouver la catégorie valide avec la priorité la plus élevée
    for categorie in ordre_priorite:
        if categorie in categories_valides:
            return categorie

    return None  # Si aucune catégorie activée (cas théorique)

def calculer_macronutriments(calories):
    """
    Calcule les grammes de glucides, protéines et lipides 
    à partir du nombre total de calories.

    Répartition : 
    - 40% glucides
    - 35% protéines
    - 25% lipides
    
    Args:
        calories (float): Le nombre total de calories.
    
    Returns:
        dict: Les grammes de glucides, protéines et lipides.
    """
    # Calcul des calories pour chaque macro
    calories_glucides = calories * 0.45
    calories_proteines = calories * 0.25
    calories_lipides = calories * 0.30

    # Conversion en grammes
    glucides = int(calories_glucides / 4)  # 1g de glucides = 4 kcal
    proteines = int(calories_proteines / 4)  # 1g de protéines = 4 kcal
    lipides = int(calories_lipides / 9)  # 1g de lipides = 9 kcal

    # Retourner les résultats dans un dictionnaire
    return {
        "Glucides (g)": glucides,
        "Protéines (g)": proteines,
        "Lipides (g)": lipides
    }



def generer_programme(intensites_reelles):
    # Classement des parties du corps par intensité décroissante
    parties_tries = sorted(intensites_reelles.items(), key=lambda x: x[1], reverse=True)
    programme = []
    jours_utilises = 0
    jours_max = 6  # Limiter à 6 jours d'entraînement

    # Garder une trace du nombre de séances par partie
    parties_entrainees = {partie: 0 for partie in intensites_reelles}

    while jours_utilises < jours_max:
        ajouté = False
        for partie, intensite in parties_tries:
            # Vérifier si la partie peut être entraînée
            if parties_entrainees[partie] < 2 and jours_utilises < jours_max:
                if intensite > 20:
                    programme.append(f"Séance {partie} (très intense)")
                elif intensite > 15:
                    programme.append(f"Séance {partie} (intense)")
                elif intensite > 10:
                    programme.append(f"Séance {partie} (modérée)")
                elif intensite > 5:
                    programme.append(f"Séance {partie} (légère)")
                else:
                    continue
                parties_entrainees[partie] += 1
                ajouté = True
                jours_utilises += 1

        # Ajouter du repos si rien n'est ajouté
        if not ajouté and jours_utilises < jours_max:
            programme.append("Repos")
            jours_utilises += 1

    return programme



##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################


# Foncton pour initialiser toutes les variables fixes dont on aura besoin dans main
# Pour rendre main lisible et maintenable
def entrees_regles():
    
    # Variable Pourcentage de Masse Grasse
    mg_partition = {
        "sec": [0.06, 0.06, 0.13, 0.14],
        "normal": [0.13, 0.14, 0.17, 0.18],
        "gras": [0.17, 0.18, 0.24, 0.25],
        "très gras": [0.24, 0.25, 0.26, 0.26]
    }
    mg = Entree_nette("Masse grasse", (0.07, 0.25, 1000), mg_partition)
    
    # Variable IMC
    imc_partition = {
        "sous poids": [0, 0, 18.5, 20],
        "poids idéal": [18.5, 20, 25, 26],
        "surpoids": [25, 26, 30, 31],
        "faible obésité": [30, 31, 35, 36],
        "obésité moyenne": [35, 36, 40, 41],
        "obésité sévère": [40, 41, 50, 50]
    }
    imc = Entree_nette("IMC", (10, 50, 1000), imc_partition)
    
    # Variable objectif de masse musculaire
    objectif_musculaire_partition = {
        "perte": [-0.4, -0.4, -0.05, 0],
        "inchangé": [-0.05, 0, 0, 0.05],
        "gain modéré": [0, 0.05, 0.4, 0.6],
        "gros gain": [0.4, 0.6, 1.1, 1.1]
    }
    objectif_musculaire = Entree_nette("Objectif", (-0.3, 1, 1000), objectif_musculaire_partition)
    
    # Variable objectif de masse grasse
    objectif_mg = Entree_nette("Objectif MG", (0.07, 0.25, 1000), mg_partition)
    
    # Variable génétique d'une partie du corps (à quel point il gagne du muscle en l'entrainant)
    genetique_partition = {
        "Mauvaise": [0, 0, 0, 0],
        "Point faible": [1, 1, 1, 1],
        "Normal": [2, 2, 2, 2],
        "Point fort": [3, 3, 3, 3],
        "Excellente": [4, 4, 4, 4]
    }
    genetique = Entree_nette("Génétique", (0, 4, 5), genetique_partition)
    
    # Variable répondance au dopage
    dopage_impact_partition = {
        "Aucun impact": [0, 0, 0, 0],
        "Peu répondant": [1, 1, 1, 1],
        "Répondant": [2, 2, 2, 2],
        "Très répondant": [3, 3, 3, 3]
    }
    dopage_impact = Entree_nette("Impact du dopage", (0, 3, 4), dopage_impact_partition)
    
    # Variable sant d'une partie du corps 0% étant le max et 100% le min
    sante_partition = {
        "Aucune blessure": [0, 0, 0, 0.2],
        "Blessure légère": [0, 0.2, 0.3, 0.45],
        "Blessure moyenne": [0.3, 0.45, 0.6, 0.7],
        "Blessure grave": [0.6, 0.7, 1, 1]
    }
    sante = Entree_nette("Santé", (0, 1, 1000), sante_partition)
    
    # Variable Apport calorique absolu
    apport_calories_partition = {
        "Apport insuffisant": [1000, 1000, 1500, 2000],  # Apport Insuffisant
        "Apport Faible": [1500, 2000, 2500, 3000],  # Apport Faible
        "Apport Suffisant": [2500, 3000, 3500, 4000],  # Apport Suffisant
        "Apport plus que Suffisant": [3500, 4000, 4500, 4500]  # Apport Plus que Suffisant
    }
    apport_calories = Entree_nette("Apports caloriques", (1000, 4500, 10000), apport_calories_partition)
    
    ##########################################################################################################################################
    
    # Règles d'inférence floue du SIF Condition Biologique
    regles_bio = {
        (("Masse grasse", "sec"), ("IMC", "sous poids")): "NM",
        (("Masse grasse", "sec"), ("IMC", "poids idéal")): "NM",
        (("Masse grasse", "sec"), ("IMC", "surpoids")): "TM",
        (("Masse grasse", "sec"), ("IMC", "faible obésité")): "TM",
        (("Masse grasse", "sec"), ("IMC", "obésité moyenne")): "EM",
        (("Masse grasse", "sec"), ("IMC", "obésité sévère")): "EM",
        (("Masse grasse", "normal"), ("IMC", "sous poids")): "LM",
        (("Masse grasse", "normal"), ("IMC", "poids idéal")): "NM",
        (("Masse grasse", "normal"), ("IMC", "surpoids")): "NM",
        (("Masse grasse", "normal"), ("IMC", "faible obésité")): "TM",
        (("Masse grasse", "normal"), ("IMC", "obésité moyenne")): "NM",
        (("Masse grasse", "normal"), ("IMC", "obésité sévère")): "EM",
        (("Masse grasse", "gras"), ("IMC", "sous poids")): "TPM",
        (("Masse grasse", "gras"), ("IMC", "poids idéal")): "LM",
        (("Masse grasse", "gras"), ("IMC", "surpoids")): "TPM",
        (("Masse grasse", "gras"), ("IMC", "faible obésité")): "LM",
        (("Masse grasse", "gras"), ("IMC", "obésité moyenne")): "LM",
        (("Masse grasse", "gras"), ("IMC", "obésité sévère")): "LM",
        (("Masse grasse", "très gras"), ("IMC", "sous poids")): "TPM",
        (("Masse grasse", "très gras"), ("IMC", "poids idéal")): "TPM",
        (("Masse grasse", "très gras"), ("IMC", "surpoids")): "TPM",
        (("Masse grasse", "très gras"), ("IMC", "faible obésité")): "TPM",
        (("Masse grasse", "très gras"), ("IMC", "obésité moyenne")): "TPM",
        (("Masse grasse", "très gras"), ("IMC", "obésité sévère")): "TPM"
    }
    
    # Règles d'inférence floue du SIF Nutrition 1
    regles_nutrition_1 = {
        (("Conditions", "TPM"), ("Objectif Musculaire Maximum", "perte")): "DANGER",
        (("Conditions", "TPM"), ("Objectif Musculaire Maximum", "inchangé")): "PC",
        (("Conditions", "TPM"), ("Objectif Musculaire Maximum", "gain modéré")): "AIA",
        (("Conditions", "TPM"), ("Objectif Musculaire Maximum", "gros gain")): "AIA",
        (("Conditions", "LM"), ("Objectif Musculaire Maximum", "perte")): "PC",
        (("Conditions", "LM"), ("Objectif Musculaire Maximum", "inchangé")): "PC",
        (("Conditions", "LM"), ("Objectif Musculaire Maximum", "gain modéré")): "AIA",
        (("Conditions", "LM"), ("Objectif Musculaire Maximum", "gros gain")): "AIA",
        (("Conditions", "NM"), ("Objectif Musculaire Maximum", "perte")): "DA",
        (("Conditions", "NM"), ("Objectif Musculaire Maximum", "inchangé")): "PC",
        (("Conditions", "NM"), ("Objectif Musculaire Maximum", "gain modéré")): "AA",
        (("Conditions", "NM"), ("Objectif Musculaire Maximum", "gros gain")): "AA",
        (("Conditions", "TM"), ("Objectif Musculaire Maximum", "perte")): "DIA",
        (("Conditions", "TM"), ("Objectif Musculaire Maximum", "inchangé")): "PC",
        (("Conditions", "TM"), ("Objectif Musculaire Maximum", "gain modéré")): "AA",
        (("Conditions", "TM"), ("Objectif Musculaire Maximum", "gros gain")): "PC",
        (("Conditions", "EM"), ("Objectif Musculaire Maximum", "perte")): "DIA",
        (("Conditions", "EM"), ("Objectif Musculaire Maximum", "inchangé")): "PC",
        (("Conditions", "EM"), ("Objectif Musculaire Maximum", "gain modéré")): "PC",
        (("Conditions", "EM"), ("Objectif Musculaire Maximum", "gros gain")): "PC",
    }
    
    # Règles d'inférence floue du SIF Nutrition 2
    regles_nutrition_2 = {
        (("Nutrition Provisoire", "DANGER"), ("Objectif MG", "sec")): "DANGER",
        (("Nutrition Provisoire", "DANGER"), ("Objectif MG", "normal")): "DANGER",
        (("Nutrition Provisoire", "DANGER"), ("Objectif MG", "gras")): "DANGER",
        (("Nutrition Provisoire", "DANGER"), ("Objectif MG", "très gras")): "DANGER",
        (("Nutrition Provisoire", "DIA"), ("Objectif MG", "sec")): "DIA",
        (("Nutrition Provisoire", "DIA"), ("Objectif MG", "normal")): "DIA",
        (("Nutrition Provisoire", "DIA"), ("Objectif MG", "gras")): "DIA",
        (("Nutrition Provisoire", "DIA"), ("Objectif MG", "très gras")): "DA",
        (("Nutrition Provisoire", "DA"), ("Objectif MG", "sec")): "DIA",
        (("Nutrition Provisoire", "DA"), ("Objectif MG", "normal")): "DA",
        (("Nutrition Provisoire", "DA"), ("Objectif MG", "gras")): "DA",
        (("Nutrition Provisoire", "DA"), ("Objectif MG", "très gras")): "PC",
        (("Nutrition Provisoire", "PC"), ("Objectif MG", "sec")): "DA",
        (("Nutrition Provisoire", "PC"), ("Objectif MG", "normal")): "DA",
        (("Nutrition Provisoire", "PC"), ("Objectif MG", "gras")): "PC",
        (("Nutrition Provisoire", "PC"), ("Objectif MG", "très gras")): "AA",
        (("Nutrition Provisoire", "AA"), ("Objectif MG", "sec")): "PC",
        (("Nutrition Provisoire", "AA"), ("Objectif MG", "normal")): "PC",
        (("Nutrition Provisoire", "AA"), ("Objectif MG", "gras")): "AA",
        (("Nutrition Provisoire", "AA"), ("Objectif MG", "très gras")): "AIA",
        (("Nutrition Provisoire", "AIA"), ("Objectif MG", "sec")): "PC",
        (("Nutrition Provisoire", "AIA"), ("Objectif MG", "normal")): "AA",
        (("Nutrition Provisoire", "AIA"), ("Objectif MG", "gras")): "AIA",
        (("Nutrition Provisoire", "AIA"), ("Objectif MG", "très gras")): "AIA",
    }
    
    # Règles d'inférence floue du SIF Intensité Nécessaire 1
    regles_intensite_necessaire_1 = {
        (("Génétique", "Mauvaise"), ("Objectif", "perte")): "N",
        (("Génétique", "Mauvaise"), ("Objectif", "inchangé")): "M",
        (("Génétique", "Mauvaise"), ("Objectif", "gain modéré")): "I",
        (("Génétique", "Mauvaise"), ("Objectif", "gros gain")): "TI",
        (("Génétique", "Point faible"), ("Objectif", "perte")): "N",
        (("Génétique", "Point faible"), ("Objectif", "inchangé")): "M",
        (("Génétique", "Point faible"), ("Objectif", "gain modéré")): "I",
        (("Génétique", "Point faible"), ("Objectif", "gros gain")): "TI",
        (("Génétique", "Normal"), ("Objectif", "perte")): "N",
        (("Génétique", "Normal"), ("Objectif", "inchangé")): "F",
        (("Génétique", "Normal"), ("Objectif", "gain modéré")): "M",
        (("Génétique", "Normal"), ("Objectif", "gros gain")): "TI",
        (("Génétique", "Point fort"), ("Objectif", "perte")): "N",
        (("Génétique", "Point fort"), ("Objectif", "inchangé")): "TF",
        (("Génétique", "Point fort"), ("Objectif", "gain modéré")): "F",
        (("Génétique", "Point fort"), ("Objectif", "gros gain")): "I",
        (("Génétique", "Excellente"), ("Objectif", "perte")): "N",
        (("Génétique", "Excellente"), ("Objectif", "inchangé")): "TF",
        (("Génétique", "Excellente"), ("Objectif", "gain modéré")): "F",
        (("Génétique", "Excellente"), ("Objectif", "gros gain")): "I",
        }
    
    # Règles d'inférence floue du SIF Intensité Nécessaire 2
    regles_intensite_necessaire_2 = {
        (("Intensité nécessaire intermédiaire", "N"), ("Impact du dopage", "Aucun impact")): "N",
        (("Intensité nécessaire intermédiaire", "N"), ("Impact du dopage", "Peu répondant")): "N",
        (("Intensité nécessaire intermédiaire", "N"), ("Impact du dopage", "Répondant")): "N",
        (("Intensité nécessaire intermédiaire", "N"), ("Impact du dopage", "Très répondant")): "N",
        (("Intensité nécessaire intermédiaire", "TF"), ("Impact du dopage", "Aucun impact")): "TF",
        (("Intensité nécessaire intermédiaire", "TF"), ("Impact du dopage", "Peu répondant")): "TF",
        (("Intensité nécessaire intermédiaire", "TF"), ("Impact du dopage", "Répondant")): "N",
        (("Intensité nécessaire intermédiaire", "TF"), ("Impact du dopage", "Très répondant")): "N",
        (("Intensité nécessaire intermédiaire", "F"), ("Impact du dopage", "Aucun impact")): "F",
        (("Intensité nécessaire intermédiaire", "F"), ("Impact du dopage", "Peu répondant")): "F",
        (("Intensité nécessaire intermédiaire", "F"), ("Impact du dopage", "Répondant")): "TF",
        (("Intensité nécessaire intermédiaire", "F"), ("Impact du dopage", "Très répondant")): "TF",
        (("Intensité nécessaire intermédiaire", "M"), ("Impact du dopage", "Aucun impact")): "M",
        (("Intensité nécessaire intermédiaire", "M"), ("Impact du dopage", "Peu répondant")): "M",
        (("Intensité nécessaire intermédiaire", "M"), ("Impact du dopage", "Répondant")): "F",
        (("Intensité nécessaire intermédiaire", "M"), ("Impact du dopage", "Très répondant")): "TF",
        (("Intensité nécessaire intermédiaire", "I"), ("Impact du dopage", "Aucun impact")): "I",
        (("Intensité nécessaire intermédiaire", "I"), ("Impact du dopage", "Peu répondant")): "I",
        (("Intensité nécessaire intermédiaire", "I"), ("Impact du dopage", "Répondant")): "M",
        (("Intensité nécessaire intermédiaire", "I"), ("Impact du dopage", "Très répondant")): "F",
        (("Intensité nécessaire intermédiaire", "TI"), ("Impact du dopage", "Aucun impact")): "TI",
        (("Intensité nécessaire intermédiaire", "TI"), ("Impact du dopage", "Peu répondant")): "TI",
        (("Intensité nécessaire intermédiaire", "TI"), ("Impact du dopage", "Répondant")): "I",
        (("Intensité nécessaire intermédiaire", "TI"), ("Impact du dopage", "Très répondant")): "I",
        }
    
    # Règles d'inférence floue du SIF Intensité Possible
    regles_intensite_possible = {
        (("Santé", "Aucune blessure"), ("Apports caloriques", "Apport insuffisant")): "M",
        (("Santé", "Aucune blessure"), ("Apports caloriques", "Apport Faible")): "I",
        (("Santé", "Aucune blessure"), ("Apports caloriques", "Apport Suffisant")): "TI",
        (("Santé", "Aucune blessure"), ("Apports caloriques", "Apport plus que Suffisant")): "TI",
        (("Santé", "Blessure légère"), ("Apports caloriques", "Apport insuffisant")): "F",
        (("Santé", "Blessure légère"), ("Apports caloriques", "Apport Faible")): "M",
        (("Santé", "Blessure légère"), ("Apports caloriques", "Apport Suffisant")): "M",
        (("Santé", "Blessure légère"), ("Apports caloriques", "Apport plus que Suffisant")): "I",
        (("Santé", "Blessure moyenne"), ("Apports caloriques", "Apport insuffisant")): "N",
        (("Santé", "Blessure moyenne"), ("Apports caloriques", "Apport Faible")): "TF",
        (("Santé", "Blessure moyenne"), ("Apports caloriques", "Apport Suffisant")): "M",
        (("Santé", "Blessure moyenne"), ("Apports caloriques", "Apport plus que Suffisant")): "M",
        (("Santé", "Blessure grave"), ("Apports caloriques", "Apport insuffisant")): "N",
        (("Santé", "Blessure grave"), ("Apports caloriques", "Apport Faible")): "N",
        (("Santé", "Blessure grave"), ("Apports caloriques", "Apport Suffisant")): "N",
        (("Santé", "Blessure grave"), ("Apports caloriques", "Apport plus que Suffisant")): "N",
        }
    
    # Dictionnaire contenant toutes ces variables fixées
    return {
        "Masse grasse" : mg,
        "IMC" : imc,
        "Objectif Musculaire" : objectif_musculaire,
        "Objectif Masse Grasse" : objectif_mg,
        "Génétique" : genetique,
        "Impact du dopage" : dopage_impact,
        "Santé" : sante,
        "Apports caloriques" : apport_calories,
        "regles SIF Conditions Biologiques" : regles_bio,
        "regles SIF Nutrition 1" : regles_nutrition_1,
        "regles SIF Nutrition 2" : regles_nutrition_2,
        "regles SIF Intensité Nécessaire 1" : regles_intensite_necessaire_1,
        "regles SIF Intensité Nécessaire 2" : regles_intensite_necessaire_2,
        "regles SIF Intensité Possible" : regles_intensite_possible
    }
    
    
    



##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

def main():

    # Toutes les données des entrées et règles pour rendre la fonction main lisible
    d = entrees_regles()
    
    
    d["Masse grasse"].entree_nette = 0.18
    input_age = 70
    input_taille = 169
    input_sexe = "M"
    input_poids = 60
    input_activite = 1
    objectifs_nets = {
        "Bras": 0.3,
        "Jambes": -0.1,
        "Dos": 0.7,
        "Torse": -0.1
    }
    d["Objectif Masse Grasse"].entree_nette = 0.18
    dopage_OK = 0
    repondance = 3
    genetiques = {
        "Bras": 1,
        "Jambes": 0,
        "Dos": 2,
        "Torse": 3
    }
    santes = {
        "Bras": 0,
        "Jambes": 0,
        "Dos": 0.7,
        "Torse": 0.4
    }
    
    '''Exemple 2
    d["Masse grasse"].entree_nette = 0.18
    input_age = 26
    input_taille = 180
    input_sexe = "F"
    input_poids = 90
    input_activite = 2
    objectifs_nets = {
        "Bras": 0.5,
        "Jambes": 0.8,
        "Dos": 0.3,
        "Torse": 0.8
    }
    d["Objectif Masse Grasse"].entree_nette = 0.20
    dopage_OK = 1
    repondance = 3
    genetiques = {
        "Bras": 1,
        "Jambes": 2,
        "Dos": 2,
        "Torse": 1
    }
    santes = {
        "Bras": 0,
        "Jambes": 0,
        "Dos": 0,
        "Torse": 0
    }
    '''

    '''exemple 3
    d["Masse grasse"].entree_nette = 0.25
    input_age = 26
    input_taille = 175
    input_sexe = "M"
    input_poids = 60
    input_activite = 1
    objectifs_nets = {
        "Bras": -0.1,
        "Jambes": -0.1,
        "Dos": -0.1,
        "Torse": -0.1
    }
    d["Objectif Masse Grasse"].entree_nette = 0.20
    dopage_OK = 0
    repondance = 3
    genetiques = {
        "Bras": 1,
        "Jambes": 2,
        "Dos": 2,
        "Torse": 1
    }
    santes = {
        "Bras": 0,
        "Jambes": 0,
        "Dos": 0,
        "Torse": 0
    }
    '''
    
    
    
    ###################################################### CONDITIONS BIOLOGIQUES #####################################################################
    
    
    print("Bienvenue dans le système flou pour l'évaluation biologique.")
    
    '''
    ###
    print("Veuillez entrer vos valeurs pour la masse grasse dans les plages suivantes :")
    print("- Masse grasse : entre 0.07 et 0.25 (en pourcentage)")
    d["Masse grasse"].entree_nette = float(input("Entrez votre pourcentage de masse grasse (valeur brute) : "))
    input_age = int(input("Entrez votre age : "))
    input_taille = int(input("Entrez votre taille en cm : "))
    input_sexe = str(input("Entrez votre sexe (M ou F): "))
    input_poids = float(input("Entrez votre poids en kg : "))
    input_activite = int(input("Entrez votre niveau d'activité (de 1 à 4) : "))
    ###
    '''
    
    calories_de_maintenance = calcul_maintenance(input_taille, input_poids, input_age, input_sexe, input_activite)
    d["IMC"].entree_nette = input_poids / (input_taille / 100) ** 2
    # Création du système flou
    SIF_conditions = SystemeFlou([d["Masse grasse"], d["IMC"]], d["regles SIF Conditions Biologiques"])
    condition_biologique = SIF_conditions.sortie_floue_normalisée("Conditions")
    
    print("\nSystème Conditions biologiques terminé avec succès.")
    try:
        print("\nDegrés d'activation des conclusions :")
        for conclusion, activation in condition_biologique.entree_floue.items():
            print(f"{conclusion}: {activation:.2f}")
    except ValueError as e:
        print(f"Erreur : {e}")
    except Exception as e:
        print(f"Une erreur inattendue s'est produite : {e}")
    
    
    
    ###################################################### OBJECTIFS #########################################################################

    
    
    print("Bienvenue dans le système flou pour les objectifs musculaires.")
    
    '''
    ###
    print("Veuillez entrer vos objectifs musculaires pour les 4 parties du corps (entre -0.3 et 1) :")
    objectifs_nets = {
        "Bras": float(input("Objectif pour les bras (-0.3 à 1) : ")),
        "Jambes": float(input("Objectif pour les jambes (-0.3 à 1) : ")),
        "Dos": float(input("Objectif pour le dos (-0.3 à 1) : ")),
        "Torse": float(input("Objectif pour le torse (-0.3 à 1) : "))
    }
    ###
    '''
    
    objectifs_fuzzifies = {}
    for partie_du_corps, valeur in objectifs_nets.items():
        d["Objectif Musculaire"].entree_nette = valeur
        a = Entree_floue("Objectif", list(d["Objectif Musculaire"].entree_floue.keys()), list(d["Objectif Musculaire"].entree_floue.values()))
        a.normaliser()
        objectifs_fuzzifies[partie_du_corps] = a
    
    print("\nFuzzification des objectifs musculaires :")
    for partie, objectif in objectifs_fuzzifies.items():
        print(f"{partie.capitalize()} : {objectif.entree_floue}")
    
    ordre_priorite = ["gros gain", "gain modéré", "inchangé", "perte"]
    objectif_musculaire_maximum = trouver_maximum_prioritaire_alpha({partie: obj.entree_floue for partie, obj in objectifs_fuzzifies.items()}, ordre_priorite, alpha=0.3)
    
    print(f"Objectif musculaire maximum : {objectif_musculaire_maximum}")
    
    partition_objectif_musculaire = ["perte", "inchangé", "gain modéré", "gros gain"]
    degres_objectif_musculaire = [1 if cat == objectif_musculaire_maximum else 0 for cat in partition_objectif_musculaire]
    objectif_musculaire_max_fuzz = Entree_floue("Objectif Musculaire Maximum", partition_objectif_musculaire)
    objectif_musculaire_max_fuzz.entree_floue = degres_objectif_musculaire
    
    # Affichage de la fuzzification
    print("\nFuzzification de l'objectif musculaire maximum :")
    for label, degre in objectif_musculaire_max_fuzz.entree_floue.items():
        print(f"{label}: {degre:.2f}")
    
    '''
    ###
    print("\nVeuillez entrer votre objectif de masse grasse (entre 0.07 et 0.25) :")
    d["Objectif Masse Grasse"].entree_nette = float(input("Objectif de masse grasse : "))
    ###
    '''
    sortie_objectif_mg = Entree_floue("Objectif MG", list(d["Objectif Masse Grasse"].entree_floue.keys()), list(d["Objectif Masse Grasse"].entree_floue.values()))
    sortie_objectif_mg.normaliser()
    
    print("\nSystème Objectifs terminé avec succès.")
    print(f"Objectif musculaire maximum : {objectif_musculaire_maximum}")
    print(f"Objectif de masse grasse fuzzifié : {d['Objectif Masse Grasse'].entree_floue}")
    
    
    
    ###################################################### NUTRITION 1 #########################################################################
    
    print("Bienvenue dans le système flou Nutrition.")
    
    print("\nPartition normalisée pour 'Objectif de Masse Grasse' :")
    for label, degre in sortie_objectif_mg.entree_floue.items():
        print(f"{label}: {degre:.2f}")
        
    print("\nPartition normalisée pour 'Conditions Biologiques' :")
    for label, degre in condition_biologique.entree_floue.items():
        print(f"{label}: {degre:.2f}")
    
    SIF_nutrition_1 = SystemeFlou([condition_biologique, objectif_musculaire_max_fuzz], d["regles SIF Nutrition 1"])
    sortie_nutrition_1 = SIF_nutrition_1.sortie_floue_non_normalisée("Nutrition Provisoire")
    
    if "DANGER" in sortie_nutrition_1.entree_floue and sortie_nutrition_1.entree_floue["DANGER"] > 0:
        print("\nDANGER FIN DU SYSTEME")
        exit("Vous êtes très peu musclé et vous demandez une perte musculaire. Nous ne pouvons pas vous fournir de programme adapté.")
    
    
    
    ###################################################### NUTRITION 2 #########################################################################
    
    SIF_nutrition_2 = SystemeFlou([sortie_nutrition_1, sortie_objectif_mg], d["regles SIF Nutrition 2"])
    sortie_nutrition_2 = SIF_nutrition_2.sortie_floue_normalisée("Apports caloriques")
    
    if "DANGER" in sortie_nutrition_2.entree_floue and sortie_nutrition_2.entree_floue["DANGER"] > 0:
        print("\nDANGER FIN DU SYSTEME")
        exit("Vous êtes très peu musclé et vous demandez une perte musculaire. Nous ne pouvons pas vous fournir de programme adapté.")
    
    valeurs_nutrition = [-500, -400, -200, 0, 200, 400]
    augmentation_apports_caloriques = sortie_nutrition_2.defuzzification(valeurs_nutrition, 1)
    d["Apports caloriques"].entree_nette = calories_de_maintenance + augmentation_apports_caloriques
    
    # Affichage du résultat défuzzifié
    print("\nSystème Nutrition terminé avec succès.")
    print(f"\nRésultat défuzzifié (calories à ajouter/soustraire) : {augmentation_apports_caloriques:.2f} kcal")
    print("Votre programme nutritionnel a été créé avec succès")
    
    
    
    ################################################################### DOPAGE #############################################################
    
    print("Bienvenue dans le système flou Dopage.")
    
    '''
    ###
    dopage_OK = int(input("Entrez si vous prenez du dopage (0 pour non, 1 pour oui): "))
    dopage_OK = bool(dopage_OK)
    repondance = int(input("Entrez votre répondance au dopage (entier entre 0 et 3): "))
    ###
    '''
    
    if dopage_OK:
        d["Impact du dopage"].entree_nette = repondance
    else:
        d["Impact du dopage"] = Entree_floue("Impact du dopage", ["Aucun impact", "Peu répondant", "Répondant", "Très répondant"], [1, 0, 0, 0])
    
    print("\nSystème Dopage terminé avec succès.")
    
    
    
    ################################################## INTENSITE NECESSAIRE 1 + 2 #############################################################
    
    print("Bienvenue dans le système flou Intensité nécessaire.")
    
    '''
    ###
    print("Veuillez évaluer votre atout génétique pour ce qui est du gain musculaire pour les 4 parties du corps (entre -0.3 et 1): ")
    genetiques = {
        "Bras": int(input("Génétique pour les bras (0 à 4) : ")),
        "Jambes": int(input("Génétique pour les jambes (0 à 4) : ")),
        "Dos": int(input("Génétique pour le dos (0 à 4) : ")),
        "Torse": int(input("Génétique pour le torse (0 à 4) : "))
    }
    ###
    '''

    intensites_nec_1, intensites_nec_2 = {}, {}
    for partie_du_corps, valeur in genetiques.items():
        d["Génétique"].entree_nette = valeur
        sys = SystemeFlou([d["Génétique"], objectifs_fuzzifies[partie_du_corps]], d["regles SIF Intensité Nécessaire 1"])
        intensites_nec_1[partie_du_corps] = sys.sortie_floue_non_normalisée("Intensité nécessaire intermédiaire")
        sys2 = SystemeFlou([d["Impact du dopage"], intensites_nec_1[partie_du_corps]], d["regles SIF Intensité Nécessaire 2"])
        intensites_nec_2[partie_du_corps] = sys2.sortie_floue_normalisée("Intensité nécessaire")
    
    print("\nSystème Intensité necessaire terminé avec succès.")
    for partie, intensite_fuzz in intensites_nec_2.items():
        print(f"{partie}: {intensite_fuzz.entree_floue}")
            
            
    
    ############################################################# INTENSITE POSSIBLE #############################################################
    
    print("Bienvenue dans le système flou Intensité possible.")
    
    '''
    ###
    print("Veuillez évaluer votre santé pour les 4 parties du corps (0 étant santé idéale et 1 étant gravement blessé/handicap): ")
    santes = {
        "Bras": float(input("Santé pour les bras (entre 0 et 1) : ")),
        "Jambes": float(input("Santé pour les jambes (entre 0 et 1) : ")),
        "Dos": float(input("Santé pour le dos (entre 0 et 1) : ")),
        "Torse": float(input("Santé pour le torse (entre 0 et 1) : "))
    }
    ###
    '''

    intensites_pos = {}
    for partie_du_corps, valeur in santes.items():
        d["Santé"].entree_nette = valeur
        sys = SystemeFlou([d["Santé"], d["Apports caloriques"]], d["regles SIF Intensité Possible"])
        intensites_pos[partie_du_corps] = sys.sortie_floue_normalisée("Intensité possible")
    
    print("\nSystème Intensité possible terminé avec succès.")
    for partie, intensite_fuzz in intensites_pos.items():
        print(f"{partie}: {intensite_fuzz.entree_floue}")

    
    ############################################################# INTENSITE REELLE #############################################################
    
    print("Bienvenue dans le système flou Intensité réelle.")
    
    # Calcul des intensités réelles
    intensites_reelles = {}
    for partie_du_corps in intensites_nec_2:
        # Utilisation de la méthode defuzzification directement
        barycentre_pos = intensites_pos[partie_du_corps].defuzzification([20, 25, 30, 15, 5, 10], gamma=1)
        barycentre_nec = intensites_nec_2[partie_du_corps].defuzzification([5, 10, 15, 20, 25, 30], gamma=1)
        # Intensité réelle
        intensites_reelles[partie_du_corps] = min(barycentre_pos, barycentre_nec)

    # Affichage des intensités réelles
    print("\nIntensités réelles pour chaque partie du corps :")
    for partie, intensite in intensites_reelles.items():
        print(f"{partie} : {intensite:.2f}")
    
    # Génération du programme d'entraînement
    programme = generer_programme(intensites_reelles)
    print("\n \n \n \nVoici votre programme d'entraînement personnalisé :")
    for jour, seance in enumerate(programme, start=1):
        print(f"Jour {jour} : {seance}")

    print("\nVoici votre programme nutritionnel : ")
    print(f"{d['Apports caloriques'].entree_nette} kcal quotidiennement")
    print(calculer_macronutriments(d["Apports caloriques"].entree_nette)) #donne les macronutriments que doit suivre l'utilisateur


        
if __name__ == '__main__':
    main()