# Analyse des premières expériences de compression LLM

Cette analyse synthétise les résultats obtenus avec GPT-2 et SmolLM-135M sur des textes de différentes longueurs.

## 1. L'Effet d'Amortissement (Amortization)
C'est la découverte la plus positive.
- **Observation :** Le taux de compression grimpe de **0.78x** (expansion) pour un texte court à **~5.0x** pour un texte long.
- **Explication :** Le prompt (le "code" compressé) a un coût fixe en bits. Plus le texte cible est long, plus ce coût est "étalé" sur un grand nombre de caractères originaux.
- **Leçon :** La méthode LLM est taillée pour la compression de **gros volumes** de données, là où le surcoût initial du prompt devient négligeable.

## 2. Le Plafond d'Accuracy (The 55% Wall)
C'est le point le plus surprenant : l'accuracy ne monte pas avec la taille du prompt sur les textes longs.
- **Observation :** Sur le texte de 245 tokens, passer de 5 à 100 tokens de prompt ne change quasiment pas l'accuracy (~54-55%).
- **Explication (L'Intelligence vs Le Code) :** 
    - Les 55% représentent ce que le modèle arrive à prédire grâce à sa **connaissance innée** (le *prior*). 
    - Les 45% restants sont des informations spécifiques (noms propres, chiffres, dates) que le prompt prefixe n'arrive pas à "forcer" une fois qu'il est transformé en mots discrets.
- **Leçon :** Ajouter des mots au début (Préfixe) a un rendement décroissant très rapide. Le modèle "oublie" le prompt au profit du texte récent.

## 3. Le "Quantization Gap" (Soft vs Discrete)
- **Observation :** En mode "Soft" (embeddings optimisés), la perte tombe à presque 0. En mode "Discret" (mots réels), l'accuracy chute.
- **Leçon :** Notre goulot d'étranglement n'est pas la capacité du modèle à comprendre, mais notre capacité à **traduire un concept mathématique optimal en mots du dictionnaire**. Nous perdons environ 50% d'efficacité lors de cette traduction.

## 4. Comparaison des Modèles
- **Observation :** SmolLM-135M surpasse GPT-2 non pas en vitesse, mais en **qualité de prédiction**.
- **Leçon :** Plus le modèle est "intelligent" (basse perplexité), moins on a besoin de "bits de correction". La compression LLM dépend directement de la qualité du modèle pré-entraîné.

## Conclusion pour la suite
Pour briser le mur des 55% d'accuracy, nous devons :
1. Arrêter de mettre tout le prompt au début (essayer des tokens latents intercalés).
2. Passer à une optimisation qui cherche directement des tokens discrets (Discrete Search) au lieu de projeter des embeddings continus.
