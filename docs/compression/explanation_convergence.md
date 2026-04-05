# Comprendre la Convergence et les Prompts Latents

Ce document explique les mécanismes mathématiques observés lors des expériences de compression LLM.

## 1. Le mécanisme de Convergence (Overfitting par Gradient)
Lorsqu'un log indique `Converged (Loss < 0.001)`, il s'agit d'un processus de **mémorisation forcée**.

- **Processus :** Nous utilisons l'algorithme Adam pour modifier les vecteurs d'embeddings du prompt. Les poids du modèle (SmolLM, GPT-2) restent fixes.
- **Objectif :** Minimiser la Cross-Entropy entre les prédictions du modèle et le texte cible.
- **Résultat :** Le prompt devient une "clé d'activation" spécifique. En passant dans les couches d'attention, ces vecteurs forcent les probabilités de sortie à se concentrer à 99%+ sur les tokens du texte original.
- **Analogie :** C'est comme si on sculptait une clé sur mesure pour ouvrir une serrure spécifique (le texte) dans le labyrinthe des probabilités de l'IA.

## 2. Le paradoxe du Fossé Discret (Quantization Gap)
Une observation courante est : `Loss Soft < 0.001` mais `Accuracy Discrete < 100%`.

- **Soft (Espace Continu) :** Le gradient trouve un point mathématique parfait dans l'espace des embeddings (768 dimensions pour GPT-2).
- **Discrete (Espace de Vocabulaire) :** Nous devons transformer ce point parfait en un "mot" existant dans le dictionnaire. 
- **Le Problème :** Le mot le plus proche est souvent à une distance significative du point optimal. Ce "petit arrondi" suffit à perturber le mécanisme d'attention du modèle, faisant chuter la probabilité du token cible de 99% à 40-60%.
- **Conclusion :** Plus le texte cible est long, plus le modèle est sensible à cette imprécision de projection.

## 3. Pourquoi éviter `n_prompt > target_len` ?
Si le nombre de tokens dans le prompt dépasse la longueur du texte, nous faisons de l'**expansion** et non de la compression.

- **Utilité :** Ce n'est utile que pour mesurer la capacité de mémorisation brute (le "canal de communication") du modèle.
- **Action :** Le script a été mis à jour pour sauter automatiquement ces configurations et économiser du temps de calcul.
