"""Registre des familles de modèles candidats (dev_notes/model_selection_small_vocab_reasoning.md).

Un seul point de vérité pour l'identifiant HuggingFace de référence de chaque
famille, pour que tout script de préparation de données / entraînement
accepte "--tokenizer lfm2" au lieu de recopier l'ID HF complet partout. Un nom
inconnu (déjà un chemin HF complet, ex. "gpt2" ou "org/repo") est renvoyé tel
quel -- rétrocompatible avec l'usage existant de --tokenizer.

Ordre d'expérimentation retenu (2026-09-16) : LFM2 d'abord (petit vocab +
petite taille), puis OLMo (ladder 1B->32B, vocab stable), puis Qwen en
dernier (comparaison, malgré son vocab exclu du filtre initial -- voir
model_selection_small_vocab_reasoning.md section 1).
"""

MODEL_FAMILIES = {
    "lfm2": "LiquidAI/LFM2-350M",  # plus petite variante -- tokenizer identique (vocab 64,400) à tout le reste de la famille v1/v2.5-1.2B
    "olmo": "allenai/OLMo-2-0425-1B",  # plus petite variante de la famille OLMo-2
    "qwen": "Qwen/Qwen3-0.6B",
}


def resolve_model_name(name: str) -> str:
    """"lfm2"/"olmo"/"qwen" (insensible à la casse) -> ID HuggingFace de
    référence pour cette famille. Toute autre valeur (déjà un chemin HF
    complet) est renvoyée inchangée."""
    return MODEL_FAMILIES.get(name.lower(), name)
