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
    "lfm2": "LiquidAI/LFM2-350M",  # plus petite variante -- tokenizer identique (vocab 64,400) UNIQUEMENT à
        # la famille v1 (LFM2-350M/700M/1.2B), verifie directement le 2026-09-20 (dictionnaires identiques,
        # meme tokenisation). NE PAS confondre avec "LFM2.5-1.2B-Thinking" (serie 2.5) : tokenizer DIFFERENT
        # (64402, dictionnaire different, meme sur un texte simple) malgre le nom proche -- utiliser ce
        # Teacher-la pour du KD contre un modele tokenise "lfm2" desalignerait silencieusement les indices
        # Top-K. Pour un Teacher plus gros que la reference garanti-aligne : LFM2-1.2B (v1), pas la variante
        # "Thinking".
    "olmo": "allenai/OLMo-2-0425-1B",  # plus petite variante de la famille OLMo-2 -- tokenizer (100,278)
        # verifie directement identique a OLMo-2-1124-7B et OLMo-2-1124-13B le 2026-09-20 (dictionnaires
        # identiques) -- ces tailles restent des Teachers surs pour du KD aligne sur ce tokenizer.
    "qwen": "Qwen/Qwen3-0.6B",
    "qwen_big": "Qwen/Qwen3.8-27B-FP8",  # NOT the same tokenizer family as "qwen" -- verifie
        # directement le 2026-09-20 (dev_notes/experiments/distillation.md) : "qwen" (Qwen3-0.6B)
        # a vocab_size=151643/len=151669, ce Teacher a vocab_size=248044/len=248077, ids differents
        # sur une phrase test -- incompatibilite totale, pas un simple ecart de tokens speciaux
        # comme lfm2/lfm2-thinking. C'est le Teacher deja utilise pour la distillation Qwen de ce
        # projet (§13.1 de la spec) ; tout script KD visant ce Teacher doit passer explicitement
        # "--tokenizer qwen_big", jamais l'alias "qwen" par defaut.
    "qwen35": "Qwen/Qwen3.5-0.8B",  # 2026-09-21 (model-design): PETIT modele de la MEME famille de
        # vocab que "qwen_big" -- Qwen 3.5 garde vocab_size=248,320 constant de 0.8B a 397B-A17B
        # (verifie sur 0.8B et 9B, dev_notes/model_selection_small_vocab_reasoning.md:26), qui
        # coincide avec le vocab reel de "qwen_big" (248,077/248,320, meme checkpoint Qwen3.5
        # utilise en mode texte). Resout le mismatch qwen/qwen_big SANS telecharger un nouveau
        # Teacher : utiliser "--tokenizer qwen35" cote etudiant pour rester aligne sur "qwen_big"
        # cote Teacher, exactement comme lfm2/lfm2-thinking ou olmo le sont deja en interne.
}


def resolve_model_name(name: str) -> str:
    """"lfm2"/"olmo"/"qwen" (insensible à la casse) -> ID HuggingFace de
    référence pour cette famille. Toute autre valeur (déjà un chemin HF
    complet) est renvoyée inchangée."""
    return MODEL_FAMILIES.get(name.lower(), name)
