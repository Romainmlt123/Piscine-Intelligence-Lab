# Choix des Modèles LLM par Matière

## Objectif

Utiliser des modèles LLM spécialisés pour chaque matière afin d'améliorer la qualité des réponses.

---

## Modèles initialement choisis (HuggingFace)

| Matière | Modèle HuggingFace | Raison du choix |
|---------|-------------------|-----------------|
| **Mathématiques** | `nvidia/AceMath-1.5B-Instruct` | Modèle NVIDIA spécialisé dans le raisonnement mathématique (Chain-of-Thought). Excellentes performances sur les benchmarks mathématiques. |
| **Physique** | `qingy2024/Benchmaxx-Llama-3.2-1B-Instruct` | Basé sur Llama 3.2, optimisé pour les benchmarks. Bon équilibre entre performance et taille. |
| **Anglais** | `Qwen/Qwen2.5-1.5B-Instruct` | Modèle multilingue performant, bon pour l'enseignement des langues. |

---

## Modèles retenus (Ollama)

> **Décision** : Nous avons opté pour les modèles équivalents disponibles sur Ollama car :
> 1. **Quantification optimisée** : Les modèles Ollama sont pré-quantifiés (Q4_K_M, Q5_K_M) pour une exécution rapide sur CPU
> 2. **Facilité d'installation** : Une seule commande `ollama pull` vs téléchargement GGUF + conversion
> 3. **Compatibilité garantie** : Testés et validés avec l'API Ollama
> 4. **Mêmes architectures de base** : Les modèles Ollama sont basés sur les mêmes familles (Qwen, Llama)

| Matière | Modèle Ollama | Base commune avec HF |
|---------|--------------|---------------------|
| **Mathématiques** | `qwen2-math:1.5b` | Modèle Qwen spécialisé math (même famille qu'AceMath) |
| **Physique** | `llama3.2:1b` | Même architecture Llama 3.2 |
| **Anglais** | `qwen2.5:1.5b` | Modèle Qwen2.5 original |

---

## Commandes d'installation

```bash
# Télécharger les modèles spécialisés
ollama pull qwen2-math:1.5b      # Pour les mathématiques (spécialisé)
ollama pull llama3.2:1b          # Pour la physique
ollama pull qwen2.5:1.5b         # Pour l'anglais
```

---

## Références

- [AceMath-1.5B-Instruct sur HuggingFace](https://huggingface.co/nvidia/AceMath-1.5B-Instruct)
- [Benchmaxx-Llama-3.2-1B-Instruct sur HuggingFace](https://huggingface.co/qingy2024/Benchmaxx-Llama-3.2-1B-Instruct)
- [Qwen2.5-Math sur Ollama](https://ollama.com/library/qwen2.5-math)
- [Llama 3.2 sur Ollama](https://ollama.com/library/llama3.2)
