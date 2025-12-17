# Script Vidéo - Tuteur IA

## 📹 Format
- **Durée** : 3-5 minutes
- **Style** : Commercial/Pitch
- **Ton** : Enthousiaste mais professionnel

---

## 🎬 INTRO (30 secondes)

### Accroche (10s)
> "Imaginez un tuteur disponible 24/7, qui connaît tout le programme scolaire français, qui vous répond à la voix, et qui ne coûte rien. C'est exactement ce qu'on a créé."

### Contexte (20s)
> "Je m'appelle Romain, et je vais vous présenter **Tuteur IA** : un assistant vocal éducatif basé sur une architecture Multi-Agent RAG. 
> 
> Contrairement à ChatGPT, notre système est 100% local, spécialisé pour l'éducation française, et cite ses sources."

---

## 🎯 PROBLÈME (30 secondes)

### Pain Points
> "Le problème ? Les tuteurs humains coûtent cher. ChatGPT n'est pas fiable pour les cours. Et les étudiants ont besoin de réponses vérifiables.
>
> Notre solution : un assistant qui consulte directement les manuels scolaires officiels pour vous répondre."

---

## 💡 DÉMO LIVE (90 secondes)

### Démonstration 1 : Question de physique (30s)
> "Regardez. Je lance le serveur... [montrer le terminal]
> 
> J'ouvre l'interface... [montrer le nouveau design ChatGPT]
> 
> Je clique sur le micro et je demande : 'C'est quoi la loi d'Ohm ?'
> 
> [Attendre la réponse]
> 
> Vous voyez ? La réponse arrive en quelques secondes, avec le badge PHYSICS qui indique quel agent a répondu, et les sources RAG en dessous."

### Démonstration 2 : Question de maths (30s)
> "Essayons une question de maths : 'Comment calculer le périmètre d'un carré ?'
>
> [Attendre la réponse]
>
> L'agent MATH répond avec le modèle qwen2.5:1.5b. Les métriques montrent 1.2 secondes de latence totale."

### Démonstration 3 : Montrer les sources (30s)
> "Le plus important : les sources. Ici on voit que la réponse provient du 'Livre troisième 2017.pdf', chunk 42. 
>
> L'étudiant peut vérifier l'information. C'est ça la différence avec un LLM classique."

---

## 🏗️ ARCHITECTURE (45 secondes)

### Explication technique
> "Côté technique, voici comment ça marche :
>
> 1. **Whisper** transcrit la voix en texte
> 2. Un **routeur** classifie la question (maths, physique, anglais)
> 3. Le **RAG hybride** cherche dans la base de connaissances - on utilise Vector Search + BM25 pour 80% de hit rate
> 4. Le **LLM spécialisé** génère la réponse avec le contexte
> 5. **Piper** synthétise la réponse en audio
>
> Tout est local, tout est open source."

---

## 📊 RÉSULTATS (30 secondes)

### Métriques clés
> "Les résultats :
> - **80% de hit rate** sur notre benchmark de 20 questions
> - **46 millisecondes** de latence RAG
> - **2640 chunks** de contenu indexé
> - **4 modèles LLM** spécialisés
>
> On est passé de 46% à 80% en implémentant le hybrid search. C'est une amélioration de plus de 70%."

---

## 🚧 DIFFICULTÉS (30 secondes)

### Blocages rencontrés
> "On a eu des difficultés :
> - Des PDFs corrompus qui polluaient les résultats - 11 fichiers à supprimer
> - Le port 8000 bloqué par Docker - changé en 8001
> - Le vector search seul qui ne suffisait pas - ajout de BM25
>
> Mais chaque problème nous a permis d'apprendre et d'améliorer le système."

---

## 🔮 CONCLUSION (30 secondes)

### Call to action
> "Tuteur IA prouve qu'on peut créer un assistant éducatif performant, 100% local, avec des technologies open source.
>
> Le code est disponible sur GitHub, branche romain-pipeline-agent-ia.
>
> Merci de votre attention !"

---

## 📝 Notes de tournage

### À montrer à l'écran
1. Terminal avec logs du serveur
2. Interface web ChatGPT-style
3. Démonstration vocale en direct
4. Sources RAG expandées
5. Métriques de latence
6. Diagramme d'architecture

### Tips
- Ne pas accélérer la vidéo pendant les démos
- Montrer les vrais temps de réponse
- Zoomer sur les éléments importants (badge agent, sources)
