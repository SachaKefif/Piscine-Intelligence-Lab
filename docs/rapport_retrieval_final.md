# Évaluation finale de la recherche QMSum

**Date :** 1er novembre 2026. **Statut :** réglage choisi sur validation, puis une évaluation sur le test QMSum. Ce rapport concerne la recherche de passages **dans la réunion de la question**, pas la recherche entre réunions ni la qualité des réponses de l'agent.

## Protocole et choix figé

Le script [`benchmark_qmsum_final.py`](../Data%20Analysis/benchmark_qmsum_final.py) compare TF-IDF, BM25, E5 initial, E5 adapté (`checkpoint-842`) et une fusion de BM25 avec E5 adapté. Tous utilisent les mêmes passages de dix tours, les mêmes questions spécifiques et un passage positif dès qu'il chevauche un `relevant_text_span` QMSum. Les métriques sont moyennées par question, avec `k=5` et une pertinence binaire pour nDCG@5.

Sur les **237 questions de validation**, trois poids E5 ont été comparés dans une *reciprocal rank fusion* : `0,25`, `0,50` et `0,75`, avec constante de rang `60`. Le poids `0,75` a donné le meilleur MRR@5 (`0,6694`) et a été figé **avant** le test, ainsi que le code, les empreintes des données et du modèle. Les réglages et résultats de validation sont dans [`frozen_config.json`](../Models/qmsum-retrieval-benchmark/frozen_config.json) et [`validation_summary.json`](../Models/qmsum-retrieval-benchmark/validation_summary.json). La règle est `score = 0,25/(60 + rang_BM25) + 0,75/(60 + rang_E5)`, avec rangs calculés dans la réunion concernée. Les égalités suivent l'ordre des passages dans le fichier.

## Résultats sur le test

Le test comprend **35 réunions, 244 questions et 2 086 passages**. Les [métriques structurées](../Models/qmsum-retrieval-benchmark/test_summary.json) et les [classements par question](../Models/qmsum-retrieval-benchmark/test_details.jsonl) sont conservés séparément de la validation.

| Méthode | Hit@5 | Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|
| TF-IDF | 0,8156 | 0,3882 | 0,5995 | 0,4336 |
| BM25 | 0,8320 | 0,4058 | 0,6314 | 0,4641 |
| E5 initial | 0,8607 | 0,4218 | 0,6356 | 0,4790 |
| E5 adapté | 0,8566 | **0,4443** | 0,6449 | 0,5081 |
| **Hybride BM25 + E5 adapté** | **0,8648** | 0,4414 | **0,6804** | **0,5203** |

L'hybride, choisi sur validation, gagne `+0,0355` MRR@5 par rapport à E5 adapté et `+0,0448` par rapport à E5 initial sur ce test. Un rééchantillonnage descriptif de 5 000 tirages par réunion (35 groupes, graine 42) donne un intervalle à 95 % de `[+0,0068 ; +0,0651]` pour la différence hybride moins E5 adapté. Cet intervalle est postérieur au gel ; il n'a servi à aucun choix de paramètre.

### Résultats par domaine

| Domaine | Méthode | Questions | Hit@5 | Recall@5 | MRR@5 | nDCG@5 |
|---|---|---:|---:|---:|---:|---:|
| Academic | TF-IDF | 49 | 0,6531 | 0,2266 | 0,5296 | 0,2977 |
| Academic | BM25 | 49 | 0,7347 | 0,2650 | 0,5449 | 0,3262 |
| Academic | E5 initial | 49 | 0,7959 | 0,3239 | 0,5694 | 0,3707 |
| Academic | E5 adapté | 49 | 0,8163 | 0,3521 | 0,5500 | 0,3980 |
| Academic | Hybride | 49 | 0,7959 | 0,3442 | **0,5772** | **0,4038** |
| Product | TF-IDF | 129 | 0,8295 | 0,2788 | 0,5544 | 0,3741 |
| Product | BM25 | 129 | 0,8295 | 0,2863 | 0,5922 | 0,3979 |
| Product | E5 initial | 129 | **0,8915** | 0,3400 | 0,6389 | 0,4627 |
| Product | E5 adapté | 129 | 0,8605 | **0,3691** | 0,6658 | **0,5068** |
| Product | Hybride | 129 | **0,8915** | 0,3595 | **0,6943** | 0,5046 |
| Committee | TF-IDF | 66 | **0,9091** | 0,7219 | 0,7396 | 0,6506 |
| Committee | BM25 | 66 | **0,9091** | **0,7437** | **0,7720** | **0,6958** |
| Committee | E5 initial | 66 | 0,8485 | 0,6543 | 0,6783 | 0,5914 |
| Committee | E5 adapté | 66 | 0,8788 | 0,6597 | 0,6747 | 0,5924 |
| Committee | Hybride | 66 | 0,8636 | 0,6736 | 0,7298 | 0,6374 |

L'hybride dépasse E5 adapté en MRR@5 dans les trois domaines du test. **BM25 seul reste meilleur sur Committee**, où son MRR@5 atteint `0,7720` contre `0,7298` pour l'hybride. E5 adapté perd en MRR@5 face à E5 initial sur Academic (`0,5500` contre `0,5694`) et Committee (`0,6747` contre `0,6783`), malgré son gain global.

### Objectifs annoncés avant le test

- **Gain de MRR@5 d'E5 adapté sur E5 initial ≥ `0,03` : non atteint.** Le gain observé est `0,6449 - 0,6356 = +0,0093`.
- **Hit@5 d'E5 adapté au moins égal à BM25 : atteint.** `0,8566` contre `0,8320`.

## Erreurs et limites des labels

- **Fusion utile — `ES2004c:10` (Product) :** sur la réduction des coûts avec reconnaissance vocale, E5 adapté ne trouve aucun passage annoté dans son top 5. BM25 place `ES2004c:430-439` au rang 2 ; l'hybride le ramène au rang 4. Le passage explique que la réutilisation de pièces déjà en stock peut réduire le coût.
- **Fusion nuisible — `Bmr023:5` (Academic) :** sur les changements du *segmenter*, E5 adapté place `Bmr023:130-139`, annoté pertinent, au rang 3. La fusion le sort du top 5, car BM25 favorise d'autres passages de la réunion.
- **Gold approximatif — `Bed003:3` (Academic) :** le top de l'hybride `Bed003:730-739` traite bien de nœuds cachés et d'une « middle layer », mais QMSum n'annote que `Bed003:140-149` pour cette question. L'échec compté par la métrique pourrait donc inclure une preuve partielle valable hors plage annotée ; il faut lire le passage avant de conclure à une erreur sémantique.

Sur les 244 questions, l'hybride trouve un passage annoté dans 7 cas où E5 adapté échoue, et le perd dans 5 cas où E5 adapté réussit. Les deux échouent selon les labels dans 28 cas. Un contrôle du tokenizer trouve **168 passages test sur 2 086** au-delà de 512 tokens. **86 questions** ont au moins un passage positif tronqué. Le Hit@5 hybride est `0,8372` pour ces 86 questions et `0,8797` pour les autres ; cette différence descriptive ne démontre pas que la troncature cause l'erreur.

## Temps, mémoire et reproduction

Pour le test entier (244 questions, 2 086 passages), le script a mesuré `0,43 s` pour TF-IDF, `1,01 s` pour BM25, `12,61 s` pour E5 initial et `12,15 s` pour E5 adapté. Les temps denses incluent le chargement du modèle et l'encodage par lots. La fusion réutilise les classements BM25 et E5 adapté ; son temps en ligne par requête n'est pas mesuré séparément. Le pic de mémoire allouée par PyTorch sur GPU est d'environ **606 Mio** pour chaque E5 ; la mémoire CPU totale n'a pas été mesurée.

Les empreintes SHA-256 des fichiers préparés, du code de benchmark et des poids adaptés sont dans [`frozen_config.json`](../Models/qmsum-retrieval-benchmark/frozen_config.json). Le script refuse de remplacer les résultats test et vérifie ces empreintes avant cette évaluation. Les limites importantes sont la recherche limitée à la réunion d'origine, les labels QMSum dérivés par chevauchement, la troncature à 512 tokens et l'absence d'une mesure de latence en ligne. Aucun réglage n'a été modifié après la lecture du test.
