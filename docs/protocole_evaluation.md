# Protocole d'évaluation — recherche de preuves QMSum

Version 1 consignée le 28 octobre 2026, **après** les premières mesures de validation et **avant** tout score de retrieval sur le split test. Ce protocole décrit l'expérience déjà réalisée et fixe les règles des comparaisons suivantes. Il ne transforme pas rétrospectivement les résultats de validation en test indépendant.

## 1. Question expérimentale et périmètre

Mesurer si l'adaptation contrastive de `intfloat/multilingual-e5-small` améliore la recherche des passages de réunion pertinents pour une question spécifique QMSum. Le benchmark actuel recherche **dans la réunion dont provient la question**. Il ne mesure pas encore la recherche parmi toutes les réunions, la détection des décisions ou les réponses de l'agent.

Les [fiches de données](datasets.md), le [schéma d'annotation](schema_annotation.md), l'[audit reproductible](../Data%20Analysis/audit_week1_data.py), ses [résultats JSON](audit_week1_data.json) et le [contrôle ponctuel de six preuves](controle_annotations.md) complètent ce protocole.

## 2. Données et séparation

| Split QMSum | Réunions | Questions spécifiques | Passages de 10 tours | Usage |
|---|---:|---:|---:|---|
| Train | 162 | 1 095 | 8 906 | Fine-tuning ; 6 734 paires positives obtenues par chevauchement. |
| Validation | 35 | 237 | 2 030 | Choix du modèle, comparaison des méthodes et analyse d'erreurs. |
| Test | 35 | 244 | 2 086 | Évaluation finale après gel des choix. Aucun score du modèle n'a encore été calculé sur ce split dans ce dépôt. |

Le split est celui fourni par QMSum, au niveau de la **réunion**. L'audit ne trouve aucun `meeting_id` ou transcript exactement identique entre ses trois splits, aucun span de question hors limites, aucun trou de couverture dans les passages, et aucune question spécifique sans passage positif. Les fichiers exacts sont identifiés par SHA-256 dans [`audit_week1_data.json`](audit_week1_data.json). Les questions générales sont exclues : elles n'ont pas de spans localisant la preuve.

**Règle contre la fuite entre corpus :** QMSum réutilise 137 réunions AMI et 59 ICSI présentes dans les exports locaux. Les 20 réunions AMI et 9 ICSI de la validation QMSum, ainsi que les 20 AMI et 9 ICSI du test QMSum, ne doivent jamais entrer dans un train complémentaire AMI/ICSI. Un mélange de corpus doit affecter le split par identité de réunion commune. MeetingBank n'est pas utilisé dans ce benchmark : son split actuel est par ligne/segment et 791 réunions inférées apparaissent dans plusieurs splits.

L'audit du test n'a contrôlé que la structure et quelques exemples de questions pour vérifier les spans ; **aucun score test** n'a servi à choisir l'encodeur ou ses paramètres. Ne pas examiner les erreurs test avant de figer les variantes et objectifs.

## 3. Candidats et vérité de référence

[`create_qmsum_10_tours.py`](../Data/Processing/create_qmsum_10_tours.py) forme des passages successifs de dix tours, sans chevauchement, avec `passage_id = meeting_id:start-end`. Pour la question `meeting_id:i`, les candidats sont les passages non vides de cette réunion seulement. Un passage est positif si son intervalle `[start_turn, end_turn]` touche au moins un intervalle `relevant_text_span` QMSum, bornes **inclusives**. Il peut y avoir plusieurs positifs.

Ce gold est une **approximation** : QMSum annote des plages utiles à une réponse résumée, pas la pertinence de chaque bloc de dix tours. Une plage très large marque parfois de nombreux passages positifs ; 242 questions ont plus de dix passages positifs, jusqu'à 59. Un Hit@5 peut donc récompenser un passage qui n'est pas à lui seul une preuve suffisante. L'encodeur tronque à 512 tokens ; 188 passages de validation sur 2 030 dépassent cette longueur. Conserver ces limites dans le rapport et isoler les erreurs concernées.

## 4. Méthodes comparées

Toutes reçoivent les **mêmes questions, candidats et labels** et renvoient les cinq meilleurs passages, y compris si un score lexical vaut zéro. Les égalités lexicales suivent l'ordre des passages dans le fichier.

| Méthode | Réglages fixés |
|---|---|
| TF-IDF | [`TfidfVectorizer`](../Data%20Analysis/evaluate_qmsum_retrieval.py) scikit-learn, paramètres par défaut ; vocabulaire ajusté aux passages de la réunion, cosinus. |
| BM25 | [`bm25_scores`](../Data%20Analysis/bm25_score_qmsum.py), `k1=1.5`, `b=0.75`, tokens Unicode d'au moins deux caractères, index construit sur la réunion. |
| E5 initial | `intfloat/multilingual-e5-small`, révision locale du cache HF `614241f622f53c4eeff9890bdc4f31cfecc418b3`, `query: ` et `passage: `, pooling moyen, vecteurs normalisés, produit scalaire. |
| E5 adapté | Même encodeur, checkpoint local `Models/qmsum-e5-small/checkpoints/checkpoint-842`, même préfixes et métrique. Le SHA-256 de `model.safetensors` est `1882ffc39c626972770c4f4778a12e72db9d0a1f60e40943c52c9689e3f21a46`. |

Le script d'entraînement utilise 6 734 paires question/passage du train, `CachedMultipleNegativesRankingLoss`, 2 époques, batch 16, mini-batch de calcul 2, longueur 512, taux d'apprentissage `2e-5`, warmup de 10 %, et graine 42. Le run achevé compte 842 pas et 838,9 s d'entraînement sur RTX 4060 Laptop GPU. Aucun négatif difficile explicitement annoté n'a encore été ajouté. Le script fixe désormais `seed=42` et `data_seed=42` pour les futurs runs ; la répétition exacte au bit près sur GPU n'est pas garantie.

Environnement utilisé pour la vérification : Windows, Python 3.13.5, `torch 2.14.1+cu130`, `sentence-transformers 6.2.0`, `transformers 5.21.0`, `datasets 5.1.0`, `accelerate 1.15.0`, `scikit-learn 1.9.1`, `numpy 2.5.4`. Les versions et empreintes des données doivent être consignées pour chaque nouveau run.

## 5. Mesures et choix

Les métriques sont moyennées **par question** :

- `Hit@5` : 1 si au moins un des cinq passages est positif, sinon 0.
- `Recall@5` : nombre de passages positifs dans le top 5 divisé par le nombre total de passages positifs de cette question.
- `MRR@5` : inverse du rang du premier passage positif, ou 0 s'il est absent du top 5.
- `nDCG@5` : à ajouter à la comparaison finale, avec pertinence binaire ; `DCG@5 = Σ rel_i / log2(i+1)` pour les rangs `i=1..5`, puis division par le DCG idéal. Consigner aussi le résultat par domaine Academic, Product et Committee.

### Validation déjà observée

| Méthode | Hit@5 | Recall@5 | MRR@5 |
|---|---:|---:|---:|
| TF-IDF | 0,734 | 0,359 | 0,517 |
| BM25 | 0,802 | 0,410 | 0,578 |
| E5 initial | 0,7975 | 0,4112 | 0,5982 |
| E5 adapté | **0,8650** | **0,4694** | **0,6598** |

Sources : [`qmsum_retrieval_validation_comparison.jsonl`](../Data/Processed/qmsum_retrieval_validation_comparison.jsonl) et [`metrics.json`](../Models/qmsum-e5-small/metrics.json). Les deux sorties sont des **résultats de validation**, pas des performances finales généralisables.

**Objectifs consignés avant l'évaluation test :** le modèle adapté doit dépasser E5 initial d'au moins `+0,03` en MRR@5 global et atteindre au moins le Hit@5 de BM25. Une baisse dans un domaine doit être rapportée explicitement même si l'objectif global est atteint. Ces objectifs sont des critères de lecture du test, non des seuils imposés par le sujet.

## 6. Ordre des prochaines expériences

1. Sur train/validation seulement, constituer des négatifs difficiles contrôlant statut, négation, personne, date et nombre. Vérifier qu'ils ne sont pas en réalité des réponses valides.
2. Comparer une variante avec négatifs difficiles à l'E5 adapté actuel, puis une ablation isolant cet ajout. Étudier les échecs et les passages tronqués sur validation. Fixer aussi la règle de fusion hybride BM25+dense et le calcul nDCG avant de regarder les scores test.
3. Geler code, paramètres, données, objectifs et empreintes ; exécuter **une fois** le même benchmark sur les 244 questions test, pour TF-IDF, BM25, E5 initial, E5 adapté et hybride. Publier résultats globaux et par domaine, exemples d'erreurs, latence et mémoire.
4. Si le test révèle un problème, le documenter. Toute modification ultérieure constitue une **nouvelle expérience** et requiert un nouveau test indépendant ; ne pas retuner sur ce test.

## 7. Commandes de reproduction de la validation

Depuis la racine du dépôt, avec les fichiers `Data/Datasets/` présents :

```powershell
.venv-gpu\Scripts\python.exe Data\Processing\data_processing_pipeline.py
.venv-gpu\Scripts\python.exe Data\Processing\create_qmsum_10_tours.py
.venv-gpu\Scripts\python.exe "Data Analysis\audit_week1_data.py"
.venv-gpu\Scripts\python.exe "Data Analysis\evaluate_qmsum_retrieval.py"
.venv-gpu\Scripts\python.exe "Data Analysis\recherche_embeddings.py"
.venv-gpu\Scripts\python.exe "Data Analysis\finetune_qmsum_e5.py" --evaluate-checkpoint "Models\qmsum-e5-small\checkpoints\checkpoint-842"
```

Régénérer `Data/Processed/` modifie potentiellement ses empreintes. Si l'audit après régénération ne correspond pas au [rapport de référence](audit_week1_data.json), consigner la différence avant de comparer les scores. Les scripts `recherche_embeddings.py` et `finetune_qmsum_e5.py` évaluent actuellement la **validation** ; le protocole test ci-dessus reste à implémenter avant l'exécution finale.
