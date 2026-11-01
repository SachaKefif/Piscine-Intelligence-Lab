# Embedding Lab - Semantic Agent Engine

Projet individuel de la Piscine Makers Intelligence Lab. L'objectif est de concevoir et d'évaluer un moteur de mémoire sémantique fondé sur des embeddings spécialisés, puis de l'utiliser dans un agent qui répond avec des preuves, construit des dossiers et prépare des actions simulées.

## État du dépôt

Le dépôt contient les quatre corpus normalisés, les baselines de recherche QMSum et un encodeur E5 adapté sur le train QMSum. La recherche de passages QMSum a été évaluée sur la validation et sur le test final avec une fusion BM25 + E5 figée avant le test. **EF-01, EF-02, EF-04 et EF-05 sont implémentées et vérifiées sur des démonstrations synthétiques** : EF-04 couvre les six relations, propose un lien pour une nouvelle paire avec abstention, et contrôle la décision courante. EF-05 expose une mémoire chronologique par sujet sur plusieurs conversations, avec remplacements et contradictions sourcés. L'évaluation d'EF-04 sur deux projets synthétiques tenus à l'écart ne prouve pas une généralisation aux réunions réelles. L'agent reste à réaliser.

## Cahier des charges

- [Version éditable](Consignes/cahier_des_charges.md)
- [Version PDF](Consignes/cahier-des-charges.pdf)

## Documentation des données et de l'évaluation

- [Fiches des quatre corpus](docs/datasets.md) : provenance, licence, préparation, effectifs et limites.
- [Schéma d'annotation](docs/schema_annotation.md) : unités, statuts, preuves, relations et contrats JSON.
- [Protocole d'évaluation](docs/protocole_evaluation.md) : splits, paramètres, métriques et règles contre la fuite de données.
- [Rapport final de recherche](docs/rapport_retrieval_final.md) : validation, test, résultats par domaine, erreurs et coûts.
- [Rapport EF-04](docs/rapport_relations_ef04.md) : règles de classification, métriques par classe, confusion et limites.
- [Mémoire EF-05](docs/memoire_ef05.md) : sujets, chronologie, signalements et reproduction.
- [Audit reproductible](docs/audit_week1_data.json) et [contrôle ponctuel des preuves](docs/controle_annotations.md).

## Principes clés

- Comparer une recherche lexicale, un encodeur pré-entraîné et un encodeur adapté par apprentissage contrastif.
- Garder chaque affirmation liée à une source, une date et un extrait vérifiable.
- Distinguer proposition, décision, contradiction et remplacement temporel.
- Préparer des brouillons et propositions d'action sans jamais envoyer de message ni modifier un service externe.
- Évaluer séparément le retrieval, les relations, les intentions et le système complet.

Le cahier des charges détaille le périmètre, les exigences, les critères de recette, les livrables et le calendrier.

## Fine-tuning E5 sur QMSum

Le script [finetune_qmsum_e5.py](Data%20Analysis/finetune_qmsum_e5.py) adapte `intfloat/multilingual-e5-small` à la recherche de passages de réunion. Il construit des paires question/passage à partir de `Data/Processed/qmsum.jsonl` et `qmsum_10_tours.jsonl`. Seul le split `train` sert à l'entraînement ; les 237 questions du split `validation` servent à comparer Hit@5, Recall@5 et MRR@5 avant et après.

```powershell
python -m pip install -U sentence-transformers datasets accelerate
python "Data Analysis/finetune_qmsum_e5.py" --prepare-only
python "Data Analysis/finetune_qmsum_e5.py" --epochs 2 --batch-size 16
```

Le modèle final est le dernier checkpoint dans `Models/qmsum-e5-small/checkpoints/` ; les mesures sont dans `Models/qmsum-e5-small/metrics.json`. Le script ne crée pas une troisième copie du modèle après l'entraînement. La loss contrastive traite les 16 paires par blocs de 2 (`--mini-batch-size`) pour limiter l'utilisation de la mémoire ; sur CPU, l'entraînement complet peut prendre plusieurs heures. Utiliser `--fp16` seulement avec un GPU compatible. Les passages pertinents sont définis par chevauchement des tours annotés dans QMSum ; ces labels sont approximatifs, particulièrement pour les longues discussions.

### GPU NVIDIA sous Windows

Le venv `.venv` de cette machine contient une version CPU de PyTorch. Le venv `.venv-gpu` contient `torch 2.14.1+cu130` et détecte la RTX 4060. Pour entraîner sur GPU, appeler explicitement son Python, même si `.venv` est activé dans PowerShell :

```powershell
.venv-gpu\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.venv-gpu\Scripts\python.exe "Data Analysis\finetune_qmsum_e5.py" --epochs 2 --batch-size 16
```

Après un entraînement terminé, si seule la sauvegarde finale ou l'évaluation a échoué, évaluer directement le dernier checkpoint sans réentraîner :

```powershell
.venv-gpu\Scripts\python.exe "Data Analysis\finetune_qmsum_e5.py" --evaluate-checkpoint "Models\qmsum-e5-small\checkpoints\checkpoint-842"
```

Pour recréer ce venv sur une machine compatible : installer une version CUDA de PyTorch selon le [sélecteur officiel PyTorch](https://pytorch.org/get-started/locally/), puis `sentence-transformers`, `datasets` et `accelerate`. Le script affiche le périphérique utilisé avant de démarrer l'entraînement.

### Expérience avec négatifs difficiles synthétiques

Le fichier [`hard_negative_triplets.jsonl`](Data/Synthetic/hard_negative_triplets.jsonl) contient 80 triplets **fictifs** question / positif / négatif difficile, répartis en huit catégories. Ils ne sont pas des extraits de QMSum. Le script d'expérience part du checkpoint E5 déjà adapté, utilise 64 triplets pour un entraînement court et garde 16 triplets pour un diagnostic séparé. Il mesure aussi la validation QMSum avant et après, sans lire le test ni écraser le modèle de référence.

```powershell
.venv-gpu\Scripts\python.exe "Data Analysis\experiment_hard_negatives_e5.py" --prepare-only
.venv-gpu\Scripts\python.exe "Data Analysis\experiment_hard_negatives_e5.py"
```

Le premier essai (1 époque, batch 8, taux `1e-5`) est enregistré dans [`metrics.json`](Models/qmsum-e5-hardneg-synthetic/metrics.json). L'exactitude sur les 16 triplets gardés à part passe de 0,8125 à 0,9375, mais le MRR@5 QMSum validation passe de 0,6598 à 0,6528 et le Hit@5 de 0,8650 à 0,8608. Ce checkpoint expérimental ne remplace donc pas le modèle de référence. Pour mesurer un gain sur la tâche principale, construire des négatifs difficiles à partir des réunions du **train QMSum**, vérifier qu'aucun n'est une réponse valide, puis comparer sur la validation QMSum avant de figer le modèle.

### Négatifs difficiles issus du train QMSum

Trois scripts préparent un essai sur des extraits réels : [`mine_qmsum_hard_negatives.py`](Data%20Analysis/mine_qmsum_hard_negatives.py) classe les passages de la même réunion avec le checkpoint E5 de référence ; [`review_qmsum_hard_negatives.py`](Data%20Analysis/review_qmsum_hard_negatives.py) crée une file de revue et n'exporte que les exemples explicitement acceptés ; [`train_qmsum_hard_negatives.py`](Data%20Analysis/train_qmsum_hard_negatives.py) entraîne sur ces triplets et mesure avant/après sur les mêmes 237 questions de validation. Aucun script ne lit le split test pour cette expérience.

```powershell
.venv-gpu\Scripts\python.exe "Data Analysis\mine_qmsum_hard_negatives.py"
.venv-gpu\Scripts\python.exe "Data Analysis\review_qmsum_hard_negatives.py" prepare
```

Ouvrir `Data/Curated/qmsum_hard_negative_review_queue.jsonl`. Pour chaque ligne, lire **toute** la question, la réponse de référence, le positif et le négatif. Renseigner `review_status` avec `accepted` ou `rejected`, puis `review_note` avec la raison. Une absence de chevauchement avec la plage QMSum ou un faible recouvrement lexical ne suffisent pas à prouver qu'un passage est sans réponse valable. Si le négatif répond même partiellement, le rejeter. Si le positif ne soutient pas la question, le rejeter aussi.

`prepare` laisse volontairement les 38 lignes à `pending`. La commande `export` refuse de créer un fichier tant que **chaque** ligne n'a pas un statut et une note. Lancer les commandes suivantes une par une : un échec d'`export` signifie que le fichier `_v2.jsonl` n'existe pas encore.

```powershell
.venv-gpu\Scripts\python.exe "Data Analysis\review_qmsum_hard_negatives.py" export --output Data\Curated\qmsum_hard_negatives_reviewed_v2.jsonl
.venv-gpu\Scripts\python.exe "Data Analysis\train_qmsum_hard_negatives.py" --triplets Data\Curated\qmsum_hard_negatives_reviewed_v2.jsonl --output D:\Piscine-Models\qmsum-e5-hardneg-review-v2 --device cpu
```

L'entraînement part de `Models/qmsum-e5-small/checkpoints/checkpoint-842`, préserve ce modèle et écrit un nouveau modèle et `metrics.json` dans le dossier de sortie. Par défaut, il utilise le CPU, des lots de 2 et 256 tokens par champ d'entraînement pour limiter la mémoire ; la validation utilise toujours la limite E5 de 512 tokens. Choisir un **nouveau** dossier de sortie sur un disque disposant d'au moins 1 Go libre. La revue de la file a retenu 25 exemples sur 38 ; les 13 rejets et leurs motifs sont conservés dans `Data/Curated/qmsum_hard_negatives_review.json`. Une revue humaine indépendante reste conseillée avant une diffusion finale des données.

Pour utiliser directement les 25 exemples déjà revus par l'assistant, sans produire le fichier `_v2.jsonl` :

```powershell
.venv-gpu\Scripts\python.exe "Data Analysis\train_qmsum_hard_negatives.py" --triplets Data\Curated\qmsum_hard_negatives_reviewed.jsonl --output D:\Piscine-Models\qmsum-e5-hardneg-assistant-review-25 --device cpu
```

**Essai exploratoire déjà exécuté :** un run a chargé 26 triplets avant la dernière revue. Sur la validation QMSum, Hit@5 est passé de 0,8650 à 0,8565, Recall@5 de 0,4694 à 0,4725 et MRR@5 de 0,6598 à 0,6513. Il n'améliore pas le classement global ; conserver le checkpoint de référence. La revue a ensuite retiré un triplet ambigu et le fichier courant contient 25 lignes. Pendant ce run, l'empreinte du fichier a été calculée après l'entraînement, au moment où le fichier avait déjà changé : elle ne décrit pas exactement les 26 lignes chargées. Le script corrigé copie désormais les triplets dans le dossier de sortie **avant** l'entraînement et calcule l'empreinte de cette copie. Utiliser un nouveau dossier de sortie pour tout nouvel essai.

## EF-01 — Implémentée et vérifiée sur la démo

Le jeu de démonstration contient [deux conversations Atlas datées](Data/Synthetic/atlas_conversations.jsonl) et [treize unités annotées](Data/Curated/atlas_semantic_units.jsonl). Il est **synthétique**, distinct de QMSum. Chaque unité exprime une seule information et conserve un identifiant stable, son type, son statut, sa source, sa date, ses tours et une citation exacte ; locuteur, responsable et échéance restent `null` quand la source ne les établit pas. Le format suit le [contrat existant](docs/schemas/semantic_unit.schema.json). Par exemple, la proposition de 40 000 € du 20 octobre et son approbation du 3 novembre sont deux unités distinctes. Les dates absolues des échéances de la démo sont résolues avec la date explicite de leur réunion.

Depuis la racine du dépôt :

```powershell
.venv-gpu\Scripts\python.exe "Data Analysis\ingest_semantic_units.py"
```

Le script contrôle le schéma, l'unicité des identifiants, les sources, les dates, les bornes de tours, les locuteurs et la présence **exacte** de chaque citation dans la transcription. Il exige au moins deux conversations représentées dans les unités puis écrit `Data/Processed/semantic_units.jsonl` ; ce fichier généré est ignoré par Git. Une vérification automatique de la citation ne prouve pas que la reformulation, le statut ou le responsable sont justes : les [annotations](Data/Curated/atlas_semantic_units.jsonl) ont été relues pour la démo et de nouvelles unités demandent une revue humaine. Les spans de pertinence QMSum ne doivent pas être convertis automatiquement en décisions.

Pour utiliser d'autres transcriptions normalisées, produire un JSONL de conversations au format `dataset`, `meeting_id`, `source_date`, `segments`, puis un JSONL d'unités conforme au schéma. Les indices `start_turn` et `end_turn` désignent les positions inclusives dans `segments`. Lancer ensuite :

```powershell
.venv-gpu\Scripts\python.exe "Data Analysis\ingest_semantic_units.py" --conversations CHEMIN\conversations.jsonl --units CHEMIN\units_annotees.jsonl --output Data\Processed\semantic_units.jsonl
```

Une date inconnue doit rester `null`. Ne pas déduire de chronologie entre réunions QMSum sans date fiable.

## EF-02 — Implémentée et vérifiée sur la démo

Le [moteur de recherche](Data%20Analysis/search_semantic_units.py) classe **toutes** les unités du fichier ingéré, sans imposer la réunion d'origine. `bm25` emploie la recherche lexicale existante ; `e5` emploie le checkpoint E5 adapté ; `hybrid` fusionne leurs rangs par RRF avec `k=60` et un poids E5 de `0,75`. Ce poids vient du benchmark de **passages QMSum** : il sert ici de point de départ et n'a pas été optimisé pour les unités. Chaque résultat contient score, identifiant, énoncé, statut, source, date et citation. Les scores BM25, cosinus E5 et RRF ont des échelles différentes ; comparer les rangs ou les métriques, pas leurs valeurs brutes. `e5` et `hybrid` demandent le checkpoint local `Models/qmsum-e5-small/checkpoints/checkpoint-842`, ignoré par Git ; `--model` permet de désigner une autre copie locale. BM25 fonctionne sans ce modèle.

```powershell
.venv-gpu\Scripts\python.exe "Data Analysis\search_semantic_units.py" --query "Quel budget a été approuvé pour le prototype Atlas ?" --method hybrid
.venv-gpu\Scripts\python.exe "Data Analysis\search_semantic_units.py" --query "Quelles propositions et décisions concernent le budget Atlas ?" --method bm25
```

Pour comparer les trois méthodes sur les **mêmes treize unités** et les [neuf questions annotées avec `relevant_unit_ids`](Data/Curated/atlas_unit_queries.jsonl) :

```powershell
.venv-gpu\Scripts\python.exe "Data Analysis\search_semantic_units.py" --questions Data\Curated\atlas_unit_queries.jsonl --output Data\Curated\atlas_unit_retrieval_metrics.json --device cpu
```

Le [diagnostic enregistré](Data/Curated/atlas_unit_retrieval_metrics.json) donne à `k=5` un MRR de `0,9444` pour BM25, `0,7500` pour E5 et `0,8056` pour l'hybride. Les trois trouvent au moins une unité pertinente pour les neuf questions ; la requête « propositions et décisions » peut renvoyer des sources issues des deux conversations. Pour la question sur le budget **approuvé**, l'hybride place pourtant la proposition avant l'approbation : le statut doit donc rester visible et la recherche seule ne décide pas de l'état courant. **Ces neuf questions synthétiques sont une démonstration fonctionnelle, pas un test indépendant sur corpus réel.** Elles ne remplacent pas le benchmark QMSum et ne justifient pas un nouveau choix de modèle.

Les critères fonctionnels EF-01/EF-02 sont couverts par cette démonstration : ingestion de deux conversations avec provenance, puis recherche classée avec score, source et preuve et comparaison des trois méthodes sur les mêmes questions. L'extraction des unités reste une étape d'annotation ; le cahier des charges n'impose pas qu'elle soit automatique. **Approfondissement possible, sans en faire une condition de clôture EF-01/EF-02 :** annoter plusieurs réunions réelles, créer des questions avec des identifiants d'unités pertinents et mesurer la recherche sur des réunions tenues à l'écart.

## EF-04 — Relations et temporalité sur la démo Atlas

Les [six relations annotées](Data/Curated/atlas_relations.jsonl) couvrent `supports`, `contradicts`, `supersedes`, `creates_action`, `answers` et `unrelated`. Chaque lien positif porte une citation exacte et sa réunion source, conformément au [schéma des relations](docs/schemas/relation.schema.json). La date du 19 novembre **remplace** explicitement la date du 12 novembre seulement envisagée ; deux affirmations opposées sur la confirmation du prix de Nova sont marquées comme **contradictoires**, sans inventer de résolution. `unrelated` est une annotation explicite d'une paire examinée, jamais la valeur automatique d'une paire absente.

Après l'ingestion EF-01 ci-dessus, le [module relations et temporalité](Data%20Analysis/relations_temporality.py) valide les liens, permet de consulter une relation et renvoie une décision courante par sujet :

```powershell
.venv-gpu\Scripts\python.exe "Data Analysis\relations_temporality.py" validate
.venv-gpu\Scripts\python.exe "Data Analysis\relations_temporality.py" classify --from-id Synthetic:Atlas-02:1-1:0 --to-id Synthetic:Atlas-01:1-1:0
.venv-gpu\Scripts\python.exe "Data Analysis\relations_temporality.py" predict --from-id Synthetic:Atlas-02:7-7:0 --to-id Synthetic:Atlas-01:3-3:0
.venv-gpu\Scripts\python.exe "Data Analysis\relations_temporality.py" current atlas_budget
.venv-gpu\Scripts\python.exe "Data Analysis\relations_temporality.py" current atlas_delivery
.venv-gpu\Scripts\python.exe "Data Analysis\evaluate_relations.py"
```

Le module vérifie les unités citées, la présence de la citation dans la réunion indiquée, les types des liens `answers` et `creates_action`, ainsi que l'ordre chronologique d'un `supersedes`. `classify` consulte uniquement les annotations validées : une paire non annotée renvoie `relation: null`. `predict` propose une relation sans la valider. Les [sujets de décision](Data/Curated/atlas_decision_topics.jsonl) regroupent explicitement les unités pertinentes. Une décision est courante si elle est `confirmed`, appuyée par sa preuve, non remplacée et sans contradiction annotée. Plusieurs décisions confirmées sans remplacement prouvé donnent `ambiguous`, même si l'une est plus récente. Une simple proposition n'est jamais présentée comme décision courante.

Pour un autre corpus, fournir `--conversations`, `--units`, `--relations` et `--topics` **avant** la sous-commande. `predict` applique un classifieur **à règles**, sans entraînement, à une paire non annotée ; il cite la preuve et s'abstient si elle est insuffisante. Ses sorties restent des propositions `reviewed: false` et ne modifient pas les relations validées. Sur 12 paires synthétiques tenues à l'écart (deux par classe), le macro-F1 est `0,9444`, avec 11/12 paires qualifiées ; une relation `supersedes` a été laissée en abstention. La [matrice de confusion et les limites](docs/rapport_relations_ef04.md) sont documentées.

## EF-05 — Mémoire entre conversations

Le [module de mémoire](Data%20Analysis/semantic_memory.py) utilise les unités et relations validées d'EF-04 ainsi que [quatre sujets Atlas annotés](Data/Curated/atlas_memory_topics.jsonl), chacun présent dans les deux conversations. Il renvoie la chronologie avec citations, les liens pertinents, les unités obsolètes et les contradictions non résolues. La proposition de livraison du 12 novembre est obsolète car le lien `supersedes` prouve son remplacement par la décision du 19 novembre. Les affirmations contradictoires sur le prix Nova restent toutes deux visibles ; aucune n'est choisie sans preuve de résolution.

```powershell
.venv-gpu\Scripts\python.exe "Data Analysis\semantic_memory.py" validate
.venv-gpu\Scripts\python.exe "Data Analysis\semantic_memory.py" list
.venv-gpu\Scripts\python.exe "Data Analysis\semantic_memory.py" show atlas_delivery
.venv-gpu\Scripts\python.exe "Data Analysis\semantic_memory.py" show atlas_sensor_price
```

Pour un autre corpus, fournir `--conversations`, `--units`, `--relations` et `--topics` **avant** la sous-commande. Les sujets sont annotés explicitement ; le module ne découvre pas automatiquement de nouveaux sujets ni ne valide les propositions `reviewed: false` d'EF-04. La [documentation EF-05](docs/memoire_ef05.md) précise le contrat et les limites.
