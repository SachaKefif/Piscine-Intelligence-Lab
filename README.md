# Embedding Lab - Semantic Agent Engine

Projet individuel de la Piscine Makers Intelligence Lab. L'objectif est de concevoir et d'évaluer un moteur de mémoire sémantique fondé sur des embeddings spécialisés, puis de l'utiliser dans un agent qui répond avec des preuves, construit des dossiers et prépare des actions simulées.

## État du dépôt

Le dépôt contient les quatre corpus normalisés, les baselines de recherche QMSum et un encodeur E5 adapté sur le train QMSum. La validation retrieval est mesurée ; l'évaluation test finale, les relations, l'agent et la démonstration restent à réaliser.

## Cahier des charges

- [Version éditable](Consignes/cahier_des_charges.md)
- [Version PDF](Consignes/cahier-des-charges.pdf)

## Documentation des données et de l'évaluation

- [Fiches des quatre corpus](docs/datasets.md) : provenance, licence, préparation, effectifs et limites.
- [Schéma d'annotation](docs/schema_annotation.md) : unités, statuts, preuves, relations et contrats JSON.
- [Protocole d'évaluation](docs/protocole_evaluation.md) : splits, paramètres, métriques et règles contre la fuite de données.
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
