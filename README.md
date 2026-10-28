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
