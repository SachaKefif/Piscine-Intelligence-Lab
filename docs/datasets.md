# Fiches des données — Embedding Lab

État vérifié le 28 octobre 2026. Les effectifs ci-dessous viennent des fichiers locaux et de [`audit_week1_data.py`](../Data%20Analysis/audit_week1_data.py), pas des tailles annoncées par les publications. Le [rapport JSON](audit_week1_data.json) contient les empreintes SHA-256 des fichiers préparés.

## QMSum — corpus principal pour la recherche

- **Source et provenance :** [dépôt Yale-LILY](https://github.com/Yale-LILY/QMSum) et [article NAACL 2021](https://aclanthology.org/2021.naacl-main.472/). QMSum réunit des réunions de conception AMI, de recherche ICSI et de comités parlementaires. Le commit du dossier local n'est pas conservé ; les empreintes des exports préparés permettent d'identifier ce jeu local.
- **Licence et réutilisation :** le [dépôt affiche une licence MIT](https://github.com/Yale-LILY/QMSum/blob/main/LICENSE). Les transcriptions AMI/ICSI proviennent aussi de corpus sous CC BY 4.0 ; la licence du dépôt QMSum ne suffit donc pas à établir les droits sur chaque source parlementaire d'origine. Conserver les attributions et vérifier les conditions des transcriptions sous-jacentes avant de redistribuer le texte brut.
- **Préparation locale :** [`data_processing_pipeline.py`](../Data/Processing/data_processing_pipeline.py) lit les fichiers JSON par domaine et split, conserve `meeting_id`, `domain`, `segments`, `queries`, `source_files`, puis écrit `Data/Processed/qmsum.jsonl`. [`create_qmsum_10_tours.py`](../Data/Processing/create_qmsum_10_tours.py) construit des blocs consécutifs de dix tours dans `qmsum_10_tours.jsonl`.
- **Effectifs locaux :** 232 réunions : 162 train, 35 validation, 35 test. Questions **spécifiques** : 1 095 / 237 / 244. Passages de dix tours : 8 906 / 2 030 / 2 086. Les questions générales ne servent pas à l'évaluation de retrieval car elles n'ont pas de `relevant_text_span`.
- **Annotations disponibles :** question, réponse de référence et un ou plusieurs intervalles de tours pertinents. L'étiquette passage pertinent est ici **dérivée** par chevauchement avec ces intervalles ; ce n'est pas un jugement humain de pertinence passage par passage. L'audit trouve 0 span invalide et 0 question spécifique sans passage positif.
- **Limites :** les intervalles peuvent couvrir une discussion longue : médiane de 3 passages positifs, maximum de 59, et 242 questions avec plus de 10 passages positifs. Sur la validation, 188 passages sur 2 030 dépassent la limite E5 de 512 tokens. Tous les `source_date` préparés sont nuls.

## AMI — réunions de conception de produit

- **Source et provenance :** [corpus officiel AMI](https://groups.inf.ed.ac.uk/ami/corpus/) ; annotations manuelles locales indiquées comme *release 1.7* dans [`00README_MANUAL.txt`](../Data/Datasets/AMI/00README_MANUAL.txt). Le projet utilise les transcriptions XML `words` et `segments`, pas les signaux multimédias.
- **Licence et réutilisation :** [CC BY 4.0](https://groups.inf.ed.ac.uk/ami/download/), confirmée dans [`LICENCE.txt`](../Data/Datasets/AMI/LICENCE.txt) ; attribution requise, y compris pour des dérivés diffusés.
- **Préparation locale :** reconstruction des segments de parole depuis les références aux mots NITE, avec locuteur, horaires lorsqu'ils existent et chemins des fichiers source. 171 réunions exportées dans `Data/Processed/ami.jsonl`.
- **Annotations et limites :** les fichiers locaux contiennent aussi d'autres couches d'annotation, mais le pipeline actuel n'en extrait pas les décisions ou relations. Le split et la date de réunion sont `null` dans l'export préparé. Les réunions de conception de QMSum proviennent d'AMI : 97 identifiants AMI correspondent au train QMSum, 20 à la validation et 20 au test.

## ICSI — réunions de recherche

- **Source et provenance :** [corpus officiel ICSI](https://groups.inf.ed.ac.uk/ami/icsi/) ; archive locale *Signals and Annotations public release 1.0* selon [`00README.txt`](../Data/Datasets/ICSI/ICSIplus/00README.txt). Le pipeline lit les annotations XML `Words` et `Segments`.
- **Licence et réutilisation :** [CC BY 4.0](https://groups.inf.ed.ac.uk/ami/icsi/), confirmée dans [`LICENCE.txt`](../Data/Datasets/ICSI/ICSIplus/LICENCE.txt). Vérifier séparément les contributions de tiers avant d'en utiliser d'autres couches.
- **Préparation locale :** même reconstruction NITE qu'AMI ; 75 réunions dans `Data/Processed/icsi.jsonl`, avec locuteur, horaires disponibles et fichiers source.
- **Annotations et limites :** aucune étiquette de relation exploitable n'est créée par le pipeline actuel ; split et date `null`. QMSum réutilise 41 identifiants ICSI dans son train, 9 dans sa validation et 9 dans son test.

## MeetingBank — réunions de conseils municipaux

- **Source et provenance :** [fiche Hugging Face utilisée par le dépôt](https://huggingface.co/datasets/huuuyeah/meetingbank) et [article ACL 2023](https://aclanthology.org/2023.acl-long.906/). Le cache local pointe vers la révision `5b4fb6f67f490be93e249d70b732780932d19fe3` ; [`download_meetingbank.py`](../Data/Processing/download_meetingbank.py) fixe désormais cette révision pour les futurs téléchargements.
- **Licence et réutilisation :** la [fiche de cette distribution](https://huggingface.co/datasets/huuuyeah/meetingbank/raw/main/README.md) indique **CC BY-NC-SA 4.0**. Attribution, usage non commercial et partage des adaptations selon les mêmes conditions sont à respecter pour cette distribution.
- **Préparation locale :** les JSONL bruts `train`, `validation`, `test` contiennent `uid`, `id`, `summary`, `transcript`. Le pipeline crée un enregistrement par ligne avec `uid` comme `meeting_id` et la transcription entière comme segment unique. 6 892 lignes préparées : 5 169 / 861 / 862.
- **Annotations et limites :** les résumés existent mais pas de questions avec spans de preuve. `source_date` est `null` dans l'export. Les lignes sont des **instances de segment**, pas des réunions indépendantes. Avec l'identifiant de réunion inféré par `ville_date` dans `uid`, 791 groupes apparaissent dans plusieurs splits. Ces splits ne conviennent donc pas tels quels pour une évaluation séparée par réunion ; il faut regrouper avant tout usage en entraînement/validation/test.

## Règles communes

- Les fichiers originaux restent dans `Data/Datasets/` ; les exports générés sont dans `Data/Processed/`. Chaque ligne préparée possède `source_files`, et l'audit a retrouvé tous ces chemins.
- Une date absente reste `null`. Ne pas déduire automatiquement une date du nom de fichier ou de `uid` sans vérifier sa signification et sa provenance.
- Avant de mélanger les corpus, utiliser **le même groupe de réunion** dans un seul split. En particulier, exclure des données d'entraînement supplémentaires toute réunion AMI/ICSI qui appartient aux splits validation ou test QMSum.
- Les corpus AMI, ICSI et MeetingBank sont préparés mais **n'ont pas servi** au fine-tuning E5 actuel ; seul le train QMSum a servi à ce modèle.
