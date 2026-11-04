# EF-07 — Réponses et dossiers sourcés

## Fonctionnement

[`generate_response.py`](../Data%20Analysis/generate_response.py) produit un JSON structuré pour `answer`, `compare` et `build_dossier`. Il charge et revalide les conversations, les unités, les relations revues et les sujets de mémoire EF-05. Les alias de sujet, annotés dans [`atlas_memory_topics.jsonl`](../Data/Curated/atlas_memory_topics.jsonl), déterminent le périmètre ; un nom de projet seul ouvre tout le projet uniquement pour un dossier ou une comparaison. Une question portant seulement sur « Atlas » demande quel sujet examiner. Un projet inconnu ne retombe pas sur Atlas.

Le routage part d'EF-06. Pour une question explicite sur un projet connu, EF-07 récupère une erreur de routage `missing_subject` ou une prédiction d'action erronée lorsque les règles reconnaissent clairement une question et que le sujet annoté est identifiable. Les demandes hypothétiques et les actions restent sans effet ; EF-08 traitera les brouillons.

La sortie contient `intent`, `status`, `project_id`, `topic_ids`, `sections`, `references` et `missing`. Chaque affirmation d'une section a un `text`, un `kind` et des `source_unit_ids`. Les affirmations de type `unit` reprennent **exactement** le `statement` de l'unité. Les affirmations de type `reviewed_relation` ne sont produites que pour des liens `supersedes` ou `contradicts` validés, avec les deux unités et la source de preuve du lien. Les références donnent la source, la date, les tours, le statut, l'extrait exact, le locuteur et le responsable éventuel. `validate_response` rejette une affirmation inventée ou une référence modifiée.

Le dossier regroupe `facts`, `chronology`, `decisions`, `open_points`, `responsibles` et `changes`. `facts` ne retient une affirmation `asserted` que si un lien `supports` revu la relie à une unité confirmée ; les faits contradictoires n'y figurent pas. Une proposition remplacée figure dans la chronologie avec le rôle `obsolete`, tandis que seule une décision confirmée et non contestée est présentée comme courante. Les risques, questions non résolues et contradictions figurent dans `open_points`. Les sections absentes sont listées dans `missing`. Une réponse sur un sujet conflictuel présente les deux affirmations et indique qu'on ne peut pas trancher.

## Reproduction

```powershell
.venv-gpu\Scripts\python.exe "Data Analysis\generate_response.py" "Quelle est la décision actuelle sur le budget Atlas ?"
.venv-gpu\Scripts\python.exe "Data Analysis\generate_response.py" "Construis un dossier Atlas." --format markdown
.venv-gpu\Scripts\python.exe "Data Analysis\generate_response.py" "Compare les deux réunions sur la livraison Atlas."
.venv-gpu\Scripts\python.exe "Data Analysis\evaluate_responses.py" --output Data/Curated/response_eval_metrics.json
.venv-gpu\Scripts\python.exe -m unittest discover -s tests -v
```

## Évaluation et limites

[`response_eval_cases.jsonl`](../Data/Curated/response_eval_cases.jsonl) contient douze scénarios Atlas rédigés pour le développement, avec 32 unités attendues. Le rapport [`response_eval_metrics.json`](../Data/Curated/response_eval_metrics.json) donne la couverture des unités attendues, l'exactitude du statut (`ok`, `uncertain`, `clarify`, `not_found`), la présence des relations attendues et l'intégrité structurelle des citations. Ces scénarios ont servi à corriger le générateur : leurs résultats ne constituent **pas** un test indépendant ni une mesure de généralisation. Le compteur d'affirmations sans preuve vérifie la provenance et la correspondance exacte avec les unités ou relations revues ; il ne remplace pas un jugement humain sur la justesse sémantique des annotations. Il reste à annoter des réponses attendues sur d'autres projets et conversations, puis à mesurer manuellement la pertinence et l'exhaustivité des dossiers.

La résolution des sujets utilise des alias annotés et ne découvre pas de nouveaux sujets automatiquement. Les réponses courtes choisissent une unité pertinente ou la décision courante ; elles ne composent pas encore une synthèse libre de plusieurs sources. Les comparaisons présentent la chronologie et uniquement les changements confirmés par des relations revues. Une information absente est signalée, sans inférence depuis une date récente ou une simple similarité.
