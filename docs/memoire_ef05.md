# EF-05 — Mémoire inter-conversations

## Contrat et fonctionnement

La mémoire lit les conversations, les unités sémantiques, les relations revues et un fichier JSONL de sujets. Chaque sujet possède `topic_id`, `project_id` et `unit_ids`, une liste d'identifiants d'unités du même projet. Les sujets peuvent se chevaucher. Le fichier de démonstration [`atlas_memory_topics.jsonl`](../Data/Curated/atlas_memory_topics.jsonl) regroupe budget, livraison, prix des capteurs et approvisionnement ; chacun couvre les deux réunions Atlas. L'appartenance au sujet est annotée, jamais déduite d'un score d'embedding.

`semantic_memory.py show` renvoie :

- `history` : unités triées par date et tour, avec source, date, citation, statut, locuteur, responsable et échéance. Une date inconnue est placée à la fin, sans prétendre déterminer sa position chronologique.
- `relations` : liens revus touchant le sujet. Les unités situées hors du sujet mais citées par un lien sont exposées dans `related_units_outside_topic`. Les liens `unrelated` externes sont ignorés car ils n'apportent pas de contexte au sujet.
- `obsolete_unit_ids` et `superseded_by` : remplacement prouvé par un lien `supersedes` validé. Une date plus récente, une similarité ou une simple proposition approuvée ne suffisent pas. `not_superseded` signifie uniquement qu'aucun remplacement revu n'est connu ; cela ne garantit pas qu'un fait soit actuel.
- `contradictions` : affirmations opposées avec la preuve de leur lien. Elles restent `unresolved` dans cette version ; le module n'en choisit pas une sur la seule date ou l'ordre des tours. Une résolution éventuelle demande une nouvelle annotation et une revue humaine.
- `current_decision` : résultat conservateur du module EF-04 pour les décisions confirmées du sujet. Les faits et les engagements ne deviennent pas des décisions par cette opération.
- `review_needed_unit_ids` : unités déjà annotées `superseded` mais sans lien de remplacement validé ; elles demandent une vérification.

Toutes les preuves sont revalidées contre les transcriptions à chaque chargement. Les relations proposées par `predict` ne figurent pas dans la mémoire tant qu'elles ne sont pas revues et ajoutées au fichier de relations.

## Démonstration

```powershell
.venv-gpu\Scripts\python.exe "Data Analysis\ingest_semantic_units.py"
.venv-gpu\Scripts\python.exe "Data Analysis\semantic_memory.py" validate
.venv-gpu\Scripts\python.exe "Data Analysis\semantic_memory.py" list
.venv-gpu\Scripts\python.exe "Data Analysis\semantic_memory.py" show atlas_budget
.venv-gpu\Scripts\python.exe "Data Analysis\semantic_memory.py" show atlas_delivery
.venv-gpu\Scripts\python.exe "Data Analysis\semantic_memory.py" show atlas_sensor_price
.venv-gpu\Scripts\python.exe -m unittest discover -s tests -v
```

Sur Atlas, `validate` trouve 16 unités, 6 relations et 4 sujets couvrant chacun deux conversations. Pour `atlas_delivery`, la date envisagée du 12 novembre apparaît avant la décision du 19 novembre et est signalée obsolète par le lien `supersedes`. Pour `atlas_sensor_price`, les deux affirmations incompatibles sur la confirmation des 8 000 € restent sourcées et non résolues. Le budget approuvé est la décision courante du sujet `atlas_budget`, mais la proposition antérieure n'est pas étiquetée obsolète sans lien de remplacement explicite. Trois unités synthétiques supplémentaires rendent explicites le contexte du prototype, le suivi du budget par Camille et l'engagement de Marc à transmettre le dossier ; elles conservent leurs citations exactes dans la deuxième conversation.

## Limites

Ce contrôle fonctionnel repose sur deux conversations synthétiques d'un même projet, avec des sujets et des liens annotés. Il ne mesure pas la qualité de découverte de sujets ou de liens sur des réunions réelles. Une contradiction non résolue exige une vérification humaine ou une nouvelle relation probante. Une unité marquée `not_superseded` peut être devenue obsolète dans une conversation non ingérée ou non annotée.
