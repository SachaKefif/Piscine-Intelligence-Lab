# EF-06 — Intentions et paramètres

## Contrat

`parse_intent.py` reçoit une demande en français et renvoie `intent`, `subject`, `person`, `due_date`, `action` et `clarification_reason`. L'intention appartient à `answer`, `compare`, `build_dossier`, `draft_email`, `create_task`, `schedule_event` ou `clarify`. Chaque paramètre absent ou non reconnu vaut `null`. Les dates explicites avec année sont normalisées en `YYYY-MM-DD` ; une date relative sans date de référence reste `null`. L'action est une courte expression à l'infinitif, normalisée en minuscules sans accents.

La classification utilise maintenant un modèle [TF-IDF de groupes de caractères + LinearSVC](../Data%20Analysis/train_intents.py), choisi sur la validation parmi trois candidats. Les paramètres sont extraits séparément, puis une [politique de contrôle](../Data%20Analysis/parse_intent.py) peut remplacer l'intention prédite par `clarify`. L'ancienne baseline à règles reste disponible avec `--rules`. Le sujet reconnu est un thème (`budget`, `livraison`, `planning`, `capteurs`, `approvisionnement`) suivi du nom d'un projet. `person` signifie seulement qu'un nom est présent dans la demande. Pour un e-mail, le destinataire doit être nommé explicitement près de la demande de message. Le module ne vérifie ni l'identité ni l'adresse d'une personne dans les sources : cette vérification appartient à EF-09.

`clarify` est renvoyé lorsque le modèle juge la demande ambiguë, lorsqu'elle est hypothétique, ou lorsqu'il manque un sujet, un destinataire nommé, une action de tâche ou une date d'événement selon l'intention. `clarification_reason` précise le motif. Le module ne crée ni brouillon ni action externe.

## Reproduction

```powershell
.venv-gpu\Scripts\python.exe "Data Analysis\generate_intent_data.py"
.venv-gpu\Scripts\python.exe "Data Analysis\train_intents.py"
.venv-gpu\Scripts\python.exe "Data Analysis\parse_intent.py" "Rédige un mail à Camille sur le budget Atlas."
.venv-gpu\Scripts\python.exe -m unittest discover -s tests
.venv-gpu\Scripts\python.exe "Data Analysis\evaluate_intents.py" --eval "Data\Curated\intent_fresh_holdout.jsonl" --output "Data\Curated\intent_fresh_eval_metrics.json"
.venv-gpu\Scripts\python.exe "Data Analysis\evaluate_intents.py" --rules --eval "Data\Curated\intent_fresh_holdout.jsonl" --output "Data\Curated\intent_fresh_rules_metrics.json"
```

Le [générateur reproductible](../Data%20Analysis/generate_intent_data.py) conserve les exemples initiaux comme graines et combine des familles de formulations avec des sujets, projets, personnes, dates et actions. Les JSONL contiennent maintenant **424 demandes d'entraînement**, **125 de validation** et **124 d'évaluation** : [train](../Data/Curated/intent_train.jsonl), [validation](../Data/Curated/intent_validation.jsonl), [évaluation](../Data/Curated/intent_eval_holdout.jsonl). Chaque jeu couvre les sept intentions. Les identifiants, textes et projets sont distincts entre jeux ; les familles de formulations générées sont également séparées. Les 15, 24 et 23 exemples initiaux restent conservés dans les fichiers `intent_seed_*.jsonl` pour permettre la régénération.

Le [rapport d'entraînement](../Data/Curated/intent_training_metrics.json) consigne les empreintes des jeux, les trois candidats et la sélection sur le macro-F1 de validation. Le modèle choisi atteint **0,975 de macro-F1** sur 125 demandes de validation ; cette valeur sert uniquement au choix du modèle.

Un [nouveau jeu de 56 demandes](../Data/Curated/intent_fresh_holdout.jsonl), rédigées individuellement pour six projets absents des autres jeux, a été évalué après fixation du modèle et des contrôles. Il comprend huit demandes par intention et n'a pas servi à l'entraînement ou à la sélection. Ce sont néanmoins des phrases synthétiques rédigées dans le cadre du projet, pas des demandes recueillies auprès d'utilisateurs indépendants.

| Méthode sur les 56 demandes | Macro-F1 | Intention et paramètres exacts | Clarifications attendues | Clarifications abusives |
| --- | ---: | ---: | ---: | ---: |
| [Règles seules](../Data/Curated/intent_fresh_rules_metrics.json) | 0,591 | 0,554 | 8/8 | 22/48 |
| [Modèle + contrôles](../Data/Curated/intent_fresh_eval_metrics.json) | **0,930** | **0,911** | **8/8** | **3/48** |

Le rapport du modèle détaille les cinq écarts (quatre intentions et une extraction d'action), la matrice de confusion, les scores par classe et par paramètre, y compris sur les seuls paramètres présents et les seules demandes correctement routées. La précision de `clarify` est de 0,727 malgré son rappel de 1,00. Le [jeu antérieur de 124 demandes](../Data/Curated/intent_eval_holdout.jsonl) avait déjà été inspecté pendant le développement ; ses résultats restent descriptifs et séparés.

## Limites

Les 424 lignes d'entraînement proviennent d'environ dix familles de formulations par intention croisées avec plusieurs projets ; leur nombre ne représente donc pas autant de formulations indépendantes. Les dates relatives et les mois sans année demandent une clarification. `subject` est une étiquette textuelle et n'est pas reliée automatiquement aux `topic_id` de la mémoire EF-05. Une personne mentionnée peut ne pas être le responsable ni le destinataire réel. Le jeu de 56 demandes est petit et synthétique : avant d'utiliser EF-06 sur des demandes ouvertes, il faudra recueillir des formulations réelles, les annoter indépendamment et refaire un test final sans les consulter pour ajuster le système.
