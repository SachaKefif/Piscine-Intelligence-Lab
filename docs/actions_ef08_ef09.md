# EF-08 et EF-09 — Brouillons simulés et destinataire

## Parcours

[`semantic_agent.py`](../Data%20Analysis/semantic_agent.py) appelle EF-06, puis EF-07 pour une réponse ou [`simulate_actions.py`](../Data%20Analysis/simulate_actions.py) pour un brouillon. Les opérations candidates sont `email.compose_draft`, `tasks.create_draft` et `calendar.create_draft`. Ce sont des **étiquettes de sortie**, pas des connecteurs. Chaque brouillon contient `intent`, `tool_candidate`, `parameters`, `content`, `context_unit_id`, `source_unit_ids`, `references`, `needs_confirmation: true` et `executed: false`. Une demande ambiguë renvoie `status: clarify`, sans opération candidate ni contenu. `validate_draft` rejette un brouillon exécuté, une citation modifiée ou une référence absente.

Les contenus reprennent une unité confirmée du sujet ; une décision contestée ou des faits contradictoires bloquent le brouillon. La tâche conserve l'action demandée par l'utilisateur mais n'infère pas de personne assignée. L'événement conserve la date explicite ; heure et participants inconnus restent vides. Le mail conserve le nom du destinataire choisi ou résolu et l'adresse reste `null`, faute de répertoire de contacts attesté.

## Destinataire EF-09

[`resolve_recipient.py`](../Data%20Analysis/resolve_recipient.py) distingue :

- `explicit_user` : l'utilisateur a nommé la personne directement après la demande de mail. Ce choix vient de la demande, sans prétendre prouver un rôle dans les sources.
- `resolved` : le rôle mentionné correspond à un `owner` explicite d'une unité `action` ou `commitment` confirmée, dans le sujet annoté, non remplacée, datée et assez récente.
- `clarify` : rôle vague, attribution absente ou périmée, projet inconnu, ou plusieurs responsables possibles. Une personne seulement citée ailleurs dans la phrase ne devient jamais destinataire.

La fenêtre de fraîcheur vaut **90 jours par défaut**, paramétrable dans le résolveur. Son point de référence reproductible est la date de la conversation chargée la plus récente, ou `--as-of` si fourni ; ce n'est pas l'horloge de la machine. Une date inconnue ne prouve pas une attribution récente. Si deux responsables restent possibles sans remplacement revu, le module s'abstient même si l'un est plus récent. `validate_resolution` vérifie les unités, les propriétaires, les dates et les citations. Sur Atlas, Camille a validé le budget mais `owner` est `null` : « responsable du budget » demande donc une précision. Julie est explicitement responsable de la commande des capteurs.

## Reproduction

```powershell
.venv-gpu\Scripts\python.exe "Data Analysis\resolve_recipient.py" "Prépare un mail à la responsable de la commande des capteurs Atlas."
.venv-gpu\Scripts\python.exe "Data Analysis\simulate_actions.py" "Rédige un mail à Camille sur le budget Atlas."
.venv-gpu\Scripts\python.exe "Data Analysis\simulate_actions.py" "Crée une tâche pour envoyer la commande des capteurs Atlas avant le 2026-11-06."
.venv-gpu\Scripts\python.exe "Data Analysis\simulate_actions.py" "Planifie une réunion sur la livraison Atlas le 2026-11-19."
.venv-gpu\Scripts\python.exe "Data Analysis\semantic_agent.py" "Quelle est la décision actuelle sur le budget Atlas ?"
.venv-gpu\Scripts\python.exe "Data Analysis\evaluate_actions.py" --output Data\Curated\action_eval_metrics.json
.venv-gpu\Scripts\python.exe -m unittest discover -s tests
```

## Mesure et limites

[`evaluate_actions.py`](../Data%20Analysis/evaluate_actions.py) mesure Precision@1 des noms choisis, faux destinataires, abstentions justifiées, rappel des cas résolubles et intégrité des preuves. Il contrôle aussi l'intention, le statut, l'unité de contexte et `executed: false` des brouillons. Les jeux Atlas comprennent 11 cas de destinataire et 8 cas d'action. Un second jeu synthétique Boréal, avec deux réunions et un remplacement explicite de responsable, contient 6 cas de destinataire et 5 cas d'action ; ses conversations et annotations sont revalidées par les mêmes contrats EF-01/EF-04/EF-05.

Le [rapport enregistré](../Data/Curated/action_eval_metrics.json) donne zéro faux destinataire et 100 % de cas conformes sur ces petits jeux. Les cas Atlas ont servi au développement ; Boréal est un autre projet synthétique, rédigé par la même équipe. Ces résultats ne démontrent pas la performance sur des formulations humaines indépendantes. La résolution par rôles utilise des mots et alias annotés ; elle peut s'abstenir sur une paraphrase non reconnue. Le système ne possède ni annuaire vérifié ni connecteur externe, donc aucune adresse n'est inventée et aucune action n'est exécutée.
