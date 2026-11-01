# EF-04 — relations et temporalité

## Fonctionnement livré

Le module `relations_temporality.py` valide les unités et les liens annotés, puis expose trois opérations : `classify` consulte un lien revu, `predict` propose un lien pour une paire nouvelle, et `current` retourne la décision courante d'un sujet. Les six classes sont `supports`, `contradicts`, `supersedes`, `creates_action`, `answers` et `unrelated`.

`predict_relations.py` est un **classifieur déterministe à règles**, sans apprentissage. Il examine les types, statuts, dates, polarité, nombres, indices explicites de remplacement ou de causalité et recouvrement de termes après retrait du nom de projet. Il retourne une citation exacte d'une unité source pour tout lien positif proposé. Les propositions portent `reviewed: false` ; une paire ambiguë reçoit `relation: null`. Une prédiction `unrelated` reste elle aussi une proposition, pas une annotation validée. Le calcul de décision courante n'utilise que les liens revus et validés.

Pour un sujet, une décision courante doit avoir le type `decision`, le statut `confirmed`, une preuve, ne pas être remplacée par un lien `supersedes` valide et ne pas avoir de contradiction annotée non résolue. Deux décisions confirmées concurrentes donnent `ambiguous` ; une proposition seule donne `no_confirmed_decision`. La validation d'un remplacement impose un ordre de dates prouvable, ou un ordre de tours dans une même réunion datée.

## Contrôle expérimental

Le jeu [`relation_eval_holdout.json`](../Data/Curated/relation_eval_holdout.json) contient deux projets synthétiques (Boréal et Cèdre), distincts d'Atlas, avec **12 paires annotées : deux par classe**. Les règles ont été écrites sur la démo Atlas ; l'erreur observée sur Cèdre a été conservée, sans ajuster les seuils au jeu de contrôle. [`relation_eval_metrics.json`](../Data/Curated/relation_eval_metrics.json) contient l'empreinte SHA-256 du jeu, les prédictions et la matrice de confusion complète. `evaluate_relations.py` vérifie aussi que chaque preuve proposée est une citation exacte de la transcription.

| Classe | Précision | Rappel | F1 | Effectif |
|---|---:|---:|---:|---:|
| `supports` | 1,00 | 1,00 | 1,00 | 2 |
| `contradicts` | 1,00 | 1,00 | 1,00 | 2 |
| `supersedes` | 1,00 | 0,50 | 0,67 | 2 |
| `creates_action` | 1,00 | 1,00 | 1,00 | 2 |
| `answers` | 1,00 | 1,00 | 1,00 | 2 |
| `unrelated` | 1,00 | 1,00 | 1,00 | 2 |

**Macro-F1 : 0,9444 ; couverture : 11/12 ; exactitude avec abstention : 11/12.** Dans la matrice de confusion, la seule erreur est `supersedes → abstain` sur la livraison Cèdre : le texte exprime un remplacement mais le recouvrement lexical n'atteint pas le seuil conservateur. Aucune autre classe n'est confondue sur ce petit jeu.

Ces résultats mesurent seulement un contrôle fonctionnel synthétique, avec deux exemples par classe. Ils **ne démontrent pas** une performance sur des réunions réelles ni la qualité d'un classifieur entraîné. Une évaluation de généralisation demanderait un corpus réel de paires revues, un split par réunion ou projet et une seconde annotation humaine. La méthode à règles peut manquer des paraphrases et des contradictions sans négation explicite. Elle s'abstient plutôt que d'inventer un lien dans ces cas.

## Reproduction

Depuis la racine du dépôt :

```powershell
.venv-gpu\Scripts\python.exe "Data Analysis\ingest_semantic_units.py"
.venv-gpu\Scripts\python.exe "Data Analysis\relations_temporality.py" validate
.venv-gpu\Scripts\python.exe "Data Analysis\relations_temporality.py" predict --from-id Synthetic:Atlas-02:7-7:0 --to-id Synthetic:Atlas-01:3-3:0
.venv-gpu\Scripts\python.exe "Data Analysis\relations_temporality.py" current atlas_delivery
.venv-gpu\Scripts\python.exe "Data Analysis\evaluate_relations.py"
.venv-gpu\Scripts\python.exe -m unittest discover -s tests -v
```
