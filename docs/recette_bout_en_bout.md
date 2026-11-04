# Recette des huit scénarios de bout en bout

## Pipeline exécuté

[`semantic_agent.py`](../Data%20Analysis/semantic_agent.py) charge et valide les conversations, unités, relations et sujets annotés (EF-01, EF-04, EF-05). Pour chaque demande, il classe l'intention (EF-06), cherche dans toutes les unités avec EF-02, puis transmet le classement au module de réponse EF-07 ou au simulateur EF-08. Un mail adressé à un rôle passe aussi par la résolution EF-09. Le mode `hybrid` emploie le checkpoint E5 adapté d'EF-03 avec BM25 ; `bm25` fournit une exécution légère sans ce checkpoint. La recherche classe les candidats, mais les statuts et liens revus décident de ce qui peut être présenté comme courant.

La démonstration Atlas comporte deux conversations **synthétiques**, seize unités et six relations revues. Trois tours ajoutés à `Atlas-02` donnent des citations exactes pour le contexte du prototype, le suivi du budget par Camille et l'engagement de Marc à transmettre le dossier le 10 novembre. Ce sont des exemples créés pour la démonstration, pas des témoignages ou comptes rendus réels.

## Attendus figés

[`atlas_acceptance_cases.jsonl`](../Data/Curated/atlas_acceptance_cases.jsonl) fixe pour chaque scénario l'intention, le statut, les unités obligatoires, les sections ou relations attendues et, pour le mail, le destinataire et `executed: false`.

| Scénario | Contrôle déterminant |
|---|---|
| 1. Budget courant | La décision approuvée est citée ; la proposition antérieure n'est pas courante. |
| 2. Date de livraison | Les deux dates restent dans la chronologie ; `supersedes` prouve le remplacement. |
| 3. Deux réunions | `contradicts` pour le prix et `supersedes` pour la livraison sont distingués et sourcés. |
| 4. Dossier à transmettre | Marc et l'échéance du 10 novembre proviennent de l'engagement cité. |
| 5. Dossier Atlas | Contexte, faits, chronologie, décisions, risques, responsables, points ouverts et références sont présents. |
| 6. Mail au responsable du budget | Camille est justifiée par son engagement explicite ; adresse `null`, confirmation requise, `executed: false`. |
| 7. Décisions non précisées | Le système demande une clarification sans proposer d'opération. |
| 8. Phrase hypothétique | Aucune opération n'est proposée ou exécutée. |

[`evaluate_acceptance.py`](../Data%20Analysis/evaluate_acceptance.py) vérifie aussi la provenance de chaque unité retrouvée, affirmation et brouillon. Les [métriques hybrides](../Data/Curated/atlas_acceptance_metrics.json) et les [huit sorties complètes](../Data/Curated/atlas_acceptance_outputs.jsonl) sont conservées pour inspection ; la baseline BM25 a ses propres métriques et sorties. Le rapport fige les empreintes des conversations, unités, relations, sujets, cas attendus et poids E5, ainsi que les paramètres de fusion. Les deux modes passent les huit scénarios sur ces données synthétiques. Cette recette vérifie le parcours fonctionnel, sans mesurer la généralisation à des conversations réelles ou à des formulations indépendantes.

## Reproduction

```powershell
.venv-gpu\Scripts\python.exe "Data Analysis\ingest_semantic_units.py"
.venv-gpu\Scripts\python.exe "Data Analysis\evaluate_acceptance.py" --method hybrid --device cpu
.venv-gpu\Scripts\python.exe "Data Analysis\semantic_agent.py" "Prépare un mail à la responsable du budget Atlas." --method hybrid --device cpu
.venv-gpu\Scripts\python.exe -m unittest discover -s tests
```

Le mode hybride demande le checkpoint local `Models/qmsum-e5-small/checkpoints/checkpoint-842`, ignoré par Git. Pour examiner la même recette sans ce modèle :

```powershell
.venv-gpu\Scripts\python.exe "Data Analysis\evaluate_acceptance.py" --method bm25 --metrics Data\Curated\atlas_acceptance_bm25_metrics.json --outputs Data\Curated\atlas_acceptance_bm25_outputs.jsonl
```

La CLI du mini-agent accepte aussi `--conversations`, `--units`, `--relations` et `--topics` pour charger d'autres sources annotées, après validation des mêmes contrats.
