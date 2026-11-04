# Contrôle de transfert sur questions QMSum externes

Le [corpus QMSum](https://github.com/Yale-LILY/QMSum) fournit des questions,
réponses et plages de tours pertinentes annotées par ses auteurs. Nous avons
figé, dans le commit `cf28fd7`, quinze questions directes originales des
réunions `education_13` et `education_17` du split test Committee. Ces deux
conversations n'avaient pas servi aux contrôles précédents de l'agent. Le
[manifeste](../Data/Curated/qmsum_external_agent_manifest.json) vérifie les
SHA-256 des transcriptions, unités, demandes, sujets, relations et poids avant
exécution. Aucun réglage du modèle ou des règles n'a été fait à partir du
résultat. Les sources ne donnent pas de date de réunion fiable.

Chaque question et réponse humaine a été reprise sans traduction. Pour rendre
la source lisible par l'agent, un script a converti la réponse annotée en unité
de type `answer`, avec la plage de transcription originale comme preuve. Les
quinze unités sont recherchées ensemble ; les alias de sujet ne donnent pas
l'unité attendue. C'est un contrôle de transfert de la chaîne complète et de
la recherche sur des **réponses déjà annotées**, pas une évaluation de
l'extraction automatique de faits bruts. Les requêtes sont en anglais alors que
le classifieur EF-06 est entraîné pour le français.

| Mesure sur 15 demandes | BM25 | Hybride |
| --- | ---: | ---: |
| Cas entièrement conformes | 0/15 | 0/15 |
| Intention `answer` correcte | 0/15 | 0/15 |
| Statut `ok` correct | 0/15 | 0/15 |
| Réponse annotée en rang 1 de la recherche | 12/15 | 12/15 |
| Réponse annotée dans les 5 premiers résultats | 15/15 | 15/15 |
| Faits cités dans la réponse finale | 0/15 | 0/15 |
| Clarifications injustifiées | 15/15 | 15/15 |

Dans les deux modes, treize clarifications viennent de `missing_subject` et
deux de `missing_action` : les questions anglaises sur des personnes ont été
prises pour des tâches. Les trois échecs de rang 1 concernent
`education_13_10`, `education_17_2` et `education_17_10`. Les 15 réponses
finales ne citent aucune unité ; le `citation_relevance_rate` est donc `null`
et ne signifie pas une pertinence de 100 %.
Le rappel des clarifications attendues est `null`, car aucune des quinze
questions annotées ne demandait de clarification.

Les [métriques BM25](../Data/Curated/qmsum_external_agent_bm25_metrics.json),
[métriques hybrides](../Data/Curated/qmsum_external_agent_hybrid_metrics.json) et
[sorties hybrides](../Data/Curated/qmsum_external_agent_hybrid_outputs.jsonl)
conservent les erreurs et les preuves récupérées. Pour rejouer :

```powershell
.venv\Scripts\python.exe "Data Analysis\evaluate_agent_holdout.py" --method hybrid --device cpu --raw-dir Data\Datasets\QMSum\data\Committee\test --meetings education_13 education_17 --conversations Data\Processed\qmsum_external_agent_conversations.jsonl --units Data\Curated\qmsum_external_agent_units.jsonl --relations Data\Curated\qmsum_external_agent_relations.jsonl --topics Data\Curated\qmsum_external_agent_topics.jsonl --cases Data\Curated\qmsum_external_agent_cases.jsonl --expected-cases 15 --frozen-manifest Data\Curated\qmsum_external_agent_manifest.json --metrics Data\Processed\qmsum_external_replay.json --outputs Data\Processed\qmsum_external_replay.jsonl
```

**Limite restante :** QMSum ne contient ni demandes d'e-mail, de tâche ou
d'événement écrites par des utilisateurs pour cet agent, ni étiquettes de
destinataire. La résolution positive et les refus de destinataire ne sont
donc pas évalués indépendamment ici. Il faut un petit lot de demandes
françaises rédigées par d'autres personnes et une annotation séparée des
destinataires, décisions et faits attendus, figé avant le prochain test.
