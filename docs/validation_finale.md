# Vérification finale des livrables

État vérifié le 13 novembre 2026 sur la copie de travail et un worktree propre du commit de livraison. Le périmètre est celui des sept [livrables du cahier des charges](../Consignes/cahier_des_charges.md), avec actions simulées et limites expérimentales déclarées.

| Livrable | Preuve livrée |
|---|---|
| 1. Dépôt exécutable et documenté | [README](../README.md), [dépendances](../requirements.txt), CLI [`semantic_agent.py`](../Data%20Analysis/semantic_agent.py), 84 tests automatisés réussis. |
| 2. Datasets, licences et splits | [Fiches des corpus](datasets.md), [protocole de séparation](protocole_evaluation.md), [schéma d'annotation](schema_annotation.md). |
| 3. Baselines, encodeur adapté et benchmarks | [Rapport retrieval](rapport_retrieval_final.md), [fine-tuning](../Data%20Analysis/finetune_qmsum_e5.py), [ablation](../Data%20Analysis/train_qmsum_hard_negatives.py), [obtention du checkpoint](checkpoint_release.md). |
| 4. Recherche, relations, intentions et contrats | [`semantic_memory.py`](../Data%20Analysis/semantic_memory.py), [`predict_relations.py`](../Data%20Analysis/predict_relations.py), [`parse_intent.py`](../Data%20Analysis/parse_intent.py), [contrats JSON](schema_annotation.md). |
| 5. Mini-agent, dossiers et actions simulées | [`semantic_agent.py`](../Data%20Analysis/semantic_agent.py), [`generate_response.py`](../Data%20Analysis/generate_response.py), [`simulate_actions.py`](../Data%20Analysis/simulate_actions.py), [`resolve_recipient.py`](../Data%20Analysis/resolve_recipient.py). |
| 6. Rapport comparatif | [Rapport expérimental](rapport_experimental_final.md) et [contrôles de transfert](extension_transfert.md) : métriques, ablation, erreurs, latence, mémoire, limites. |
| 7. Démonstration multi-conversations | [Huit scénarios Atlas et sorties](recette_bout_en_bout.md), recette rejouée à 8/8 en BM25 et 8/8 en hybride. |

## Vérifications exécutées

- `python -m unittest discover -s tests` : **84/84 tests réussis**.
- `evaluate_acceptance.py` : **8/8** dans chacun des modes BM25 et hybride.
- `evaluate_agent_holdout.py` sur `ES2004a/b` : **11/12** en BM25, diagnostic après correction.
- `evaluate_massive_recipients.py` : **17/17** destinataires nommés et **49/49** abstentions appropriées.
- `evaluate_agent_holdout.py` sur le jeu figé `Bro004/Bro019` : **11/12** en BM25 et hybride ; un autre jeu figé donnait **4/10**, signe d'une généralisation instable.
- `evaluate_independent_ami_review.py` : `pending_human_review` pour **33** paires de décisions et **14** actions ; aucun score humain inventé.

Dans le worktree propre, les **84 tests**, la recette BM25 **8/8**, le jeu QMSum figé `Bro004/Bro019` **11/12** et la vérification des empreintes AMI passent. Les 43 artefacts figés nouvellement suivis ont été comparés octet par octet à leur version indexée dans Git. Les XML AMI sont hachés après normalisation CRLF vers LF pour éviter qu'un checkout Windows change leur empreinte logique.

## Limite à ne pas confondre avec une fonction manquante

Le [protocole AMI](protocole_annotation_independante_ami.md) rend possible une mesure indépendante de contradiction, remplacement et attribution sur de nouvelles réunions. Les quatre formulaires restent vierges. Deux personnes doivent les remplir séparément avant le calcul d'une précision ou d'un rappel sur ces tâches. Le sujet demande de documenter la qualité et les limites de l'évaluation ; il ne fixe pas de seuil de réussite ni n'exige de recueillir des jugements humains externes pour déclarer la recette fonctionnelle achevée.
