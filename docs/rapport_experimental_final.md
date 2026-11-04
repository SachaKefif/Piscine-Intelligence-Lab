# Rapport expérimental final — Semantic Agent Engine

## Périmètre et protocoles

Le prototype couvre EF-01 à EF-09 du [cahier des charges](../Consignes/cahier_des_charges.md). Ses sorties sont des réponses structurées et des **brouillons simulés** : aucune connexion à un service externe, aucune action exécutée. Les conversations Atlas et Boréal sont fictives. Deux couples de réunions du test QMSum sont des transcriptions publiques authentiques, tenues hors de l'entraînement de l'encodeur. Le premier contrôle (`ES2004a/b`) possède sept unités et douze demandes ; le second (`ES2011a/b`) six unités et neuf demandes, avec [attendus et empreintes figés avant exécution](../Data/Curated/qmsum_agent_final_manifest.json). Les demandes ont été rédigées pour le projet et ne constituent pas un échantillon d'utilisateurs indépendants. Les dates de ces réunions sont inconnues et restent `null`.

Les jeux, splits, poids, graines, réglages et empreintes des benchmarks de recherche sont fixés dans le [protocole](protocole_evaluation.md) et le [rapport retrieval](rapport_retrieval_final.md). Le test QMSum de recherche n'a servi ni à choisir le poids de fusion ni à entraîner l'encodeur. Le contrôle d'agent sur `ES2004a/b` a d'abord révélé une fausse réponse à une question de responsabilité ; une règle générale exigeant une attribution explicite a été ajoutée. **Les scores finaux de ce contrôle sont donc des diagnostics après correction, et non un test aveugle intact.** Les requêtes et attendus sont restés identiques. Les neuf demandes `ES2011a/b` ont été évaluées ensuite sans modifier les règles ni le modèle à leur lecture.

## Résultats par composant

| Composant | Jeu | Mesure | Résultat | Interprétation |
|---|---|---|---:|---|
| Recherche EF-02/03 | QMSum test, 244 questions, 35 réunions | MRR@5 BM25 / E5 initial / E5 adapté / hybride | 0,6314 / 0,6356 / 0,6449 / **0,6804** | La fusion a été choisie sur validation ; recherche limitée à la réunion de la question. |
| Recherche EF-02/03 | Même test | Hit@5 hybride ; nDCG@5 hybride | **0,8648 ; 0,5203** | Le gain MRR d'E5 adapté seul sur E5 initial est seulement +0,0093, sous l'objectif annoncé de +0,03. |
| Relations EF-04 | 12 paires synthétiques distinctes d'Atlas | Macro-F1 ; paires qualifiées | **0,9444 ; 11/12** | Une relation de remplacement est laissée en abstention. [Matrice détaillée](rapport_relations_ef04.md). |
| Intentions EF-06 | 56 demandes synthétiques rédigées séparément | Macro-F1 ; intention et paramètres exacts | **0,930 ; 0,911** | 8/8 clarifications attendues, mais 3/48 demandes claires sont abusivement clarifiées. [Détails](intentions_ef06.md). |
| Réponses EF-07 | 12 scénarios Atlas de développement | Couverture de 35 faits ; intégrité structurelle des citations | **1,000 ; 1,000** | La fidélité sémantique des annotations n'est pas prouvée par ce contrôle. [Détails](reponses_ef07.md). |
| Actions EF-08/09 | Atlas et Boréal synthétiques | 13 actions conformes ; 17 destinataires, faux choix | **13/13 ; 0** | Taille faible, aucune adresse inventée, `executed=false`. [Détails](actions_ef08_ef09.md). |
| Parcours complet | Huit scénarios Atlas | Cas conformes BM25 / hybride | **8/8 ; 8/8** | Recette fonctionnelle sur données fictives avec [sorties figées](recette_bout_en_bout.md). |
| Agent sur autre source | Deux réunions QMSum test, 12 demandes | Cas conformes BM25 / hybride | **11/12 ; 10/12** | Contrôle après correction ; [métriques BM25](../Data/Curated/qmsum_agent_holdout_metrics.json), [hybrides](../Data/Curated/qmsum_agent_holdout_hybrid_metrics.json) et sorties inspectables. |
| Agent, second contrôle figé | Deux autres réunions QMSum test, 9 demandes | Cas conformes BM25 / hybride | **9/9 ; 9/9** | Premier score sans adaptation au jeu ; [métriques BM25](../Data/Curated/qmsum_agent_final_bm25_metrics.json), [hybrides](../Data/Curated/qmsum_agent_final_hybrid_metrics.json) et sorties inspectables. |

Sur le **premier** couple QMSum annoté pour l'agent, le BM25 couvre les huit unités attendues par les questions et le dossier (`1,000`) ; l'hybride en couvre sept (`0,875`). Les citations respectent toutes le contrat de provenance exact, mais cette intégrité structurelle ne garantit ni la pertinence d'une unité choisie ni la vérité d'une annotation. Ce contrôle ne teste pas les relations temporelles ou la résolution positive d'un responsable, faute de dates et d'attributions explicites dans les extraits retenus.

Sur le **second** couple de réunions, les deux méthodes couvrent toutes les unités attendues et clarifient les demandes sans destinataire ou sans décision confirmée. Le dossier conserve les propositions, risques et points ouverts sans inventer de décision courante. Les [sorties BM25](../Data/Curated/qmsum_agent_final_bm25_outputs.jsonl) et [hybrides](../Data/Curated/qmsum_agent_final_hybrid_outputs.jsonl) permettent de lire chaque affirmation. Le jeu reste très petit, proche du domaine et des formulations de développement ; son 9/9 n'est pas une estimation robuste sur des demandes spontanées.

## EF-03 : négatifs difficiles et ablation

Les [25 triplets revus par l'assistant](../Data/Curated/qmsum_hard_negatives_reviewed.jsonl) viennent uniquement de réunions QMSum *train* ; 13 autres candidats ont été rejetés après lecture des passages. Une validation humaine indépendante de cette revue reste souhaitable. Le [contrôle aléatoire](../Data/Curated/qmsum_random_negative_control_25.jsonl) garde **les mêmes 25 questions et positifs** et tire, avec la graine 42, un autre passage de la même réunion hors de tous les spans pertinents annotés. Ce dernier n'a pas fait l'objet d'une revue sémantique humaine : un passage non annoté peut néanmoins être utile.

Les deux variantes repartent de `checkpoint-842`, sur le même GPU, une époque, batch 2, mini-batch 1, 256 tokens d'entraînement et taux `1e-5`. Elles sont comparées sur les **237 questions de validation**, sans nouvelle consultation du test. Les [métriques et empreintes](../Models/qmsum-e5-hardneg-ablation/metrics.json) conservent l'expérience.

| Variante | Hit@5 | Recall@5 | MRR@5 |
|---|---:|---:|---:|
| Référence, sans entraînement complémentaire | **0,8650** | 0,4694 | 0,6598 |
| 25 négatifs difficiles revus | 0,8565 | **0,4732** | **0,6667** |
| 25 négatifs aléatoires hors spans | 0,8354 | 0,4664 | 0,6511 |

L'écart MRR@5 entre les deux variantes entraînées est `+0,0156` en faveur des négatifs revus ; l'écart entre cette variante et la référence est `+0,0069`, avec un Hit@5 inférieur de `0,0085`. Une répétition CPU des 25 négatifs revus donne MRR@5 `0,6632`, ce qui montre une variabilité de run. Ce petit essai ne démontre pas un gain généralisable. **Le modèle de l'agent reste le checkpoint E5 initialement adapté sur QMSum**, choisi avant l'évaluation test. Aucun score test n'est revendiqué pour la nouvelle variante.

Le run GPU du contrôle aléatoire a lu un fichier CRLF écrit sous Windows. Le fichier suivi par Git est en LF ; les deux ne diffèrent que par les fins de ligne et leurs 25 objets JSON parsés sont identiques. Les deux SHA-256 sont consignés dans le rapport structuré. Une tentative de répétition GPU après normalisation a échoué au chargement du modèle par manque de mémoire disponible, avant l'entraînement ; le score publié reste celui du run initial.

## Erreurs qualitatives et contrôles de preuve

- `h10` demande le prix des batteries. Aucune unité ne le donne. L'agent demande une précision au lieu de retourner `not_found` : la clarification est prudente, mais son motif n'explique pas correctement l'absence de la donnée.
- `h02` demande le coût de production de 12,50 €. En hybride, la fusion place le prix de vente de 25 € en tête ; la réponse cite une **vraie** unité, mais la mauvaise. BM25 sélectionne l'unité attendue. Le contrôle de provenance seul ne détecte pas cette confusion.
- Le premier passage sur `h11` répondait à « qui est responsable du circuit » avec un risque exprimé par le *Industrial Designer*, sans attribution de responsabilité. La correction impose une clarification lorsqu'aucune unité du sujet ne porte de responsable explicite. Ce cas explique le passage de 10/12 à 11/12 en BM25.
- Les dossiers QMSum affichent les deux sources sans dates de réunion fiables ; l'ordre technique d'affichage ne doit pas être lu comme une chronologie datée. Les décisions confirmées sont marquées `confirmed_date_unknown`, jamais `current_decision`. Aucune relation `supersedes` n'a été inventée pour ce contrôle.

Les treize preuves QMSum des deux contrôles sont des sous-chaînes exactes des transcriptions originales. Le [préparateur](../Data%20Analysis/prepare_agent_holdout.py) régénère les conversations depuis les JSON QMSum suivis par Git, puis valide les unités contre leurs tours d'origine. Les [attendus du premier contrôle](../Data/Curated/qmsum_agent_holdout_cases.jsonl), ses [sorties BM25](../Data/Curated/qmsum_agent_holdout_outputs.jsonl) et [hybrides](../Data/Curated/qmsum_agent_holdout_hybrid_outputs.jsonl) permettent une revue manuelle des faits et des abstentions. La rédaction des unités et leur jugement de pertinence restent le fait d'un seul annotateur ; une double annotation indépendante n'a pas été réalisée.

Depuis la racine du dépôt, le second contrôle figé est rejouable en BM25 sans checkpoint :

```powershell
python "Data Analysis/evaluate_agent_holdout.py" --method bm25 --meetings ES2011a ES2011b --conversations Data/Processed/qmsum_agent_final_conversations.jsonl --units Data/Curated/qmsum_agent_final_units.jsonl --relations Data/Curated/qmsum_agent_final_relations.jsonl --topics Data/Curated/qmsum_agent_final_topics.jsonl --cases Data/Curated/qmsum_agent_final_cases.jsonl --expected-cases 9 --frozen-manifest Data/Curated/qmsum_agent_final_manifest.json --metrics Data/Processed/controle_final_bm25.json --outputs Data/Processed/controle_final_bm25.jsonl
```

Remplacer `--method bm25` par `--method hybrid --device cpu --model Models/qmsum-e5-small/checkpoints/checkpoint-842` et choisir des chemins de sorties distincts pour rejouer le mode dense lorsque le checkpoint est disponible. Le manifeste refuse un changement des sources ou attendus figés.

## Latence, mémoire et coût

Le [benchmark du mini-agent](../Data%20Analysis/benchmark_agent_system.py) rejoue les huit demandes Atlas trois fois **après une première requête**, sur Windows 11, Python 3.13.5 et CPU. Le RSS est celui du processus entier, échantillonné toutes les 5 ms. Les [sorties BM25](../Data/Curated/agent_system_bm25_metrics.json) et [hybrides](../Data/Curated/agent_system_hybrid_metrics.json) conservent les 24 mesures individuelles par mode.

| Mode | Chargement | Première requête | Médiane / p95 chaud | Pic RSS du processus |
|---|---:|---:|---:|---:|
| BM25 | 0,003 s | 2,380 s | 1,4 / 1,8 ms | 163 Mo |
| Hybride E5 + BM25 sur CPU | 20,219 s | 0,148 s | 18,5 / 22,6 ms | 1 657 Mo |

La première requête BM25 charge paresseusement le classifieur d'intentions ; le temps de construction du pipeline seul ne représente donc pas le vrai démarrage à froid. Le mode hybride charge le modèle dense dès la construction. Ce sont des mesures locales sur 16 unités, non des garanties de service ; la taille du corpus, la concurrence, le cache et le matériel modifieront latence et mémoire. Le coût monétaire d'API est nul : aucun appel externe n'est effectué. Le coût local est le temps CPU et la mémoire ci-dessus, sans valorisation en euros ni mesure énergétique.

## Reproduction et limites finales

Le [README](../README.md) donne les commandes depuis une copie propre, les dépendances, le mode BM25 disponible sans poids privés du poste et la procédure de réentraînement du checkpoint dense. Dans une copie de travail séparée, avec le même environnement Python préinstallé, la génération QMSum a produit les **mêmes empreintes SHA-256** pour les réunions normalisées et les 13 022 passages ; la recette BM25, le contrôle d'agent et la suite de tests ont été rejoués sans modèle dense. Les fins de ligne JSON/JSONL sont fixées en LF par [`.gitattributes`](../.gitattributes) pour stabiliser les empreintes sur Windows. Le SHA-256 de `model.safetensors` utilisé par les mesures hybrides est `1882ffc39c626972770c4f4778a12e72db9d0a1f60e40943c52c9689e3f21a46`. Les poids restent hors Git en raison de leur taille ; un réentraînement GPU ne garantit pas une empreinte identique au bit près. Les sorties hybrides archivées restent auditables même sans ces poids.

Les principales limites sont les annotations petites et largement synthétiques, les contrôles QMSum de seulement quatre réunions et vingt et une demandes créées par le projet, les dates inconnues, l'absence de contacts réels et l'absence d'une évaluation humaine indépendante à grande échelle. Avant le score hybride du second contrôle, deux chargements du modèle ont échoué par manque de mémoire disponible ; le premier run ayant produit des réponses a ensuite abouti sans changement de données ni de code. Le prototype remplit la recette et livre les mesures demandées ; il n'est pas validé pour une utilisation autonome sur des réunions ouvertes ou des actions réelles.
