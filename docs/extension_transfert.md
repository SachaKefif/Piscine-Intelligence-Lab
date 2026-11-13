# Extension : ingestion et contrôles de transfert

Cette extension suit le [rapport expérimental initial](rapport_experimental_final.md). Les résultats initiaux et leurs fichiers ne sont pas réécrits. Les contrôles rejoués après correction sont des **diagnostics postérieurs**, pas de nouveaux tests aveugles.

## Ingestion de conversations nouvelles

[`bootstrap_memory.py`](../Data%20Analysis/bootstrap_memory.py) accepte deux fichiers texte ou plus, un tour par ligne, ou des conversations JSONL normalisées. Il produit les quatre fichiers lus par l'agent (`conversations`, `units`, `topics`, `relations`) et une file de `relation_suggestions`. Chaque unité garde un extrait exact, son tour, sa source et sa date éventuelle. L'ingestion refuse les sources dupliquées, les preuves non présentes et un dossier de sortie non vide. Sur les deux transcriptions publiques `ES2011a/b` tenues hors entraînement, elle a créé 652 unités avec validation de provenance. Ce nombre représente des tours, **pas** 652 faits vérifiés.

La classification automatique du type est heuristique. Une proposition ou une hypothèse reste ainsi marquée comme telle, tandis qu'une décision, un engagement ou un responsable potentiel reste `unknown` ou `null` tant qu'il n'est pas revu. Les relations proposées ne sont pas chargées comme relations validées. Cette règle empêche de transformer une similarité en décision actuelle ou en attribution de personne. La mémoire produite permet déjà une recherche et une réponse citant un énoncé incertain ; une réponse décisionnelle ferme et un brouillon d'action demandent des annotations revues.

## Relations sur questions–réponses publiques

Le classifieur [`answer_relation_model.py`](../Data%20Analysis/answer_relation_model.py) utilise des similarités TF-IDF mot/caractère, un recouvrement lexical et une concordance des nombres. Il apprend sur 162 réunions QMSum train, choisit son seuil sur 35 réunions validation, puis est mesuré sur 35 réunions test distinctes. Le [rapport JSON](answer_relation_evaluation.json) donne :

| Jeu | Paires | Rappel `answers` | Précision proxy | Faux positifs proxy |
|---|---:|---:|---:|---:|
| Validation | 709 | 0,7764 | 0,5732 | 0,2903 |
| Test | 732 | 0,8074 | 0,5760 | 0,2971 |

Les négatifs sont d'autres réponses de la même réunion. Certains peuvent répondre partiellement à la question : leur statut négatif est un **proxy**, non une annotation indépendante. Un second diagnostic de 220 paires positives et 220 paires négatives issues des questions originales QMSum, en écartant `education_13/17`, donne un rappel de 0,8091 et un taux de faux positifs proxy de 0,3636. Le classifieur ne couvre que `answers` en anglais ; `supports`, `contradicts`, `supersedes`, `creates_action` et `unrelated` restent soumis aux règles prudentes et à une revue humaine. Aucune prédiction de modèle n'entre automatiquement dans la mémoire validée.

## Intentions, destinataires et agent : diagnostics après correction

Les jeux QMSum et MASSIVE ci-dessous avaient déjà révélé des échecs et ont servi à orienter les changements. Les chiffres suivants mesurent le **même** jeu après correction. Les sorties de contrôle sont régénérables avec [`evaluate_agent_holdout.py`](../Data%20Analysis/evaluate_agent_holdout.py) et [`evaluate_massive_recipients.py`](../Data%20Analysis/evaluate_massive_recipients.py), en choisissant des chemins de sortie distincts des rapports historiques.

| Contrôle | Avant | Après | Limite |
|---|---:|---:|---|
| 15 questions originales QMSum `education_13/17`, cas conformes | 0/15 | 12/15, BM25 et hybride | Trois réponses citent la mauvaise unité malgré une citation valide. |
| 17 destinataires directement nommés dans MASSIVE, nom correct | 0/17 | 17/17 | Ne teste pas l'attribution d'un rôle à partir d'une réunion. |
| 49 demandes MASSIVE sans identité de destinataire, abstention correcte | 49/49 | 49/49 | Aucun annuaire ni adresse n'est fourni. |
| 66 intentions MASSIVE, catégorie attendue | 0/66 | 49/66 | Le jeu vocal contient des consultations et notifications hors des sept intentions du prototype. |

Le parsing reconnaît désormais les questions anglaises directes et les noms écrits en minuscules après « e-mail à/pour ». Il sépare une personne mentionnée dans une consultation d'e-mails d'un destinataire nommé. Un projet ou une preuve manquants continuent de bloquer la préparation d'une action ; aucune adresse n'est inventée.

## Nouveau contrôle QMSum figé avant exécution

[`prepare_public_agent_transfer.py`](../Data%20Analysis/prepare_public_agent_transfer.py) a créé, puis figé par [empreintes](../Data/Curated/qmsum_academic_transfer_manifest.json), 12 questions originales des annotateurs sur `Bed003` et `Bed008` du test Academic. Ces réunions n'avaient pas servi aux contrôles antérieurs de l'agent. Le premier passage BM25 et le premier passage hybride ont été exécutés sans adaptation au jeu. Les [métriques BM25](../Data/Curated/qmsum_academic_transfer_bm25_metrics.json), [métriques hybrides](../Data/Curated/qmsum_academic_transfer_hybrid_metrics.json) et sorties complètes sont conservées.

| Mesure | BM25 | Hybride |
|---|---:|---:|
| Cas entièrement conformes | 5/12 | 5/12 |
| Intentions exactes | 8/12 | 8/12 |
| Couverture des réponses attendues | 5/12 | 5/12 |
| Pertinence des citations parmi les réponses citées | 5/8 | 5/8 |
| Contrat de provenance des citations | 12/12 | 12/12 |

Le jeu fournit à l'agent les réponses humaines sous forme d'unités liées aux tours de source. Il évalue le routage, la recherche et la sélection de l'unité, **pas** l'extraction automatique de la réponse à partir de la transcription brute. Le score 5/12 montre que la généralisation de l'agent restait insuffisante après la correction des erreurs connues. Une version suivante a ajouté le classement des unités `answer` QMSum par le modèle de paires et a reconnu les demandes anglaises de résumé, sur la base de la validation QMSum. Sur les 237 questions de cette validation, le modèle classe la réponse correcte en premier dans 54,01 % des cas, contre 45,99 % pour BM25 seul. Le premier jeu de 12 questions ne constitue donc plus un test final indépendant pour cette version.

Un **second** jeu, `Bmr006/Bmr014`, a été [figé avant sa première exécution](../Data/Curated/qmsum_academic_transfer2_manifest.json) avec dix questions originales d'annotateurs QMSum. Sans nouveau réglage sur ce jeu, la version avec classement par modèle obtient [4/10 cas conformes en BM25](../Data/Curated/qmsum_academic_transfer2_bm25_metrics.json) et [4/10 en hybride](../Data/Curated/qmsum_academic_transfer2_hybrid_metrics.json). Les dix intentions et statuts sont corrects, mais seulement quatre réponses choisissent l'unité attendue ; les dix sorties respectent le contrat de provenance. Le gain sur la validation **ne s'est pas traduit par une sélection fiable sur ces nouvelles réunions**. Les sorties sont conservées pour examiner chaque erreur sans retoucher ce test.

Un seuil d'abstention issu de la validation a ensuite été appliqué au classement des réponses, afin qu'un score faible ne force pas une réponse. Un **troisième** jeu, `Bro004/Bro019`, a été [figé avant son premier passage](../Data/Curated/qmsum_academic_transfer3_manifest.json). Cette version obtient [11/12 cas conformes en BM25](../Data/Curated/qmsum_academic_transfer3_bm25_metrics.json) et [11/12 en hybride](../Data/Curated/qmsum_academic_transfer3_hybrid_metrics.json) sur 12 questions originales ; 12/12 intentions et statuts sont corrects, et 11/12 réponses citent l'unité attendue. Aucune règle n'a été modifiée après ce passage. Le contraste entre 4/10 et 11/12 signale une variation importante selon les réunions : ces petits jeux ne donnent pas une estimation stable de la performance générale.

## Travail qui reste scientifiquement ouvert

Une évaluation humaine indépendante des décisions, contradictions, responsabilités et dossiers sur des conversations nouvelles n'est pas disponible dans ces corpus. Un [lot AMI figé](protocole_annotation_independante_ami.md) prépare désormais 33 paires de décisions et 14 actions issues de quatre réunions absentes des splits QMSum locaux, avec deux paquets aveugles et un évaluateur qui refuse les scores sans deux avis humains complets. Son état actuel est `pending_human_review` : aucun résultat de contradiction, remplacement ou responsable n'est encore établi. Pour une mesure générale, il faudra davantage de réunions, des annotations séparées de plusieurs personnes, des splits par projet ou réunion et un nouveau jeu final préservé de toute sélection de règles ou de seuils. Les dates absentes de QMSum ne permettent pas de tester honnêtement `supersedes` ou la fraîcheur des responsabilités. Les relations à faux positifs élevés doivent rester des propositions à confirmer.
