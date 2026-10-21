# Cahier des charges - Embedding Lab

**Projet :** Semantic Agent Engine  
**Porteur :** Sacha KEFIF  
**Cadre :** Piscine Makers Intelligence Lab, projet individuel  
**Version :** 1.0 - 21 octobre 2026  
**Période indiquée dans le sujet :** 16 octobre au 16 novembre 2026  
**Effort de réalisation prévu par le sujet :** trois semaines

## 1. Objet et finalité

Concevoir un prototype de moteur d'intelligence sémantique capable de transformer plusieurs conversations en unités de connaissance interrogeables. Le projet porte d'abord sur l'entraînement ou l'adaptation d'embeddings pour retrouver les bonnes preuves malgré les reformulations et les ambiguïtés. Un mini-agent exploite ensuite ces preuves pour répondre à des questions, construire des dossiers et préparer des actions simulées.

La question centrale est la suivante : comment relier une demande aux informations pertinentes tout en distinguant les propositions, décisions confirmées, changements, responsabilités et informations devenues obsolètes ? Une proximité vectorielle ne constitue pas, à elle seule, une preuve de vérité ou de compatibilité logique.

## 2. Périmètre

### Inclus

- Ingestion de plusieurs conversations textuelles, avec identité de source et date.
- Segmentation en unités sémantiques typées : décision, action, engagement, question, réponse, fait, risque ou idée.
- Baselines de recherche lexicale et dense ; indexation et recherche hybride.
- Adaptation d'un encodeur par apprentissage contrastif avec négatifs difficiles.
- Identification de relations entre unités : appui, contradiction, remplacement, création d'action, réponse et absence de lien.
- Analyse des intentions et extraction des paramètres utiles à une réponse ou une proposition d'action.
- Génération de réponses et de dossiers sourcés, ainsi que de brouillons d'e-mail, de tâche ou d'événement.
- Interface simple de démonstration, en ligne de commande, application légère ou API.
- Évaluation reproductible des composants et du parcours complet.

### Hors périmètre

- Entraînement d'un grand modèle de langage depuis zéro.
- Connexions OAuth ou synchronisation avec des services réels.
- Envoi d'e-mails, création effective de tâches ou modification d'agenda.
- Invention d'une adresse, d'une attribution de responsabilité ou d'une décision absente des sources.

## 3. Utilisateurs et cas d'usage

L'utilisateur charge des comptes rendus ou conversations, puis interroge le moteur. Les scénarios de recette sont les suivants :

1. Retrouver la décision actuelle concernant le budget d'Atlas, avec passage justificatif et référence de source.
2. Distinguer une date proposée d'une date confirmée malgré une forte similarité lexicale.
3. Comparer deux réunions et expliquer si une information est contradictoire ou simplement remplacée.
4. Identifier qui s'est engagé à transmettre un dossier et pour quelle échéance.
5. Construire un dossier Atlas couvrant contexte, chronologie, faits, décisions, risques, responsables, points ouverts et références.
6. Préparer un brouillon au responsable du budget, en justifiant le destinataire ; laisser l'adresse vide si elle est inconnue.
7. Demander une clarification lorsque le destinataire, la décision ou la donnée demandée est ambiguë ou absente.
8. Ne déclencher aucune action pour une phrase hypothétique telle que « On pourrait peut-être revoir le budget ».

## 4. Exigences fonctionnelles

**EF-01 - Ingestion et provenance.** Le système accepte au moins deux conversations distinctes. Chaque unité produite conserve son identifiant, son texte, son type, la source, la date et l'extrait justificatif ; le locuteur et le responsable sont renseignés lorsqu'ils sont connus.

**EF-02 - Recherche.** Une requête renvoie des unités classées avec score, source et preuve. Les variantes lexicale, dense et hybride sont comparables sur les mêmes requêtes.

**EF-03 - Spécialisation des embeddings.** Un encodeur pré-entraîné est adapté au moyen de paires ou triplets contrastifs. Des négatifs difficiles portent notamment sur le statut, la négation, la personne, la date et le nombre. Le modèle adapté est comparé au modèle initial sur un test indépendant.

**EF-04 - Relations et temporalité.** Un module qualifie au minimum les liens `supports`, `contradicts`, `supersedes`, `creates_action`, `answers` et `unrelated`. Une décision courante s'appuie sur le statut, les dates et les preuves, pas sur la seule similarité.

**EF-05 - Mémoire.** Les unités et leurs liens permettent de suivre un sujet à travers plusieurs conversations, d'exposer l'historique et de signaler les informations obsolètes ou contradictoires.

**EF-06 - Intentions.** Le système distingue `answer`, `compare`, `build_dossier`, `draft_email`, `create_task`, `schedule_event` et `clarify`. Il extrait les paramètres disponibles : sujet, personne, échéance et action.

**EF-07 - Réponses et dossiers.** Toute affirmation importante est reliée à une ou plusieurs unités de preuve. Le dossier présente les faits établis, la chronologie, les décisions, les points ouverts, les responsables et les références. Une information introuvable est explicitement signalée.

**EF-08 - Actions simulées.** Les brouillons indiquent leur intention, leur opération candidate, leur contenu, les unités sources et le besoin de confirmation. Le champ `executed` reste `false`. Aucun connecteur ne réalise d'effet externe.

**EF-09 - Résolution du destinataire.** Une personne citée n'est pas automatiquement responsable. Le choix doit reposer sur une attribution explicite et suffisamment récente. Une adresse inconnue reste vide ; en cas de concurrence ou d'insuffisance de preuve, le système demande une précision.

## 5. Données et contrats

Les jeux de données sont recherchés et documentés par le porteur du projet. Pistes : corpus publics de réunions ou dialogues, jeux de recherche d'information, paraphrases et NLI, complétés par des conversations synthétiques ciblées. Avant usage, relever pour chaque corpus la provenance, la licence, les conditions de réutilisation, la préparation et les limites.

Un schéma d'annotation définit les unités, leurs statuts, les liens, les questions, les preuves et les intentions. Les jeux d'entraînement, validation et test sont séparés par conversation ou groupe de réunions. Le test doit, si possible, contenir des projets, formulations et personnes absents de l'entraînement. Les données synthétiques ne doivent pas être évaluées seulement sur des formulations issues des mêmes patrons.

Les transcriptions privées et données personnelles réelles ne sont pas utilisées sans autorisation appropriée. La démonstration emploie des contacts fictifs.

### Unité sémantique minimale

- `unit_id`, `source_id`, `source_date`, `type`, `statement`, `evidence`, `status`.
- Champs conditionnels : `speaker`, `owner`, `due_date`, `project_id`.

### Relation minimale

- `from_id`, `to_id`, `relation`, `evidence_source_id`.

### Brouillon d'action minimal

- `intent`, `tool_candidate`, paramètres de l'action, `source_unit_ids`, `needs_confirmation: true`, `executed: false`.
- Pour l'e-mail : nom du destinataire, adresse éventuellement nulle, justification par unités et statut de résolution.

## 6. Architecture cible

1. Charger et normaliser les conversations ; conserver les métadonnées de provenance.
2. Segmenter et annoter les unités sémantiques.
3. Indexer les unités dans des moteurs lexical et vectoriel.
4. Classer la requête et extraire les paramètres utiles.
5. Rechercher, fusionner, filtrer et éventuellement réordonner les résultats.
6. Vérifier les relations, la chronologie, le statut et l'attribution des personnes.
7. Produire une réponse, un dossier, un brouillon ou une demande de clarification.
8. Retourner la sortie structurée et les références, sans exécuter l'action proposée.

Technologies envisagées, sous réserve de mesure et de ressources disponibles : Python, PyTorch, Sentence Transformers, scikit-learn, BM25, FAISS ou équivalent, Cross-Encoder ou NLI léger, puis CLI, Streamlit, Gradio ou FastAPI. Un LLM peut formuler les sorties, mais les erreurs de recherche, de relation et de routage restent évaluées séparément.

## 7. Expérimentation et critères de recette

Le protocole fixe les corpus, les splits, les versions de modèles, les paramètres, les graines aléatoires et les requêtes avant comparaison. Les mêmes jeux de test servent aux baselines et à l'encodeur adapté. Au moins une étude d'ablation isole l'effet des négatifs difficiles, du contexte ou du reranking.

- **Retrieval :** Recall@K, MRR et nDCG@K ; comparaison lexicale, dense initiale, dense adaptée et hybride.
- **Embeddings :** gain ou perte par rapport à l'encodeur initial, notamment sur les négatifs difficiles.
- **Relations :** macro-F1, précision et rappel par classe, matrice de confusion.
- **Intentions :** macro-F1, exactitude des paramètres et taux de clarification appropriée.
- **Destinataires :** Precision@1, faux destinataires et abstentions justifiées.
- **Réponses et dossiers :** couverture des faits attendus, exactitude des citations et affirmations sans preuve.
- **Système :** latence, mémoire utilisée et coût d'inférence.

La recette fonctionnelle exige que les huit scénarios de la section 3 soient rejouables avec sorties et preuves inspectables. Le rapport présente les résultats chiffrés, les erreurs qualitatives et les limites. Le sujet ne fixe aucun seuil numérique de performance ; les objectifs chiffrés seront définis après les premières baselines, puis consignés avant l'évaluation finale.

## 8. Plan de réalisation et jalons

**Semaine 1 - Données et baselines.** Étude des méthodes, sélection et vérification des corpus, schéma d'annotation, premier jeu de questions et preuves, baselines BM25 et embeddings pré-entraînés. Jalon : données documentées, protocole de test et premiers scores.

**Semaine 2 - Adaptation et relations.** Paires ou triplets contrastifs, entraînement raisonnable selon le calcul disponible, négatifs difficiles, classifieur de relations, reranking et étude d'ablation. Jalon : modèle adapté, comparaisons reproductibles et analyse des erreurs.

**Semaine 3 - Agent et démonstration.** Mémoire interrogeable, intentions, réponses et dossiers sourcés, outils simulés, clarification, tests de bout en bout et stabilisation. Jalon : prototype fonctionnel, démonstration et rapport expérimental.

La période administrative indiquée dans le sujet s'étend du 16 octobre au 16 novembre 2026 ; elle ne change pas la répartition en trois semaines de travail décrite ci-dessus.

## 9. Livrables

1. Dépôt structuré, exécutable et documenté, avec instructions de reproduction.
2. Fiche des datasets : provenance, licences, préparation, annotations et splits.
3. Baselines, modèle adapté et scripts d'entraînement, d'inférence et de benchmark.
4. Modules de recherche, relations et intentions ; contrats JSON validés.
5. Mini-agent avec interface ou API, dossiers et brouillons sourcés ; connecteurs simulés.
6. Rapport expérimental incluant résultats, ablations, erreurs, coûts et limites.
7. Démonstration reproductible couvrant les scénarios de recette.

## 10. Risques et décisions à prendre

- **Qualité des annotations :** définir des consignes explicites et contrôler un échantillon annoté avant l'entraînement.
- **Fuite de données :** séparer les conversations avant de générer les exemples et auditer les doublons entre splits.
- **Confusion logique :** compléter la similarité par un contrôle des relations, statuts, dates et preuves.
- **Ressources de calcul :** choisir un encodeur compact et consigner temps d'entraînement, mémoire et coût.
- **Corpus insuffisants :** documenter les lacunes des sources publiques et créer des cas synthétiques ciblés, avec test distinct.
- **Hallucinations et fausses attributions :** imposer l'abstention ou la clarification quand une preuve ou une identité manque.

Les choix précis de corpus, d'encodeur, de stratégie de stockage et d'interface restent à arrêter après les premières mesures. Toute modification du périmètre ou des critères de recette doit être tracée dans le dépôt.
