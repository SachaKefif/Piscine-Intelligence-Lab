# TODO - Embedding Lab / Semantic Agent Engine

## Semaine 1 - Recherche, données et baselines

### Cadrage et protocole
- [ ] Formuler les hypothèses : apport des embeddings adaptés au retrieval, puis des relations, du contexte et du reranking à l'agent.
- [ ] Étudier les représentations de phrases, tours de parole et unités métier, la recherche sémantique, le retrieval conversationnel, les Transformers et l'apprentissage contrastif.
- [ ] Fixer avant comparaison corpus, requêtes, jugements de pertinence, splits, métriques, versions des modèles, hyperparamètres et graines aléatoires.
- [ ] Évaluer les ressources de calcul ; choisir et justifier un encodeur compact, le stockage/index et l'interface après les premiers essais.
- [ ] Suivre les expériences et tracer toute décision technique, modification du périmètre ou des critères de recette.

### Corpus et annotations
- [ ] Rechercher et sélectionner des corpus publics de réunions/dialogues, recherche d'information, paraphrases ou NLI ; examiner notamment AMI, ICSI, MeetingBank et QMSum.
- [ ] Documenter pour chaque corpus la provenance, licence, conditions de réutilisation, contenu, préparation, annotations, limites et usage prévu.
- [ ] Écarter les transcriptions privées et données personnelles réelles sans autorisation ; employer des contacts fictifs pour la démonstration.
- [ ] Constituer au moins deux conversations distinctes partageant des sujets ; préparer questions, réponses attendues et passages justificatifs identifiables.
- [ ] Définir les consignes d'annotation des unités, statuts, liens, questions, preuves, intentions et paramètres ; contrôler manuellement un échantillon.
- [ ] Couvrir `decision`, `action`, `commitment`, `question`, `answer`, `fact`, `risk`, `idea`, ainsi que les objections et simples mentions utiles.
- [ ] Inclure des cas difficiles : proposition/décision, accord conditionnel/validation, annulation/négation, changements de date/budget/responsable et nombres proches mais faux.
- [ ] Inclure plusieurs réunions sur un sujet, contradictions/obsolescence, personne citée/responsable, information/instruction d'action, destinataire absent/ambigu et question sans réponse.
- [ ] Compléter si besoin par des conversations synthétiques ciblées et documenter leurs patrons et limites.
- [ ] Séparer entraînement, validation et test par conversation ou groupe de réunions AVANT de créer les exemples ; auditer doublons et fuites.
- [ ] Réserver si possible au test des projets, personnes et formulations inédits ; ne pas évaluer les données synthétiques seulement sur les mêmes patrons.

### Représentation et premiers scores
- [ ] Définir et valider le contrat JSON des unités : `unit_id`, `source_id`, `source_date`, `type`, `statement`, `evidence`, `status` ; champs conditionnels `speaker`, `owner`, `due_date`, `project_id`.
- [ ] Charger, normaliser, segmenter et annoter les conversations en conservant source, date, locuteur et extrait d'origine.
- [ ] Comparer découpage fixe, par tour de parole et en unités sémantiques, notamment sur requêtes courtes et reformulations.
- [ ] Construire une baseline lexicale BM25 ou TF-IDF et une baseline dense avec encodeur pré-entraîné multilingue (E5, BGE, Sentence-BERT ou équivalent).
- [ ] Indexer les unités ; rechercher par cosinus/plus proches voisins (FAISS ou équivalent si utile) ; renvoyer unités classées, scores, sources et preuves.
- [ ] Mesurer Recall@K, MRR et nDCG@K sur les mêmes requêtes ; relever les confusions de statut, négation, personnes, dates et nombres.
- [ ] Jalon S1 : données et licences documentées, schéma et splits, protocole écrit et premiers scores reproductibles.

## Semaine 2 - Embeddings adaptés, relations et recherche

### Apprentissage contrastif (coeur ML/DL obligatoire)
- [ ] Construire des paires ou triplets requête/positif/négatif à partir des preuves annotées.
- [ ] Miner des négatifs difficiles proches du sujet mais faux par statut, négation, personne, date, nombre ou rôle ; contrôler leurs étiquettes.
- [ ] Choisir et documenter une perte contrastive : InfoNCE, Multiple Negatives Ranking Loss, Triplet Loss ou Supervised Contrastive Learning.
- [ ] Adapter l'encodeur aux ressources disponibles ; conserver modèle initial, modèle adapté, configuration et scripts d'entraînement/inférence.
- [ ] Comparer initial et adapté sur un test indépendant, dont un sous-ensemble de négatifs difficiles ; rapporter gains et pertes.
- [ ] Étudier l'effet des négatifs, du contexte ajouté et, si possible, de la taille du modèle ; mener au moins une ablation contrôlée.

### Relations, temporalité et retrieval
- [ ] Définir et valider le contrat des relations : `from_id`, `to_id`, `relation`, `evidence_source_id`.
- [ ] Annoter puis détecter `supports`, `contradicts`, `supersedes`, `creates_action`, `answers`, `unrelated` ; comparer si utile classifieur léger, Cross-Encoder ou NLI.
- [ ] Distinguer contradiction, mise à jour, validation, proposition et absence de lien à partir des statuts, dates, projets et preuves, sans confondre similarité et compatibilité logique.
- [ ] Relier les unités entre conversations ; déterminer l'état courant justifié et exposer l'historique, les informations remplacées et les contradictions non résolues.
- [ ] Ajouter la recherche dense adaptée et la fusion hybride ; étudier filtres par type/date/projet et reranking.
- [ ] Comparer lexical, dense initial, dense adapté et hybride sur le même test ; mesurer l'effet des filtres et du reranking.
- [ ] Évaluer les relations avec macro-F1, précision/rappel par classe et matrice de confusion ; analyser les erreurs logiques et temporelles.
- [ ] Jalon S2 : encodeur adapté, module de relations, comparaisons et ablation reproductibles, analyse qualitative des erreurs.

## Semaine 3 - Mémoire, agent et démonstration

### Mémoire et intentions
- [ ] Construire une mémoire interrogeable d'unités et liens, consultable par sujet/projet à travers plusieurs conversations.
- [ ] Implémenter la chaîne explicite : intention/paramètres -> recherche -> fusion/filtrage/reranking -> vérification des relations, dates, statuts et personnes -> opération -> sortie sourcée.
- [ ] Détecter `answer`, `compare`, `build_dossier`, `draft_email`, `create_task`, `schedule_event`, `clarify`, y compris les demandes combinant plusieurs opérations.
- [ ] Extraire sujet, personne, échéance et action ; détecter paramètres absents/ambigus et router vers une clarification.
- [ ] Évaluer intentions et paramètres séparément de la formulation éventuelle par LLM.

### Réponses, dossiers et actions simulées
- [ ] Générer des réponses et comparaisons où chaque affirmation importante cite une ou plusieurs unités avec extrait, source, date et statut ; signaler toute donnée introuvable.
- [ ] Générer un dossier Atlas multi-sources en Markdown ou JSON : contexte, faits, chronologie, décisions/budget, risques, responsables, prochaines étapes, points ouverts et références.
- [ ] Dédupliquer le dossier, résoudre les versions remplacées et exposer les contradictions sans inventer de décision.
- [ ] Définir et valider le contrat JSON des brouillons : `intent`, `tool_candidate`, paramètres/contenu, `source_unit_ids`, `needs_confirmation: true`, `executed: false`.
- [ ] Simuler brouillon d'e-mail, tâche et événement (document si utile), sans effet externe.
- [ ] Pour l'e-mail, produire objet, corps factuel, nom du destinataire, `evidence_unit_ids` et `resolution_status` ; garder `email: null` si l'adresse est inconnue.
- [ ] Choisir le destinataire par attribution explicite et assez récente, jamais par simple mention ; clarifier en cas de candidats concurrents ou de preuve insuffisante.
- [ ] Ne déduire ni décision ni action d'une hypothèse ou simple demande d'information ; clarifier si décision, donnée ou personne manque.
- [ ] Proposer une interface simple (CLI, Streamlit, Gradio ou API) pour charger plusieurs conversations, interroger le moteur et inspecter sorties, scores et preuves.
- [ ] Garder retrieval, relations et routage évaluables séparément si un LLM formule les sorties.
- [ ] Jalon S3 : prototype exécutable, agent et outils simulés, démonstration complète et rapport expérimental.

## Recette fonctionnelle - conserver sorties et preuves inspectables
- [ ] R1. Retrouver la décision EN VIGUEUR sur le budget Atlas, avec extrait, date et source.
- [ ] R2. Distinguer une date proposée d'une date confirmée malgré la similarité des textes.
- [ ] R3. Comparer deux réunions et expliquer contradiction ou remplacement temporel, preuves à l'appui.
- [ ] R4. Identifier qui s'est engagé à transmettre un dossier et son échéance, avec intervention source.
- [ ] R5. Construire le dossier Atlas complet et traçable à partir de plusieurs conversations.
- [ ] R6. Préparer un e-mail au responsable du budget, avec destinataire justifié, adresse nulle si inconnue et aucune exécution.
- [ ] R7. Demander une clarification si destinataire, décision ou donnée est ambiguë/absente ; ne rien inventer.
- [ ] R8. Vérifier que « On pourrait peut-être revoir le budget » ne déclenche ni décision ni action.
- [ ] Tester aussi les propositions de tâche et d'événement, questions sans preuve et changements de responsable/date/budget.

## Évaluation finale, qualité et livrables
- [ ] Fixer des objectifs chiffrés après les premières baselines et AVANT l'évaluation finale (aucun seuil imposé par le sujet).
- [ ] Mesurer retrieval (Recall@K, MRR, nDCG@K) et gain/perte de l'encodeur adapté, notamment sur négatifs difficiles.
- [ ] Mesurer relations (macro-F1, précision/rappel par classe, confusion) et intentions (macro-F1, exactitude des paramètres, clarification appropriée).
- [ ] Mesurer destinataires (Precision@1, faux destinataires, abstentions justifiées).
- [ ] Mesurer réponses/dossiers (couverture des faits, exactitude des citations, affirmations sans preuve).
- [ ] Mesurer système complet (latence, mémoire, coût d'inférence) et temps/coût/mémoire d'entraînement.
- [ ] Analyser négations, changements temporels, confusions de personnes, fausses attributions, dossiers incomplets et requêtes sans preuve.
- [ ] Vérifier les contrats JSON et tests de bout en bout ; rendre les huit scénarios rejouables avec preuves inspectables.
- [ ] Livrer un dépôt structuré, exécutable et documenté avec dépendances et commandes de préparation, entraînement, inférence, benchmark et démonstration.
- [ ] Livrer fiche des datasets, annotations/splits, baselines, modèle adapté, modules recherche/relations/intentions et scripts associés.
- [ ] Livrer agent/interface, contrats JSON validés, dossiers/brouillons sourcés et connecteurs simulés.
- [ ] Rédiger le rapport : protocole, résultats, comparaisons, ablations, erreurs, coûts, limites et choix techniques.
- [ ] Vérifier le périmètre : pas de LLM entraîné depuis zéro, pas d'OAuth/synchronisation réelle, pas d'envoi ni de création effective de tâche/événement, pas d'adresse/responsabilité/décision inventée.

## Extensions facultatives (après la recette)
- [ ] Si utile, exporter le dossier en PDF/DOCX ; Markdown ou JSON suffit pour le sujet.
- [ ] Étudier de futurs connecteurs autorisés (messagerie, contacts, agenda, base documentaire, gestion de projet) avec contrôle des permissions et confirmation avant toute action réelle.
