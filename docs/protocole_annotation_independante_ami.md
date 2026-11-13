# Validation humaine indépendante sur quatre réunions AMI

Le [jeu figé](../Data/Curated/independent_ami_review/manifest.json) utilise quatre réunions AMI (`IB4003`, `IB4005`, `IB4010`, `IB4011`) absentes des trois splits QMSum locaux. Les transcriptions manuelles et résumés de décisions/actions proviennent du [corpus AMI, annotations manuelles v1.6.2](https://groups.inf.ed.ac.uk/ami/download/), sous CC BY 4.0. Les [descriptions des annotations](https://groups.inf.ed.ac.uk/ami/corpus/annotation.shtml) indiquent que les sections « décisions » et « actions » sont écrites par des annotateurs. Selon la [description officielle des identifiants](https://groups.inf.ed.ac.uk/ami/corpus/meetingids.shtml), les réunions `IB` mettent en scène des scénarios fictifs d'aménagement de bureaux ou de ciné-club. Ce corpus **ne fournit pas** d'étiquettes de contradiction logique, de remplacement entre réunions ni de responsable structuré pour les paires retenues ; ces jugements restent à recueillir.

## Matériel prêt à annoter

- [Dossier source](../Data/Curated/independent_ami_review/source_packet.md) : résumés humains et liens vers les quatre transcriptions manuelles reconstruites, avec horodatage et locuteur.
- [Paquet A](../Data/Curated/independent_ami_review/reviewer_a_packet.zip) : formulaire [relations](../Data/Curated/independent_ami_review/reviewer_a_pairs.csv), formulaire [responsables](../Data/Curated/independent_ami_review/reviewer_a_owners.csv), sources et transcriptions.
- [Paquet B](../Data/Curated/independent_ami_review/reviewer_b_packet.zip) : formulaire [relations](../Data/Curated/independent_ami_review/reviewer_b_pairs.csv), formulaire [responsables](../Data/Curated/independent_ami_review/reviewer_b_owners.csv), sources et transcriptions.

Les formulaires contiennent **33 paires de décisions** : le produit complet des décisions humaines dans chacun des deux couples de réunions, sans filtrage par le modèle, et **14 actions**. Les [prédictions du système](../Data/Curated/independent_ami_review/model_predictions.jsonl) sont dans un fichier distinct à ne communiquer aux évaluateurs qu'après remise de leurs formulaires. Les empreintes des sources, candidats et prédictions sont fixées dans le manifeste. Pour les XML AMI, le calcul d'empreinte convertit uniquement les fins de ligne CRLF en LF afin qu'un checkout Windows et une copie propre donnent le même résultat ; les autres fichiers figés sont vérifiés octet par octet. Ce petit lot est un pilote pour mesurer les erreurs ; il ne suffira pas à établir une performance générale avec précision statistique.

## Consignes aux deux évaluateurs

Remettre uniquement le paquet A au premier évaluateur et le paquet B au second. Les évaluateurs doivent travailler séparément, sans consulter les prédictions ni les réponses de l'autre. Ils lisent au minimum les deux énoncés et le contexte des sections correspondantes dans le dossier source. Ils peuvent consulter les transcriptions pour vérifier une formulation. Ils ne modifient que les colonnes de jugement des CSV ; les identifiants, textes et chemins sources restent intacts.

Pour chaque **paire** :

- `semantic_relation` : `supports` si les deux décisions expriment le même fait compatible ; `contradicts` si elles sont incompatibles **pour le même objet et dans le même contexte** ; `unrelated` si elles ne portent pas sur la même proposition ; `uncertain` si les sources ne permettent pas de trancher. Un désaccord argumentatif ou une différence de formulation ne suffit pas à constituer une contradiction.
- `temporal_relation` : `supersedes` seulement si une source identifiable établit que la seconde décision remplace la première ; `not_supersedes` si les deux concernent des sujets différents ou si un remplacement est explicitement exclu ; `unknown_order` si l'ordre ou l'acte de remplacement n'est pas établi. La [table officielle des enregistrements](https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml) date `IB4003/IB4005` du **2 novembre 2004** et `IB4010/IB4011` du **8 avril 2005**. Les dates seules ne prouvent pas l'ordre. La [description officielle des identifiants](https://groups.inf.ed.ac.uk/ami/corpus/meetingids.shtml) situe `IB4005` après les groupes précédents, dont `IB4003` ; cela établit l'ordre des réunions de bureaux, pas le remplacement d'une décision précise. Aucun ordre n'y est établi pour `IB4010/IB4011`.
- `chronology_evidence` : pour `supersedes`, indiquer la source précise et l'extrait établissant à la fois l'ordre et le remplacement. L'évaluateur note `uncertain` ou `unknown_order` dès que l'une de ces preuves manque.

Pour chaque **action** :

- `owner` : copier le nom ou le rôle explicitement chargé de l'action dans l'énoncé, ou écrire `none`, `unknown` ou `ambiguous`. Une personne citée, le locuteur ou un groupe indéfini ne devient pas responsable par défaut.
- `owner_evidence` : copier le passage exact de la phrase qui établit l'attribution. Le programme refuse un extrait absent du texte.

Chaque évaluateur ajoute une note lorsqu'il hésite. Un troisième arbitrage est requis pour les désaccords ; le système ne choisit jamais automatiquement l'évaluateur A ou B comme vérité.

## Audit et calcul des scores

```powershell
.venv\Scripts\python.exe "Data Analysis\evaluate_independent_ami_review.py"
```

L'évaluateur vérifie les empreintes, la complétude, la séparation des deux formulaires et les désaccords. Il renvoie actuellement `pending_human_review`, sans précision, rappel ni F1. Une fois les deux formulaires remplis, les désaccords peuvent être consignés dans des CSV d'arbitrage indiqués par `--adjudicated-pairs` et `--adjudicated-owners`. Ce n'est qu'après ces étapes qu'il calcule la précision et le rappel de `contradicts` et `supersedes`, ainsi que l'exactitude et les fausses attributions de responsables.

Après réception des quatre formulaires, `--adjudication-dir CHEMIN` crée des CSV vierges contenant uniquement les items en désaccord, avec les deux jugements dans `notes`. Un troisième évaluateur renseigne les colonnes de jugement et leurs preuves ; relancer ensuite l'audit avec `--adjudicated-pairs CHEMIN/adjudicated_pairs.csv` et/ou `--adjudicated-owners CHEMIN/adjudicated_owners.csv`. La création refuse d'écraser un fichier d'arbitrage existant.

Un résultat sans exemples positifs d'une classe doit garder sa précision ou son rappel à `null` ; il ne prouve pas que le détecteur maîtrise cette classe. Si les annotateurs ne trouvent pas de remplacement explicite dans le couple de bureaux, `supersedes` sera **non évaluable** sur ce lot. Il faudra alors recueillir d'autres réunions avec décisions successives et ordre vérifié.

## État actuel

Les quatre transcriptions ont été extraites des fichiers manuels AMI et les 47 items sont prêts. Aucun jugement humain indépendant n'a encore été reçu. Les scores de contradiction, remplacement et attribution ne sont donc **pas établis** ; toute valeur chiffrée avant réception des deux formulaires serait fabriquée.
