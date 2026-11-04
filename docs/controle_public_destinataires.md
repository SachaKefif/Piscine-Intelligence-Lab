# Contrôle public des destinataires nommés

Le [corpus MASSIVE](https://github.com/alexa/massive) fournit des demandes
d'assistant vocal localisées en français, avec intentions et segments `person`
ou `relation` annotés et revus. Les données sont sous licence CC BY 4.0.
Le [test français publié sur Hugging Face](https://huggingface.co/datasets/AmazonScience/massive/blob/2c02f1bc446b202987536ab9b2014579b1e7af3e/fr-FR/massive-test.parquet)
a été fixé à cette révision et à l'empreinte SHA-256
`aacd57aa2bcf6e1d037a350a41b70cb756bbf51199d2023f40a3ebb9d38a1981`.

Les [66 cas](../Data/Curated/massive_fr_recipient_cases.jsonl) et leur
[manifeste](../Data/Curated/massive_fr_recipient_manifest.json) ont été figés
dans le commit `f434242` **avant** l'exécution de l'évaluateur. La sélection
retient tous les exemples du split test pour lesquels les juges ont confirmé
unanimement intention et segments, avec une note de grammaire au moins égale
à 3. Elle comprend :

- 17 demandes `email_sendemail` où un segment `person` suit directement
  « e-mail à/pour » ; ce segment sert de **proxy** pour un destinataire nommé ;
- 16 demandes `email_sendemail` avec une relation (`mon chef`, `ma famille`…)
  sans identité de personne ; l'agent doit s'abstenir sans annuaire ;
- 33 demandes de consultation d'e-mails ou de contacts qui mentionnent une
  personne ; celle-ci ne doit pas devenir le destinataire d'un brouillon.

| Mesure | Résultat |
| --- | ---: |
| Destinataire nommé correct au rang 1 | **0/17** |
| Abstentions injustifiées sur nom direct | **17/17** |
| Abstentions appropriées sans identité de destinataire | **49/49** |
| Faux destinataires dans ces 49 cas | **0** |
| Precision@1 parmi les destinataires choisis | `null` (aucun choix) |
| Intention EF-06 correcte selon la catégorie | **0/66** |
| Adresses inventées | **0** |

Les [métriques détaillées](../Data/Curated/massive_fr_recipient_metrics.json)
et les [66 sorties](../Data/Curated/massive_fr_recipient_outputs.jsonl)
permettent la revue. EF09 retourne `recipient_not_designated` dans les
66 cas. Les 17 noms directs sont écrits en minuscules dans le corpus ; le
détecteur actuel de destinataire explicite exige une majuscule initiale.
EF-06 clarifie 64 demandes et classe deux demandes de consultation comme
création de tâche. Aucune règle n'a été modifiée à partir de ce test.

Pour reproduire depuis la racine du projet :

```powershell
.venv\Scripts\python.exe "Data Analysis\prepare_massive_fr_eval.py"
.venv\Scripts\python.exe "Data Analysis\evaluate_massive_recipients.py"
```

**Portée exacte.** MASSIVE annote des personnes et des relations, mais pas la
fonction de « destinataire » ni l'attribution d'un budget ou d'une action dans
une réunion. Le destinataire attendu des 17 exemples est déduit de la
position du segment annoté dans la demande d'e-mail. Ce contrôle teste donc
les destinataires nommés et les abstentions ; il ne mesure pas une résolution
positive de rôle EF-09 fondée sur une responsabilité récente. Il n'y a pas
d'annuaire dans le projet, et les adresses restent `null`.

Le jeu public [Enron People Assignment](https://github.com/RevRameshkumar/EPADataset)
contient des responsables de tâches annotés dans des e-mails authentiques,
mais ses phrases de tâches ne sont pas des demandes de brouillon adressées à
l'agent et ses exemples n'apportent pas directement la date d'attribution
exigée par EF-09. Les convertir en tests de résolution positive sans inventer
une demande ou une date changerait la tâche évaluée.
