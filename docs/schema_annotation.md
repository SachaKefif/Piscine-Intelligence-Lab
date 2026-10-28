# Schéma d'annotation des conversations

Version 1, 28 octobre 2026. Ce document définit les **annotations à produire** pour le moteur de mémoire. Les exports actuels `Data/Processed/*.jsonl` sont des transcriptions normalisées ; ils ne contiennent pas encore ces unités sémantiques ou relations validées. Les contrats machine sont [`semantic_unit.schema.json`](schemas/semantic_unit.schema.json) et [`relation.schema.json`](schemas/relation.schema.json).

## Unité et preuve

Une unité exprime **une seule** information vérifiable. Les identifiants sont stables : `source_id = corpus:meeting_id`, `unit_id = source_id:start_turn-end_turn:ordinal`. Conserver les indices de tours originaux, inclusifs, pour retrouver l'extrait exact.

| Champ | Règle |
|---|---|
| `unit_id` | Identifiant unique de l'unité. |
| `source_id`, `source_date` | Réunion d'origine et date ISO `YYYY-MM-DD`, ou `null` si inconnue. Ne jamais inventer une date. |
| `type` | `decision`, `action`, `commitment`, `question`, `answer`, `fact`, `risk` ou `idea`. |
| `statement` | Reformulation brève et fidèle, sans ajouter de personne, chiffre ou certitude. |
| `evidence` | Citation textuelle du ou des tours d'origine ; elle doit être retrouvable dans la transcription. |
| `start_turn`, `end_turn` | Bornes inclusives dans la réunion source, `0 ≤ start_turn ≤ end_turn`. |
| `status` | `proposed`, `confirmed`, `rejected`, `hypothetical`, `superseded`, `asserted` ou `unknown`. |
| `speaker`, `owner`, `due_date`, `project_id` | Valeur explicite dans la source, sinon `null`. `speaker` n'implique pas `owner`. |

**Décision :** `confirmed` exige une adoption explicite ou une confirmation attribuable. Une suggestion reste `idea`/`proposed` ; une hypothèse reste `hypothetical`. Une tâche promise par une personne est `commitment`, avec `owner` seulement si l'attribution est explicite. `asserted` signifie qu'une personne a affirmé un fait ; cela ne garantit pas sa vérité externe.

Pour une négation, conserver la polarité dans `statement`. Pour une date ou un nombre, copier exactement la valeur et son unité. Une nouvelle décision ne rend l'ancienne `superseded` qu'avec une preuve du changement et une chronologie fiable. Si `source_date` manque, ne pas inférer l'ordre entre réunions à partir de l'ordre des fichiers.

## Relations

Un lien possède `from_id`, `to_id`, `relation` et `evidence_source_id`. Les deux identifiants d'unité doivent exister. `evidence_source_id` indique la réunion où la relation est établie ; il peut différer des sources des deux unités.

| Relation | Test d'annotation |
|---|---|
| `supports` | La preuve d'une unité corrobore explicitement l'autre. |
| `contradicts` | Les deux affirmations portent sur le même sujet et ne peuvent être vraies ensemble dans le même contexte temporel. |
| `supersedes` | `from_id` est la nouvelle information qui remplace `to_id`, avec preuve du remplacement. |
| `creates_action` | `from_id` est une décision ou un engagement qui crée la tâche `to_id`. |
| `answers` | `from_id` est une réponse à la question `to_id`. |
| `unrelated` | Paire comparée sans relation démontrée ; ne pas confondre absence de preuve et contradiction. |

Une similarité d'embedding sert à proposer une paire candidate ; elle ne valide pas sa relation. Le lien doit citer une source. Les cas ambigus restent sans relation automatique et sont dirigés vers une vérification ou une clarification.

## Questions, pertinence et intentions

Pour QMSum, `query_id = meeting_id:index_specific_query`. Les `relevant_text_span` sont des intervalles de tours **inclusifs** fournis par le corpus. Le premier benchmark dérive `relevant_passage_ids` par tout chevauchement d'un passage avec un span. Stocker `label_origin = "derived_overlap"` : ce label est plus faible qu'une annotation humaine passage par passage. Les questions générales, sans spans, ne reçoivent pas ce label.

Les intentions futures ont les valeurs `answer`, `compare`, `build_dossier`, `draft_email`, `create_task`, `schedule_event`, `clarify`. Pour chaque demande, annoter séparément `subject`, `person`, `due_date`, `action`, chacun nullable. Le routage vers une action simulée ne suppose jamais que la personne mentionnée est la destinataire.

## Procédure de contrôle

1. Annoter d'abord les unités avec leur extrait, puis les statuts et les personnes, enfin les relations. Un champ non prouvé reste `null` ou `unknown`.
2. Vérifier automatiquement unicité des identifiants, bornes de tours, présence des champs, existence des cibles des relations et conformité aux deux schémas JSON.
3. Revoir manuellement un échantillon par domaine et split, en lisant le tour complet et la question. Noter séparément erreurs de span, de statut, de personne et de date ; arbitrer les désaccords avant de créer un jeu d'entraînement de relations.
4. Ne pas transformer les spans QMSum en décisions ou engagements sans nouvelle annotation humaine : QMSum fournit des preuves de pertinence pour des résumés, pas les statuts et relations requis par l'agent.

### Exemple fictif de contrat

```json
{
  "unit_id": "Demo:Atlas-02:14-14:0",
  "source_id": "Demo:Atlas-02",
  "source_date": "2026-10-20",
  "type": "commitment",
  "statement": "Camille enverra le dossier vendredi.",
  "evidence": "Camille : Je vous enverrai le dossier vendredi.",
  "start_turn": 14,
  "end_turn": 14,
  "status": "confirmed",
  "speaker": "Camille",
  "owner": "Camille",
  "due_date": null,
  "project_id": "Atlas"
}
```

`due_date` reste `null` dans cet exemple : « vendredi » n'a pas été résolu en date absolue. Il faudrait une date de réunion fiable et une règle explicite de résolution avant de remplir ce champ.
