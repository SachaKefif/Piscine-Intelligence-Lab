# Contrôle ponctuel des preuves QMSum

Contrôle réalisé le 28 octobre 2026 sur six questions **train/validation**, une par domaine et split. Pour chaque cas, la question, la réponse QMSum, les tours du `relevant_text_span` et le passage de dix tours correspondant ont été lus. Les exemples ont été choisis avec une graine 42 parmi les questions à un seul passage positif ; le domaine Academic de validation n'en ayant pas dans cette sélection, un exemple à deux passages a été pris. Aucun score test n'a été utilisé pour cette revue.

| Split / domaine | Question | Observation |
|---|---|---|
| Train / Academic | `Bed004:5` | Les tours 220–224 contiennent l'explication de Grad D sur les types d'action et les objets ; le passage `220-229` couvre le sujet. |
| Train / Committee | `covid_3:4` | Le tour 138 donne la réponse du ministre Bill Blair sur la restriction des déplacements ; le passage `130-139` le contient. |
| Train / Product | `ES2015d:1` | Le tour 214 décrit la méthode d'évaluation du prototype par Marketing ; le passage `210-219` couvre la discussion. |
| Validation / Academic | `Bro022:1` | Les tours 24–39 portent sur la proposition et l'examen de Grad F ; deux passages (`20-29`, `30-39`) se partagent les éléments de réponse. |
| Validation / Committee | `education_0:10` | Le tour 83 contient l'explication de Julie Morgan sur le sondage auprès des enfants ; le passage `80-89` le contient. |
| Validation / Product | `ES2006a:1` | Le tour 233 contient le prix, le coût maximal et le profit visé ; le passage `230-239` le contient. |

**Résultat :** 6/6 plages de tours inspectées correspondent au thème et à la personne de la question. Ce contrôle ne prouve pas que chaque passage marqué positif suffit à répondre : un span peut s'étendre sur plusieurs passages, comme `Bro022:1`, ou être tronqué par l'encodeur. Il ne remplace pas une double annotation humaine des unités, statuts et relations à produire pour la suite du projet.
