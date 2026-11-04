# Checkpoint E5 d'inférence

Le paquet `qmsum-e5-small-checkpoint-842-inference.zip` contient le modèle E5
adapté utilisé pour le mode hybride. Il ne contient pas les états d'optimiseur,
le calendrier d'entraînement, l'état aléatoire ni les arguments du trainer.
Le modèle de base est [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)
(licence MIT affichée sur sa fiche) ; l'adaptation a été effectuée sur QMSum train.

Depuis la racine d'une copie propre :

```powershell
py -3.13 "Data Analysis\download_e5_checkpoint.py"
py -3.13 "Data Analysis\ingest_semantic_units.py"
py -3.13 "Data Analysis\evaluate_acceptance.py" --method hybrid --device cpu --metrics Data\Processed\recette_hybrid.json --outputs Data\Processed\recette_hybrid.jsonl
```

Installer auparavant `requirements.txt` dans l'environnement Python utilisé pour
le test. Le téléchargeur vérifie l'empreinte SHA-256 de l'archive et celle de
`model.safetensors` avant l'installation. Les valeurs et l'URL sont figées dans
[`inference_release.json`](../Models/qmsum-e5-small/inference_release.json).

Le paquet peut être reconstruit à partir du checkpoint d'entraînement local avec
`py -3.13 "Data Analysis\package_e5_checkpoint.py"`. L'archive est déterministe
pour ces fichiers : son SHA-256 doit être
`42d256e949320b7d05d7bf130605990c3a7c5a0f50329235ed89cd4ebbd6ed79`.
