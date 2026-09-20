# Qwen-Endstand: cells10-json-20260918

Lokal abgerufen, geprüft und ausgewertet am 20.09.2026 auf `master`.
Pre-Result-Freeze: `778d043`; Ausgangsstand der Auswertung: `faf48b2`.
Protokoll, Baselines, Generierung und Evaluationsregeln wurden nicht geändert.

## Abschluss und Vollständigkeit

| Konfiguration | Planned | Found | Valid | Invalid | Missing |
| --- | ---: | ---: | ---: | ---: | ---: |
| qwen_thinking | 840 | 840 | 840 | 0 | 0 |
| qwen_nonthinking | 840 | 840 | 840 | 0 | 0 |
| Gesamt | 1680 | 1680 | 1680 | 0 | 0 |

Alle 18 Array-Tasks der Jobkette `7035301`, `7035302`, `7035303` und
Archivjob `7035304`: `COMPLETED`, Exitcode `0:0`. Bereits Runde 1 erzeugte
alle Antworten; Runde 2 und 3 meldeten jeweils 280 vorhandene und 0 offene
Requests pro Shard. Sie führten keine erneuten Generierungen aus.
Archivjob beendet: 18.09.2026 11:38:17 (Zeitdarstellung von Slurm).
Bei der Abschlussprüfung waren keine eigenen Slurm-Jobs mehr aktiv.

Je Modus/Wiederholung liegen 280 Antworten vor, je Arm insgesamt 420.
Es gibt 1680 passende Attempt-Dateien, keine doppelten oder fremden Request-IDs,
keine fehlenden Antworten, technischen Endfehler, Tokenlimits, leeren Endantworten
oder ungeschlossenen Reasoning-Blöcke. Alle Endzustände sind `model_end`.
Es wurde kein JSON-Teilstring extrahiert und keine Antwort durch eine Referenz ersetzt.

| Modus | Outputtokens gesamt | Median pro Antwort | Maximum |
| --- | ---: | ---: | ---: |
| Thinking | 7958399 | 8583 | 43572 |
| Non-thinking | 46445 | 55 | 63 |

Inputtokenzählungen stimmen zwischen Engine, Admission und den 560 archivierten
gerenderten Prompts überein. Non-thinking liefert kurze direkte JSON-Schätzungen;
formale Gültigkeit bedeutet hier nicht geringe Schätzfehler.

## Ergebnisse auf den sechs realen Quellen

MAE₂ und ProfileMAE sind bedingt auf gültige Antworten, mit gleichen
Quellengewichten. Hier beträgt die Validität in jeder Zelle 100 %.
Je Arm und Modus: 90 Antworten. Angaben als Schätzwert ± MCSE, keine
Generalisierungs-Konfidenzintervalle über andere Netzwerke.

| Arm | Modus | MAE₂ ± MCSE | ProfileMAE ± MCSE |
| --- | --- | ---: | ---: |
| R | Thinking | 0.018945 ± 0.002453 | 0.012618 ± 0.001219 |
| R | Non-thinking | 0.533215 ± 0.008812 | 0.431313 ± 0.011236 |
| S | Thinking | 0.071300 ± 0.011124 | 0.045461 ± 0.008503 |
| S | Non-thinking | 0.512249 ± 0.011746 | 0.413141 ± 0.014289 |
| H | Thinking | 0.130172 ± 0.005525 | 0.121170 ± 0.013559 |
| H | Non-thinking | 0.437926 ± 0.016142 | 0.328399 ± 0.014933 |
| B | Thinking | 0.212786 ± 0.013266 | 0.112909 ± 0.011320 |
| B | Non-thinking | 0.560971 ± 0.009856 | 0.476848 ± 0.015847 |

Die festgelegten primären Referenzen erreichen MAE₂ R=0.017521, S=0.030790,
H=0.034492, B=0.076531; die gepoolten ExtraTrees R=0.030089, S=0.040005,
H=0.041732, B=0.063564. Thinking ist auf diesem realen Panel in beiden
Fehlermaßen durchgehend besser als Non-thinking, im MAE₂ aber in keinem Arm
besser als die primäre Referenz. Die Unterschiede sind besonders bei H/B groß.
Die Modi unterscheiden sich auch in Samplingparametern; dies ist kein isolierter
kausaler Reasoning-Vergleich.

Die synthetischen Strata werden separat ausgewiesen. Eine pauschale Rangfolge
über alle Strata wäre irreführend: Bei B liegt Non-thinking für `dar_a08`
(MAE₂ 0.171278 gegenüber 0.413259) und `ad_memory` (0.103844 gegenüber 0.502345)
vor Thinking. Bei `dar_a08`/H hat Thinking den kleineren MAE₂, aber den größeren
ProfileMAE (0.205143 gegenüber 0.119588).

Alle 40 Zellen einschließlich MCSE und gepaarten Referenzdifferenzen:
[summary.csv](results/cells10_json_20260918_qwen/summary.csv).
Quellenwerte: [source_results.csv](results/cells10_json_20260918_qwen/source_results.csv).
Kein zusätzliches, nach Ergebnissichtung definiertes Gesamtranking.

## Provenienz und Audit

Clusterquelle:
`uc3:/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot/cells10_json_20260918/mainexp/archive/`.
Lokales Roharchiv, Sammlung und vollständige Evaluation:
`results/main_experiment/cells10_json_20260918_qwen/` (Git-ignorierte Bulk-Artefakte).
Die ursprünglichen Clusterdateien und ihre Prüfsummen bleiben unverändert erhalten.

Alle 3680 Dateien aus `CHECKSUMS.json` wurden lokal nachgehasht, ebenso die 307
Dateien des ursprünglichen Bundles. Die Archivzahl „1681 answers“ zählt auch
`answers/engine_inputs.json`; tatsächlich sind es genau 1680 Antwortdateien.
Requestmanifest und Beobachtungen stimmen mit der lokalen Offline-Vorbereitung
überein. Prompt-/Payload-/Runner-Hashes und Seeds wurden für jede Antwort geprüft.
Alle 40 MAE₂-/ProfileMAE-Zellwerte wurden unabhängig aus Roh-JSON und Truth
mit gleichen Quellengewichten nachgerechnet.

`SPEC_COMMIT` im Bundle nennt `d74f2dc13494f2556d633917634d86b4f32b907e`:
Das Bundle wurde aus einem damals uncommitteten Arbeitsbaum erstellt.
Der Commitname allein wäre deshalb kein korrekter Freeze-Nachweis. Alle 18
gebündelten Experiment-Quelldateien einschließlich Runner stimmen per SHA-256
mit `778d043` und dem aktuellen Arbeitsbaum überein. Die ursprüngliche
`WORKTREE_STATUS` und Bundle-Prüfsummen sind lokal gesichert und im Audit gebunden.

Modell: Qwen3.6-35B-A3B, gepinnte Revision
`995ad96eacd98c81ed38be0c5b274b04031597b0`. Die Engine-Bindung stimmt mit den
archivierten Hashes aller 26 Gewichtsshards überein; Konfiguration, Tokenizer und
Chat-Template wurden zusätzlich vom Cluster abgerufen und nachgehasht.
Die Gewichte selbst wurden nicht lokal kopiert.
Primäre Baseline-SHA-256:
`62747ece39b63dbc8347790d791338bf59d3358f12a613cfc187ff973aecf544`.

Kompakte versionierte Evidenz: [audit.json](results/cells10_json_20260918_qwen/audit.json),
[Antwortmanifest](results/cells10_json_20260918_qwen/answer_manifest.csv),
[Prüfsummen](results/cells10_json_20260918_qwen/checksums.json).
Die vollständigen Rohantworten bleiben lokal; der Push enthält Audit und Ergebnisse.

## Verifikation und Wiederholung

125 Tests in `tests/frozen_main`: erfolgreich, keine Skips.
Die Testanpassungen beheben nach dem Cleanup veraltete Runner-Importe und stellen
den Evaluationstest von historischen auf aktuelle Offline-Fixtures um.
Experimentcode und methodische Regeln bleiben unverändert.
Zusätzlich erfolgreich: `verify_main_offline.py`, `verify_protocol_revision.py`,
Mock-Integration mit 3360 Requests und unabhängiger Qwen-Ergebnisaudit.
[Testprotokoll](results/cells10_json_20260918_qwen/tests.txt).

Sammlung und Evaluation: Befehle im [Runbook](RUNBOOK_JSON_REVISION_20260918.md).
Kompakten Audit aus den lokal gesicherten Artefakten erneut erstellen:

```bash
.venv/bin/python scripts/audit_qwen_results.py \
  --run results/main_experiment/cells10_json_20260918 \
  --archive results/main_experiment/cells10_json_20260918_qwen \
  --baselines results/baseline_revision_cells10_json_20260918/primary_baselines.json \
  --out docs/results/cells10_json_20260918_qwen
```

Qwen ist vollständig abgeschlossen. Das Vier-Konfigurationen-Hauptexperiment
bleibt unvollständig: GPT/Sol und DeepSeek wurden nicht gestartet, ihre insgesamt
1680 geplanten Requests sind weiterhin offen. Daher ist
`complete_main_result=false` im vollständigen Evaluationsbericht korrekt.
