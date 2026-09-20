# Durchführung der Revision cells10-json-20260918

Maßgebend ist [die Protokollrevision](PROTOCOL_REVISION_20260918.md).
Qwen-Endstand vom 20.09.2026: **1680/1680 Antworten gültig**, beide Modi fertig.
Siehe [Ergebnisse, Provenienz und Tests](QWEN_RESULTS_JSON_20260918.md).
Die unten beschriebenen Generierungsschritte sind bereits abgeschlossen.
Die alten Ergebnisse unter `cells10_20260917` bleiben historische Ergebnisse.
Die Word-Datei ist für die dort ersetzten Regeln überholt; insbesondere entfällt
„ungültige Antwort → Plug-in“ vollständig aus der LLM-Auswertung.

## Offline neu aufbauen

```bash
bash scripts/run_json_revision_offline.sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/frozen_main -v
```

Neue Artefakte:

- Hauptvorbereitung: `results/main_experiment/cells10_json_20260918`
- Pool und Fits: `results/baseline_revision_cells10_json_20260918`
- Protokolle: `results/json_revision_20260918_logs`
- API-Ledger: `results/main_experiment/cells10_json_20260918_api`
- Explizite Testantworten: `results/main_experiment/cells10_json_20260918_mock_revision`

Veränderliche API-Zustände und Mock-Auswertungen liegen außerhalb des mit
Prüfsummen gebundenen Offline-Laufs. Ein späteres Ergebnis verändert damit nicht
dessen geprüfte Eingabeartefakte.

Es werden 500 neue Poolgraphen, 14 Referenzfits (sieben LOSO-/Synthetikfolds,
mit und ohne Pool) und die festgelegten Referenzvorhersagen erstellt. Die
Hauptvorbereitung enthält zusätzlich ihre sieben nur auf Realdaten trainierten
Kontrollfits. Die realen Archive, synthetischen Hauptgraphen und Beobachtungen
sollen identisch zum vorherigen Design sein; Request-IDs und Generierungs-Seeds
sind neu. Nach einer Codeänderung einen neuen versionierten Lauf anlegen, keine
Bindings oder `.sha256`-Dateien löschen, um einen Resume zu erzwingen.

## Qwen: neue Generierung erforderlich

Beide alten Qwen-Modi sind wegen der alten aufgabenspezifischen Grammatik nicht
für den neuen Hauptvergleich verwendbar. Ein neues Clusterverzeichnis verwenden.
Der Bundle-Befehl nimmt den tatsächlichen Arbeitsbaum samt Hashes mit; ein
Commit-Name allein ist ausdrücklich keine Identitätsgarantie.

Vor der Hauptgenerierung einen technischen Smoke mit separaten, nicht zum
Hauptpanel gehörenden Eingaben durchführen. Zu prüfen sind in beiden Modi:
Ende des Reasonings, allgemeiner JSON-Modus, Tokenzählung und unveränderte
Samplingparameter. Keine Auswahl nach Schätzfehlern. Der lokale Konstruktor
`StructuredOutputsParams(json_object=True)` allein beweist noch keinen korrekten
vollständigen GPU-Lauf.

```bash
# Erst nach abgeschlossenem technischem Smoke:
bash scripts/cluster_bundle.sh cells10_json_20260918 results/main_experiment/cells10_json_20260918
```

Der Runner benötigt die festgelegten Bibliotheksversionen. Er bindet Requestdatei,
Runner, Modell-/Tokenizerdateien, Generierungseinstellungen und Shardzahl. Eine
vor dem Enqueue gespeicherte `.attempt` verhindert eine zweite Generierung nach
Prozessabbruch. Beim Resume wird ein begonnener, nicht abgeschlossener Request
als technischer Ausfall abgeschlossen; nur unbegonnene Requests werden zugelassen.
Die Slurm-Folgerunden sind damit **keine Wiederholungen fehlgeschlagener Antworten**.
Die vorhandene Zwei-Stunden-Jobgrenze ist ein zusätzlicher technischer Grenzwert
von Qwen. Ausfälle deswegen separat berichten, nicht als fehlende Modellfähigkeit
interpretieren. Kein Code repariert eine unvollständige Antwort.

Nach dem Abruf der neuen Antwortdateien:

```bash
.venv/bin/python scripts/collect_qwen_answers.py \
  --run results/main_experiment/cells10_json_20260918 \
  --answers results/main_experiment/cells10_json_20260918_qwen/answers \
  --out results/main_experiment/cells10_json_20260918_qwen/responses.jsonl
.venv/bin/python scripts/evaluate_main_responses.py \
  --run results/main_experiment/cells10_json_20260918 \
  --baselines results/baseline_revision_cells10_json_20260918/primary_baselines.json \
  --responses results/main_experiment/cells10_json_20260918_qwen/responses.jsonl \
  --out results/main_experiment/cells10_json_20260918_qwen/evaluation
```

Bei noch fehlenden API-Konfigurationen bleibt der Gesamtbericht unvollständig;
abgeschlossene Qwen-Zellen werden dennoch regulär ausgewiesen. `--extract-trailing-json`
ist in der neuen Auswertung nicht zulässig.

## GPT/DeepSeek: vorbereitet, nicht freigegeben

`run_main_api.py prepare` erzeugt ausschließlich ein lokales Ledger und eine
Vorlage mit `authorized: false`. Auch `status` und `export` arbeiten ohne Netzwerk.
Die Anwendung liest API-Schlüssel erst nach erfolgreicher Freigabeprüfung.

Der spätere Release muss die Hashes des neuen Request- und Referenzartefakts,
bestätigte zurückgelieferte Modellkennungen, aktuellen Preisstand, Ablaufzeit und
prüfbare Belege eines technischen API-Smokes auf separaten Eingaben enthalten.
Diesen Smoke hat die Revision entsprechend dem Nutzerauftrag **nicht ausgeführt**.
Die Kontoverfügbarkeit und Annahme aller Parameter sind daher noch nicht live
bestätigt. Die Preisprüfung darf höchstens 24 Stunden alt sein; ein erneuter
Zeitstempel darf nur nach wirklicher Prüfung eingetragen werden.

Der Runner verwendet Responses Batch für Sol (höchstens 64 Requests pro Aufruf)
und sequenzielles SSE für DeepSeek. Ein einziger Generierungsversuch pro Request,
keine SDK- oder selbstgebauten Generierungs-Retries. Fortsetzung geschieht durch
erneutes Starten desselben Befehls im selben gebundenen Dispatch-Verzeichnis.
Für Sol ruft `collect` existierende Batches ab und gleicht unklare Create-Antworten
über die gespeicherte Batch-Metadatenkennung ab. Ohne eindeutigen Treffer bleibt
die Reservierung bestehen; es wird kein neuer Batch erzeugt. Bei DeepSeek kann
ein Prozessabbruch aus dem gespeicherten SSE rekonstruiert werden. Unklare Kosten
blockieren weitere Requests dieser Konfiguration; sie sind extern abzugleichen.
Keine Reservierung auf Verdacht freigeben und keinen neuen Ledger anlegen, um
Kosten- oder Wiederholungsregeln zu umgehen.

Die konservativ abgerechneten Beträge ignorieren Cache-Rabatte und mögliche
DeepSeek-Nebenzeitrabatte. Die Reservierung beträgt 1,30 USD pro Sol-Request
beziehungsweise 0,48 USD pro DeepSeek-Request. Primärgrenzen: 180 bzw. 50 USD.
Der frühere Sol-Retry-Topf wird bei genau einem Versuch nicht gebraucht.
Ein Kostenstopp wird als unvollständige Durchführung dokumentiert.

## Ergebnisdarstellung

`summary.csv` enthält `valid_fraction`, `AE2` und `ProfileAE` samt MCSE.
`AE2` ist immer **bedingt auf gültige Antworten**, niemals eine ersetzte Systemleistung.
Spalten `*_all` beschreiben Referenzen über alle Beobachtungen; `*_matched` und
`delta_*` verwenden genau die gültigen LLM-Fälle. `source_results.csv` zeigt die
Quellenwerte. `answer_errors.csv` hält jeden geplanten Request einschließlich
fehlender/ungültiger Ergebnisse fest. `report.json` unterscheidet Vollständigkeit
von definierter Genauigkeit und bindet die verwendete Baseline-Datei per SHA-256.

Bei unterschiedlicher Validität keine alleinige MAE-Rangliste als allgemeine
Überlegenheit interpretieren. Die Unsicherheit ist bedingt auf das feste Panel;
keine additiven Modell-/Sampleranteile und keine Generalisierungs-Konfidenzintervalle
über alle temporalen Netzwerke daraus ableiten.

Für die abschließende gemeinsame Auswertung die Qwen-Sammlung und den API-Export
in eine neue JSONL-Datei zusammenführen und diese mit demselben Evaluator auswerten.
Doppelte IDs, fehlende Prompt-/Nutzlasthashes oder alte Request-IDs werden abgewiesen.
Ein neues Auswertungsverzeichnis verwenden; eine vorhandene Auswertung akzeptiert
keine nachträglich geänderten Eingaben.
