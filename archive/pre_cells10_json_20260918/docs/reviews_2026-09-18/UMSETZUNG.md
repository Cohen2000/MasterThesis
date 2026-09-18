# Umsetzung der beauftragten Korrekturen 2–6

Neue Version: **cells10-json-20260918**. Maßgebend sind die
[Protokollrevision](../../PROTOCOL_REVISION_20260918.md) und das
[Runbook](../../RUNBOOK_JSON_REVISION_20260918.md).

Die Regeln wurden vor dem neuen Pool und neuen LLM-Antworten festgelegt. Die
vorherigen Ergebnisse waren bekannt. Die Revision wird deshalb als solche
benannt; sie macht aus dem vorhandenen Quellenpanel kein unberührtes Testpanel.

| Befund | Umgesetzte Lösung | Konsequenz |
|---|---|---|
| 2: widersprüchliche Varianzkomponenten | Nur direkter, über Samplerziehungen geclusterter MCSE; keine Modell-/Samplerzerlegung | Drei Antworten dienen weiterhin der Erfassung stochastischer Variation, nicht als zusätzliche unabhängige Quellen |
| 3: Poolaufteilung verschiebt N/Rundenzahl | Neue versionierte Seeds; innerhalb jeder Partition unabhängig gemischte, ausgeglichene Parametergitter | 500 neue Poolgraphen; sämtliche Fits neu; Auswahl der Referenzverfahren und Hyperparameter bleibt fest |
| 4: Qwen erhält stärkere Ausgabehilfe | Allgemeiner JSON-Modus für alle vier Konfigurationen; gleiche strikte Nachvalidierung | Beide Qwen-Modi benötigen neue Antworten; alte Antworten bleiben historische Daten |
| 5: ungültige Antworten / Ersatzscore | Keine Ersatzvorhersage; Validitätsanteil und bedingter MAE gemeinsam berichten | Bei vollständigem Ausfall: 0 % gültig und kein MAE; keine künstliche Strafzahl |
| 6: fehlender API-Produktionspfad | Sol-Batch und DeepSeek-SSE, persistentes Ledger, Rohdaten, Kostenreserven, Hashbindungen und Wiederanlauf ohne zweiten Generierungsversuch | Implementiert und mit Transportattrappen prüfbar; echter API-Smoke und Freigabe stehen bewusst noch aus |

Die Genauigkeit wird je Quelle über gültige Antworten gemittelt, anschließend
über die festgelegten Quellen gleich gewichtet. Die Vergleichsreferenz verwendet
für jede gepaarte Differenz exakt dieselben Fälle. Fällt eine komplette Quelle
aus, bleibt der Panel-MAE undefiniert. Zusätzlich stehen die Referenzfehler auf
allen Beobachtungen bereit. Das verhindert, dass eine Konfiguration allein durch
Auslassen schwieriger Fälle als allgemein überlegen dargestellt wird. Einen
solchen Schluss darf man auch künftig nicht allein aus dem bedingten MAE ziehen.

Unbegonnene und offene Requests zählen als unvollständige Durchführung. Erst nach
Abschluss aller vorgesehenen Versuche ist die Validitätsquote einer Zelle ein
fertiges Ergebnis. Technische Fehler, Tokenlimits und leere Eingaben sind getrennt
sichtbar. Eine vollständige Durchführung kann somit ohne definierte Genauigkeit
enden. Das ist ein interpretierbares Ergebnis, keine Lücke, die mit einer fremden
Schätzung geschlossen werden muss.

Die Hauptgraphen, Sampler, Ziele, Budgets und Prompts bleiben gleich. Das hält den
Vergleich nachvollziehbar und verhindert zusätzliche Änderungen ohne methodischen
Nutzen. Die Referenzen bleiben R: Plug-in, S: Walk-Korrektor, H: Schrankenmittelpunkt,
B: vorab gewählter Mischungskorrektor, plus die gelernte Referenz. Der numerische
Rückfall des B-Referenzverfahrens auf seinen homogenen Spezialfall ist weiterhin
expliziter Bestandteil **dieses Referenzverfahrens** und wird ausgewiesen. Er wird
nirgends als LLM-Antwort verwendet. Die vorigen Post-hoc-Shrinkage- und
Median-von-drei-Tabellen gehören nicht zur neuen Hauptauswertung.

Die neue Durchführung verwendet genau einen Generierungsversuch pro Request.
Ein abgebrochener POST wird nicht blind wiederholt. Bei OpenAI wird ein eventuell
bereits angelegter Batch über seine gespeicherte Kennung gesucht und abgerufen.
DeepSeek speichert SSE einschließlich Reasoning und Usage fortlaufend. Unklare
Kosten behalten ihre Reservierung und blockieren weitere Aufrufe derselben
Konfiguration. Die Antwort darf auch nach Formatfehlern nicht nachgebessert werden.
Das Verfahren ist bewusst konservativ; technische Ausfälle werden nicht als
fehlende Fähigkeit des Modells interpretiert.

Zusätzlich wurden die betroffenen Wiederaufnahmepfade abgesichert: Pooldefinition
und Pooldateien, Entwicklungscache, vollständige Request-Nutzlast, Qwen-Runner und
Modellartefakte sowie die tatsächlich ausgewertete Referenzdatei. Das Clusterbundle
enthält den aktuellen Arbeitsbaum samt Prüfsummen. Historische Startskripte stoppen
mit Verweis auf das neue Protokoll, statt alte Ergebnisse mit neuem Code zu vermischen.

Die installierte Clusterumgebung akzeptiert `StructuredOutputsParams(json_object=True)`
unter vLLM 0.29.0. Das war ein Konstruktorcheck ohne GPU-Generierung. Ein vollständiger
Smoke beider Qwen-Modi und der API-Anbieter bleibt vor dem späteren Hauptlauf nötig.
Die unterschiedlichen Decodingparameter und technischen Ressourcen bleiben Teil
des Konfigurationsvergleichs; eine isolierte Wirkung des Reasonings wird nicht behauptet.

Abnahmeergebnisse und Prüfsummen stehen nach Abschluss in
[revision_acceptance.json](revision_acceptance.json). Alle neuen Hauptantworten
stehen bis dahin und bis zur gesonderten Ausführung aus. Es wurden keine API-Aufrufe
an Modellanbieter gestartet.
