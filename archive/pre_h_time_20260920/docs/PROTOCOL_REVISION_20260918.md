# Protokollrevision vom 18.09.2026

Status: Implementierung und Offline-Prüfung; **keine Freigabe für API-Aufrufe**.
Diese Revision ersetzt für neue Hauptläufe die entsprechenden Regeln von
`cells10-20260917` und dem Word-Designfreeze. Historische Artefakte bleiben erhalten.
Anlass sind die Prüfung vom 18.09.2026 und der anschließende ausdrückliche Auftrag,
die Befunde 2–6 zu beheben, ohne ungültige Modellantworten durch Referenzen zu ersetzen.
Die Regeln hier wurden vor Erzeugung des neuen Pools und neuer LLM-Antworten festgelegt.
Die alten Qwen-Ergebnisse und Referenzleistungen waren dabei bereits bekannt;
dies ist eine offen dokumentierte Designrevision, keine rückwirkende Präregistrierung.

## Festgelegte Änderungen und Folgen

1. **Keine Varianzzerlegung.** Der MCSE berücksichtigt die Hierarchie Quelle →
   Samplerziehung → Modellwiederholung. Die geschätzten Modell-/Sampleranteile
   werden gestrichen. Fünf Ziehungen erlauben nur eine grobe Unsicherheitsschätzung.
   Die drei Modellwiederholungen bleiben sinnvoll: Sie erfassen stochastische
   Antwortvariation und Ausfälle, sind aber keine zusätzlichen unabhängigen Quellen.
2. **Neuer synthetischer Pool.** 200 DAR- und 200 AD-Trainingsgraphen sowie je 50
   Entwicklungsgraphen, bisherige Parameterbereiche und Strata. N ist in jeder
   Familie und Partition gleich verteilt. Bei AD sind N und Rundenzahl in jeder
   Partition auch gemeinsam gleich verteilt. Die Zuweisung erfolgt ohne Labels.
   Neue versionierte IDs/Seeds; alle Fits werden neu erstellt. Entwicklungsergebnisse
   sind eine Validierung, kein Anlass, die bereits festgelegten Referenzen auszutauschen.
3. **Allgemeiner JSON-Modus für alle vier Konfigurationen.** Keine Qwen-spezifische
   Grammatik für Schlüssel, Wertebereich, Reihenfolge oder Dezimalstellen. Gleicher
   Prompt und gleiche nachgelagerte Validierung. Anbieterimplementierungen und
   Decodingparameter bleiben unterschiedlich; verglichen werden Konfigurationen,
   kein isolierter kausaler Effekt des Reasonings. Beide Qwen-Modi müssen neu laufen.
4. **Kein Ersatzwert für LLM-Ausfälle.** Ungültige Antworten, Verweigerungen,
   technische Endfehler und leere Eingaben erhalten keine Vorhersage. Kein Plug-in,
   Median, Clipping, Nachfragen oder Herauslösen eines günstigen JSON-Teilstrings.
   Ein einzelner umschließender Markdown-Fence darf weiterhin entfernt werden;
   das wird ausgewiesen. Eine formal gültige Endantwort am Tokenlimit bleibt gültig,
   sofern der Transport vollständig ist; das Limit wird zusätzlich ausgewiesen.
5. **Zuverlässigkeit und bedingte Genauigkeit gemeinsam berichten.** Primär:
   Anteil gültiger Schätzungen an allen geplanten Wiederholungen und MAE₂ der
   gültigen Antworten; Profil-MAE und gepaarte Referenzdifferenzen ergänzend.
   Quellen zählen gleich. Innerhalb einer Quelle zählt jede gültige Antwort gleich.
   Jede Referenzdifferenz verwendet exakt dieselben gültigen Fälle wie das LLM.
   Zusätzlich werden Referenzfehler auf allen Beobachtungen berichtet.
   Bei fehlender gültiger Antwort einer Quelle ist der gesamte Panel-MAE undefiniert;
   vorhandene Quellen werden nicht stillschweigend höher gewichtet. Bei 100 %
   Ausfall stehen 0 % gültige Antworten und kein MAE. Ein vollständig abgearbeiteter
   Lauf kann somit vollständig dokumentiert sein, ohne einen Genauigkeitswert zu haben.
   Unbegonnene/offene Requests sind keine beobachteten Modellfehler: Solange solche
   fehlen, gibt es nur Fortschrittszahlen, keine fertige Panelmetrik.
   Ein geringerer bedingter MAE bei geringerer Zuverlässigkeit rechtfertigt keine
   allgemeine Überlegenheitsbehauptung. Keine künstliche Strafzahl oder Gesamtrangliste.
6. **Dauerhafte, gebundene Durchführung.** Neue Request-IDs; Hashbindung von Prompt,
   kompletter Nutzlast, Protokoll, Runner und Referenzartefakten. Reservierungen,
   Rohdaten, Usage und Endzustände werden vor/bei Ausführung dauerhaft gespeichert.
   Unklare Abbrüche werden nicht blind erneut verschickt. Zum einfachen, für alle
   Konfigurationen gleichen Primärprotokoll gehört genau ein Generierungsversuch
   je logischem Request: keine automatischen Wiederholungen von Modellaufrufen.
   Lesendes Abrufen eines bestehenden Batch und lokale Rekonstruktion gespeicherter
   Rohdaten sind keine neuen Generierungsversuche. Kostenstopp bedeutet unvollständig.

## Unsicherheit und Gewichtung

Für Quelle g und Ziehung s seien X_gs die Summe der gültigen Fehler und C_gs die
Zahl gültiger Antworten. Der bedingte Fehler ist μ_g = Σ_s X_gs / Σ_s C_gs.
Für S > 1 wird die Varianz des Verhältnisschätzers mit
`S/(S-1) * Σ_s (X_gs - μ_g C_gs)² / (Σ_s C_gs)²` geschätzt.
Bei vollständig gültigen Antworten reduziert sich das exakt auf
`Var(Mittelwerte je Ziehung, ddof=1)/S`. Ziehungen ohne gültige Antwort gehen mit
X=C=0 ein. Bei weniger als zwei Ziehungen mit gültigen Antworten wird kein
Genauigkeits-MCSE berichtet. Bei einer nachweislich deterministischen Ziehung
wird nur die Variation zwischen den Modellwiederholungen verwendet; mindestens
zwei gültige Antworten sind dafür nötig. Sind alle Referenzwerte deterministisch,
ist deren MCSE null. Die Zuverlässigkeit wird analog aus den Validitätsindikatoren
über alle geplanten Antworten berechnet. Der Panelwert ist das ungewichtete
Mittel der Quellenwerte; die Varianzen werden summiert und durch G² geteilt.
Alle Angaben sind bedingt auf dieses Quellenpanel, Training und Kalibrierung.

## Was diese Revision bewusst beibehält

Zielgröße, W=5, sechs reale Testquellen, acht synthetische Hauptinstanzen, R/S/H/B,
10-%-Matching auf aktive Dyade-Fenster, Sampler-Seeds, Beobachtungen und Prompt
bleiben gleich. Neue zufällige Beobachtungen würden die Fehler nicht beheben und
keine neuen unabhängigen Testquellen schaffen. Die neue Laufidentität unterscheidet
Generierungsprotokoll und Pool; alte Antworten passen nicht zu neuen Request-IDs.

Statistische Referenzen und ihre Auswahl werden nicht anhand des neuen Pools
optimiert: R Plug-in, S Walk-Korrektor, H Schrankenmittelpunkt, B festgelegter
Mischungskorrektor, dazu ExtraTrees mit unveränderten Hyperparametern/Gewichten und
LOSO-Ausschlüssen. Der bereits definierte numerische Rückfall des B-Schätzers auf
seinen homogenen Spezialfall bleibt Teil dieses ausdrücklich zusammengesetzten
Referenzverfahrens und wird in den Ergebnissen ausgewiesen. Er liefert niemals
eine Ersatzantwort für ein LLM. Post-hoc-Shrinkage und Median-von-drei-Ranglisten
gehören nicht zur neuen Hauptauswertung.

## Technische Quellen

- [OpenAI JSON-Modus / Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [OpenAI Batch](https://developers.openai.com/api/docs/guides/batch)
- [GPT-5.6 Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol)
- [DeepSeek Chat Completion](https://api-docs.deepseek.com/api/create-chat-completion/)
- [DeepSeek JSON-Modus](https://api-docs.deepseek.com/guides/json_mode/)
- [vLLM 0.29 Structured Outputs](https://docs.vllm.ai/en/v0.29.0/features/structured_outputs/)

Dokumentationsprüfung und Offline-Transporttests können Kontozugang, tatsächlich
bediente Modellversion und einen echten API-Smoke nicht beweisen. Solche Aufrufe
bleiben gemäß Auftrag aus. Vor dem späteren Hauptlauf sind Preis-/Versionsprüfung
und ein technischer Smoke auf separaten Eingaben nötig; keine Ergebnisoptimierung.
