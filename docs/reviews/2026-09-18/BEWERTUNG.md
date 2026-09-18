> Historischer Prüfbericht des unveränderten Standes `d74f2dc`. Die anschließend beauftragten Korrekturen sind in [PROTOCOL_REVISION_20260918.md](../../PROTOCOL_REVISION_20260918.md) dokumentiert. Reproduktionsskripte dieses Prüfberichts benötigen den damaligen Quellcode; sie sind keine Tests des revidierten Protokolls.

**Unabhängige Prüfung des Hauptversuchs am 18. September 2026**

Geprüft: `cells10-20260917`, Branch `experiment/offline-freeze-20260916`, Commit `d74f2dc13494f2556d633917634d86b4f32b907e`. Grundlage sind der lokale Code, Rohdaten und Ergebnisartefakte, das freigegebene Word-Dokument und die ergänzende Einschätzung. Letztere ist eine Referenz, keine Prüfvorschrift. Die Identitäten der Dokumente stehen in [provenance.json](provenance.json).

**Urteil: wissenschaftlich tragfähig, aber keine unveränderte Freigabe für die bezahlten Hauptläufe.** Die vorhandene Untersuchung kann eine gute, begrenzte Masterarbeit tragen. Ich habe keinen Fehler gefunden, der die bisherigen Zielwerte oder Qwen-MAE-Ergebnisse entwertet. Es gibt aber konkrete Auswertungs- und Validierungsprobleme sowie noch offene Entscheidungen zur Durchführung. Ich würde diese vor GPT/DeepSeek abschließen. Ein vollständiger Neustart oder neue Qwen-Generierungen sind dafür nach den vorliegenden Befunden nicht erforderlich.

„100 % sicher“ ist kein erreichbares wissenschaftliches Abnahmekriterium. Erreichbar sind: überprüfte Implementierung, vor weiteren Ergebnissen fixierte Regeln, verständliche Annahmen und Schlussfolgerungen, deren Reichweite zu den Daten passt. Ein positives LLM-Ergebnis ist dafür nicht notwendig.

**Was tatsächlich überprüft wurde**

| Prüfung | Ergebnis |
|---|---|
| Tests des aktuellen Hauptversuchs | 115 bestanden; keine Fehler oder übersprungenen Tests |
| Vorhandener Offline-Verifizierer | 1.537 Prüfsummen; alle 600 Beobachtungsblöcke aus Seeds erneut hergeleitet; 3.360 geplante Requests geprüft |
| Unabhängige Rekonstruktion aus den sechs realen Quelldateien | Eventzahlen, Dyadenzahlen und alle vier Zielkomponenten stimmen exakt mit den gespeicherten Werten überein |
| Rohdateien der 16 realen Trainingsquellen | Alle SHA-256-Werte stimmen mit den Manifesten überein |
| 14 Modelle: gepoolt und nur real, jeweils sieben Folds | Modellprüfsummen, Ausschluss der jeweiligen realen Testquelle, Ausschluss des Entwicklungspools und Vorhersagen auf allen Hauptbeobachtungen geprüft |
| 500 tatsächliche Poolartefakte | Parameter und Partition stimmen mit der gespeicherten sowie neu erzeugten Pooldefinition überein |
| Alle 1.680 Qwen-Rohantworten | Request, Prompt, Seed, Runner, Antwortbeschränkung, Reasoning-Abschluss und Tokenzählung geprüft; keine Abweichungen |
| Unabhängige Nachrechnung aus den Qwen-Antworten | Sämtliche AE₂, Profilfehler und Differenzen zum primären Korrektor stimmen bis auf Rundungspräzision mit der Auswertung überein |
| Qwen-Validität | 839/840 Thinking; 840/840 Non-thinking; die eine nichtmonotone Antwort korrekt durch Plug-in ersetzt |
| Cluster, lesend | 1.680 vorhanden, keine fehlenden/unerwarteten/doppelten Antworten, alle regulär beendet; Archivbericht: 1.990 Dateien, keine Rücklesefehler |
| Bindung des Offline-Laufs an den aktuellen Quellcode | Keine geänderten Hashes gegenüber `preparation_inputs.json` |

Belege: [audit_results.json](audit_results.json), [offline_verification.json](offline_verification.json), [unit_tests.log](unit_tests.log). Die zusätzliche Nachrechnung kann mit `.venv/bin/python docs/reviews/2026-09-18/reproduce_audit.py` wiederholt werden. Sie liest vorhandene Daten und Modelle, trainiert nichts und ruft kein LLM auf. Die Trainingsfits und GPU-Generierung wurden bei dieser Prüfung nicht vollständig von Grund auf wiederholt. Der Clusterarchivbericht wurde gelesen; das gesamte Archiv wurde nicht nochmals auf dem Cluster neu gebaut.

**1. Vor weiteren Ergebnissen: Untersuchungsgegenstand und Designgeschichte eindeutig festlegen**

Der stärkste allgemeine Einwand ist die Entwicklung am bereits bekannten Panel. Budget, Historienarm, gelernte Referenz und Ausgabesteuerung wurden nach früheren Ergebnissen verändert. Das Repository benennt dies offen. Das Word-Dokument klingt mit Aussagen wie „Parameter werden nicht anhand der späteren Testleistung angepasst“ und „vor der Testauswertung festgelegt“ dagegen stellenweise stärker prospektiv, als die Gesamtgeschichte trägt.

Neue Sampler-Seeds und neue Antworten auf denselben Archiven schaffen keine neuen, unberührten Testquellen. Der Ausschluss einer Quelle aus einem Trainingsfold verhindert deren direkte Nutzung im Fit; er hebt die menschliche beziehungsweise agentengestützte Designentwicklung mit Kenntnis dieser Quelle nicht auf. Das ist meine Anwendung des allgemeinen Auswahlbias-Problems auf diesen Versuch, nicht der Nachweis einer quantifizierten Verzerrung deiner Ergebnisse. [Cawley & Talbot](https://www.jmlr.org/papers/v11/cawley10a.html)

Ich würde die Arbeit als **iterativ entwickelten Benchmark mit anschließend fixierter vergleichender Auswertung** beschreiben. Eine kurze Chronologie muss nennen: Version, Änderung, bereits bekannte Ergebnisse, Zeitpunkt der Festlegung und neu erzeugte Artefakte. Besonders relevant sind die drei Budget-/H-/Ausgaberevisionen und die Erweiterung der gelernten Referenz. Die jetzigen GPT-/DeepSeek-Läufe können prospektiv unter dem letzten festgelegten Protokoll stattfinden; das gesamte Panel wird dadurch nicht nachträglich unabhängig.

Eine passende Forschungsfrage lautet:

> Wie genau schätzen ausgewählte LLM-Konfigurationen die fensterbasierte Wiederkehr von Dyaden im vollständigen Archiv aus standardisierten Zusammenfassungen partieller Beobachtungen, verglichen mit festgelegten statistischen und auf diese Aufgabenverteilung trainierten Referenzen?

Das untersucht statistische Inferenz aus Graphzusammenfassungen. Es untersucht weder allgemein Graphverständnis noch zukünftigen Beziehungserhalt. EstGraph liefert einen sinnvollen Bezug zur Eigenschaftsschätzung aus aufgabenspezifisch verdichteten Walk-Informationen. Daraus folgt keine Suffizienz deiner Darstellung. [Maurya & Liu](https://aclanthology.org/2026.acl-long.1846/)

Für diese begrenzte, explorative Benchmarkarbeit würde ich keinen neuen realen Datensatz erzwingen. Für eine starke Behauptung unabhängiger Bestätigung oder allgemeiner Überlegenheit wäre ein bisher unbenutztes, vorab festgelegtes externes Testpanel erforderlich. Das ist eine andere Reichweite der Arbeit.

**2. Konkreter Auswertungsfehler: Die Varianzkomponenten sind nicht additiv**

Fundstelle: [evaluation.py](../../../src/main_experiment/evaluation.py), Zeilen 73–92. Der Code berechnet die Gesamtvarianz des Mittelwerts aus fünf Samplemittelwerten. Den Modellanteil schätzt er separat aus der Streuung der drei Antworten innerhalb eines Samples. Den verbleibenden Sampleranteil setzt er bei negativen Werten auf null, belässt aber die Gesamtvarianz unverändert.

Dadurch gilt in solchen Fällen gerade nicht `Gesamtvarianz = Modellvarianz + Samplervarianz`. Nicht die Standardfehler selbst, sondern ihre Quadrate müssten sich bei einer additiven Darstellung ergänzen.

Das ist kein hypothetischer Randfall: Für AE₂ tritt eine negative Restschätzung in **25 von 48 realen Quelle–Arm–Qwen-Zellen** auf. Über beide geprüften Metriken, AE₂ und gepaarte AE₂-Differenz, und alle 14 Graphen sind es 103 von 224 Zellen. Beispiel der aggregierten Tabelle: Qwen Non-thinking in S hat MCSE 0,00866, der separat ausgewiesene Modellanteil allein aber MCSE 0,01361.

Die Hauptmittelwerte sind davon unberührt. Auch der direkte MCSE aus den Samplemittelwerten ist nicht allein deshalb falsch: Zwei aus wenigen Wiederholungen geschätzte Größen können widersprüchlich ausfallen. Falsch wäre die Interpretation als kohärente additive Erklärung der Unsicherheit.

Meine bevorzugte, kleine Korrektur: Den direkt geschätzten bedingten MCSE als Hauptangabe behalten; den ungekürzten Varianzrest und ein Negativ-Flag für die Diagnose speichern; die separate Komponentenschätzung ausdrücklich als instabil kennzeichnen und keine additiven Anteilsdiagramme daraus ableiten. Alternativ kann eine konsistente nichtnegative Varianzkomponentenschätzung spezifiziert werden. Sie darf nicht stillschweigend an die Stelle des bisherigen Schätzers treten. Danach die Tabellen aus denselben Antworten unter einer neuen Auswertungsversion erzeugen. Neue LLM-Aufrufe werden nicht benötigt.

Sechs gezielt ausgewählte Quellen bleiben sechs Quellen. Fünf Samples und drei Antworten erhöhen die Präzision innerhalb dieses Panels, nicht seine externe Repräsentativität. Der bestehende MCSE ist bedingt auf Quellen, Training, Modelle und Kalibrierung. Ein Standardfehler zwischen sechs Quellen sollte nicht als universelles Konfidenzintervall über temporale Netzwerke verkauft werden. [Morris, White & Crowther](https://arxiv.org/html/1712.03198v3)

**3. Neuer konkreter Befund: Der Entwicklungspool enthält einen systematischen Verteilungswechsel**

Fundstelle: [pool.py](../../../src/main_experiment/pool.py), Zeilen 107–135. Die Parameterzyklen und der Schnitt „erste Zeilen Training, letzte Zeilen Entwicklung“ laufen synchron. Die gespeicherten 500 Artefakte bestätigen folgende Verteilung:

| Familie/Parameter | Training | Entwicklung |
|---|---|---|
| DAR, Knotenzahl | 200/300/500 jeweils 50 Graphen; 800/1.200 jeweils 25 | ausschließlich 800 und 1.200, jeweils 25 |
| AD, Rundenzahl | 500/750/1.000/1.500 jeweils 50 Graphen | ausschließlich 2.000, insgesamt 50 |

Der Pool ist also sauber getrennt, aber keine gewöhnliche, hinsichtlich dieser Parameter gleich verteilte Trainings-/Entwicklungsaufteilung. Insbesondere sieht das AD-Training **keinen einzigen Graphen mit 2.000 Runden**. Bei AD beeinflusst die Rundenzahl die Entwicklung des Kontaktgedächtnisses und der aggregierten Beziehungen. Der Entwicklungsbefund kann deshalb nicht ohne Weiteres als repräsentative Prüfung über das gesamte angegebene Parameterraster interpretiert werden.

Das ist keine Testlabel-Leckage und macht die bestehenden Hauptvorhersagen nicht falsch. Die Hauptinstanzen N=500 und AD mit 1.000 Runden liegen in den trainierten Bereichen. Es ist aber ein tatsächlicher Mangel der beabsichtigten Poolabdeckung, der in den bisherigen Tests nicht auffällt: Dort werden Mengen, Seeds und Parametergrenzen geprüft, nicht die Abdeckung je Partition.

Ich würde die bestehenden Fits und Hauptreferenzen beibehalten und die tatsächliche Aufteilung offenlegen. Wenn die Entwicklungsergebnisse eine zentrale Begründung für die Auswahl des Korrektors tragen sollen, empfehle ich eine kleine zusätzliche, ausgewogene synthetische Prüfung mit neuen IDs und Seeds. Sie wäre nachträgliche Sensitivität; die Hauptreferenz darf danach nicht nach Testleistung ausgetauscht werden. Für künftige Poolversionen sollten N und Rundenzahl unabhängig von der Partitionszuweisung balanciert und genau diese Verteilungen getestet werden. Die alten Artefakte dürfen nicht umetikettiert werden.

**4. Vor den API-Läufen: Die Ausgabeunterstützung ist unterschiedlich stark**

Fundstellen: [requests.py](../../../src/main_experiment/requests.py), Zeilen 7–30; [common.py](../../../src/main_experiment/common.py), `ANSWER_REGEX`.

Qwen erhält eine Generierungsgrammatik mit vier bestimmten Schlüsseln, fester Reihenfolge, Werten in [0,1] und höchstens sechs Dezimalstellen. GPT und DeepSeek sind mit `json_object` geplant. Allgemeiner JSON-Modus garantiert syntaktisches JSON, aber nicht dieselben Schlüssel, Typen und Wertebereiche. OpenAI unterscheidet JSON-Modus ausdrücklich von schemaerzwungenen Structured Outputs. [Offizielle OpenAI-Dokumentation](https://developers.openai.com/api/docs/guides/structured-outputs), [DeepSeek JSON Output](https://api-docs.deepseek.com/guides/json_mode/)

Die Aussage „alle liefern JSON“ verdeckt deshalb eine relevante Asymmetrie. Diese kann die Häufigkeit des Plug-in-Rückfalls beeinflussen; dessen Einfluss auf den Fehler ist nicht zwingend immer günstig oder ungünstig. Die gemeinsame Endvalidierung bleibt sinnvoll, macht die Generierungsbedingungen aber nicht identisch.

Meine Empfehlung ist, den angekündigten **Vergleich vollständig definierter Konfigurationen** beizubehalten und diese Unterschiede vor den API-Ergebnissen explizit festzuschreiben. Dazu gehören eine kleine Tabelle der erzwungenen beziehungsweise nur angeforderten Antwortbedingungen und eine getrennte Ausweisung von Validität, Rückfallhäufigkeit, Systemscore und gültigen Rohantworten. Ein reiner Vergleich der Modellfähigkeiten unter identischer Unterstützung ist es damit nicht. Wenn stattdessen identische Ausgabeunterstützung Teil der Forschungsbehauptung sein soll, muss das Protokoll vor weiteren Hauptläufen harmonisiert und die betroffene Durchführung neu versioniert werden. Das wäre eine inhaltliche Revision, kein stiller technischer Fix.

Auch Qwen Thinking versus Non-thinking ist ein Konfigurationsvergleich: Thinking darf lange Zwischengenerierungen erzeugen; Non-thinking wird sofort auf das Zahlenobjekt beschränkt. Dazu kommen unterschiedliche Decodingparameter. Die Parameter entsprechen der offiziellen Modellkarte, beweisen aber keinen isolierten Kausaleffekt „des Denkens“. [Versionierte Qwen-Modellkarte](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/995ad96eacd98c81ed38be0c5b274b04031597b0/README.md)

**5. Zweite Auswertungskorrektur: Der Hauptscore verschwindet bei ausschließlich ungültigen Antworten**

Fundstelle: [evaluate_main_responses.py](../../../scripts/evaluate_main_responses.py), Zeilen 152–196, insbesondere `fallback_only` und `MODEL_METRICS`.

Ein ausdrücklich als Mock markierter Integrationstest setzt alle Sol-Antworten auf ungültig und die übrigen Konfigurationen auf gültige konstante Antworten. Jede ungültige Einzelantwort bekommt korrekt den Plug-in-Ersatz. Anschließend meldet die Auswertung `complete_main_result=true`, lässt den gesamten AE₂-Hauptscore für Sol aber leer. [Nachweis](all_invalid_mock.json)

Diese Unterdrückung ist im Code absichtlich eingebaut und sogar getestet. Sie verhindert zu Recht, dass ein reiner Plug-in-Wert als eigene Modellschätzung bezeichnet wird. Sie passt aber nicht zur angekündigten Primärmetrik **System aus LLM plus fester Rückfallregel**. Dieses System besitzt auch bei 100 % Rückfall einen definierten Fehler. Bei nur 99 % Rückfall wird dessen Mischwert schließlich ebenfalls berichtet.

Ich würde zwei eindeutig benannte Ergebnisebenen führen: den Systemscore immer bei vollständiger Durchführung und daneben die nur bei vorhandenen gültigen Antworten definierte Rohmodell-Auswertung. Bei ausschließlich ungültigen Antworten steht dann beispielsweise „System-MAE = Plug-in-MAE; 100 % Rückfall; keine gültige Modellschätzung“. Die bisherige Qwen-Tabelle ist durch diesen Randfall nicht betroffen. Vor zwei neuen Anbietern sollte die Regel aber durchgängig implementiert sein.

**6. Die API-Durchführung ist noch keine fertige, getestete Hauptpipeline**

`requests.py` enthält Nutzlasten sowie reine Funktionen für Retry, Reservierung und Watchdog. Das Modul sagt selbst „No network client is exposed“. Im aktiven Hauptversuch fehlen die produktiven Sol-/DeepSeek-Transporte mit dauerhaftem Ledger und Wiederaufnahme. Historische Clients unter `archive/` ersetzen diese Implementierung für den aktuellen Vertrag nicht.

Die wesentlichen geplanten Parameter sind laut den geöffneten offiziellen Dokumentationen plausibel: Sol unterstützt `high`, Responses/Batch und 128.000 Ausgabetokens. DeepSeek dokumentiert `deepseek-flash`, `high`, bis zu 393.216 Tokens und wirksames `top_p` im Thinking-Modus. Das ist eine Dokumentationsprüfung, keine Bestätigung des konkreten Kontozugangs oder einer tatsächlich akzeptierten Kombination aller Parameter. [GPT-5.6 Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol), [Batch](https://developers.openai.com/api/docs/guides/batch), [DeepSeek API](https://api-docs.deepseek.com/api/create-chat-completion/), [Thinking-Modus](https://api-docs.deepseek.com/guides/thinking_mode/)

Vor dem bezahlten Hauptlauf müssen die vorgesehenen Regeln auch ausführbar nachgewiesen sein: eindeutige Request-/Provider-IDs; unveränderliche Nutzlasten; Rohantwort und Usage pro Versuch; explizite endgültige Fehlerzustände; Wiederanlauf ohne doppelte Bezahlung oder verlorene Resultate; Abgleich unklarer Abbrüche vor erneutem Versand; vollständige Budgetreservierung und Abrechnung. Technische Tests sollten auf separaten Entwicklungs-/Testeingaben stattfinden und keine Auswahl anhand günstiger Schätzresultate erlauben.

Die geplanten harten Kostenobergrenzen garantieren nicht, dass alle 840 Antworten pro Anbieter finanzierbar sind. Das ist bei hohen Ausgabelimits normal: tatsächlicher Verbrauch und Reservierung müssen einen transparenten Stopp erlauben; ein solcher Stopp bleibt unvollständige Durchführung. Anbieterpreise und die hinter dem Alias bediente Version gehören unmittelbar vor dem Lauf in ein Releaseprotokoll. Insbesondere nennt DeepSeek aktuell hinter `deepseek-flash` eine konkrete V4.1-Version. [DeepSeek Modelle und Preise](https://api-docs.deepseek.com/quick_start/pricing/)

**7. Wiederaufnahme: Starke Teile vorhanden, aber der Schutz ist nicht durchgängig**

Die Offline-Hauptpipeline bindet Quellen, Konfiguration und Umgebung an gespeicherte Hashes. Dieser Schutz ist sinnvoll und hat für den aktuellen Lauf bestanden. Mehrere andere Stufen prüfen hingegen nur Existenz:

- [run_cells10_offline.sh](../../../scripts/run_cells10_offline.sh): Eine `.done`-Datei überspringt die Stufe, ohne ihre Abhängigkeiten zu prüfen.
- [pool.py](../../../src/main_experiment/pool.py), `build_pool`: Eine vorhandene Beobachtungsdatei wird übernommen, bevor Parameter-/Codeidentität geprüft werden.
- [run_baseline_revision.py](../../../scripts/run_baseline_revision.py), `stage_dev`: Eine vorhandene `development_observations.csv` wird wiederverwendet, ohne die erzeugenden Modelle und Beobachtungen daran zu binden.
- [run_qwen_engine.py](../../../scripts/run_qwen_engine.py), `pending`: Existenz der Ergebnisdatei genügt; ein verändertes Decoding oder Modell wird dort nicht gegen den gespeicherten Lauf geprüft.
- [evaluate_main_responses.py](../../../scripts/evaluate_main_responses.py): Ein fehlender Prompt-Hash wird akzeptiert; der Hash der primären Baseline-Datei fehlt im abschließenden Evaluationsbericht.

Das ist ein Risiko beim Fortsetzen oder Verändern, kein Nachweis vermischter aktueller Ergebnisse. Die gegenwärtigen Identitäten und Werte haben die Prüfung bestanden. Für den neuen API-Runner würde ich Hashbindung von Prompt, kompletter Nutzlast, Modellkonfiguration, Parser und Referenzartefakt verbindlich machen. Für vorhandene Stufen sollte eine abgeschlossene Stufe ihre Eingabebindung verifizieren oder bei Änderungen ein neues Ausgabeverzeichnis verlangen. Eine `.done`-Datei allein ist kein wissenschaftlicher Identitätsnachweis.

**8. Budget und Informationszugang: vertretbare Laborbedingungen, keine Gleichheit des Informationsgehalts**

Die Arme sind auf die erwartete Zahl beobachteter aktiver Dyade-Fenster kalibriert. Das Matching ist implementiert und geprüft. Daraus folgt weder gleicher statistischer Informationsgehalt noch gleicher Erhebungsaufwand. Ob vollständige Historien weniger Dyaden oder einzelne Events vieler Dyaden vorliegen, verändert die Identifizierbarkeit und den Fehler. `Walk_A` liefert in S zudem bereits die Summen, aus denen der klassische Korrektor unmittelbar berechnet werden kann.

Die Kalibrierung verwendet das vollständige Archiv. Das ist für einen kontrollierten Benchmark legitim, aber in der späteren Anwendung bei unbekanntem Vollarchiv nicht einfach verfügbar. Formuliere deshalb „mit Vollarchivwissen kalibrierte Beobachtungsszenarien“. Der Anspruch, direkt eine praktisch umsetzbare Methode für gleich teure Datenerhebung zu liefern, wäre nicht gedeckt.

Auch „Vollarchivinformationen bleiben ausgeschlossen“ braucht eine präzise Bedeutung: Es werden keine Vollarchivkennzahlen direkt als Features übergeben. Die übergebenen Parameter `n_panel`, `n_dyads`, `L` und `p` werden aber aus dem Vollarchiv bestimmt. Sie können entsprechend indirekte Information tragen. ExtraTrees lernt außerdem aus gelabelten Aufgaben mit derselben 10-%-Regel und erhält bereits Plug-in-/Korrektorwerte als abgeleitete Features. Das ist keine von mir gefundene direkte Testlabel-Leckage, wohl aber ein zusätzlicher aufgabenspezifischer Informationsvorteil gegenüber dem Zero-shot-LLM. Der aktuelle Ergebnisbericht benennt ihn bereits; er gehört auch in die Hauptdarstellung.

Vier Beobachtungsszenarien sind als Vergleich sinnvoll. Sie bilden kein kausales 2×2-Experiment, in dem ausschließlich „Selektion ja/nein“ und „Historienverlust ja/nein“ variiert werden. Ich würde die vorhandene Zerlegung des Plug-in-Fehlers in Auswahl- und Historienkomponente zur Erklärung verwenden und keine kausalen Haupteffekte aus den bloßen Armnamen ableiten.

**9. Identifizierbarkeit und die Bedeutung von Persistenz**

Für H lässt sich die Informationsgrenze sogar unter deiner aktuellen Budgetregel konstruktiv beweisen. Zwei Archive haben dieselben 100 Dyaden, dieselbe Topologie, dieselben fünf jüngsten Events jeder Dyade in Fenster 5, jeweils 600 Events und 200 aktive Dyade-Fenster:

- Archiv A: Jede Dyade hat zusätzlich ein Event in Fenster 1. Damit ist ρ₂=1.
- Archiv B: Die Hälfte hat keine früheren Events; die andere Hälfte hat je eines in Fenster 1 und 2. Damit ist ρ₂=0,5.

In beiden Archiven setzt die H-Kalibrierung `n_dyads=20`. Für jede mögliche Auswahl von 20 Dyaden ist der H-Input identisch. Damit stimmen die gesamten Verteilungen der H-Beobachtung überein, obwohl sich ρ₂ um 0,5 unterscheidet. Jeder Schätzer hat über diese beiden möglichen Archive im Mittel mindestens 0,25 absoluten Fehler. Das ist eine eigene Herleitung, mit dem aktuellen Sampler nachvollzogen: [Gegenbeispiel](h_identifiability_counterexample.json).

Die Konsequenz lautet nicht, dass der Benchmark sinnlos wäre. Er prüft, welche Annahmen und gelernten Regelmäßigkeiten auf dem gewählten Panel gute Punktvorhersagen ergeben. Ein Verfahren kann aber aus diesem Input nicht allgemeingültig das richtige Archivprofil rekonstruieren. Die Unterscheidung zwischen Informationsverlust, Modellannahmen und gewöhnlicher Stichprobenunsicherheit sollte sichtbar werden.

Außerdem ist ρ₂ eine **fensterbasierte Wiederkehr**, keine Beziehungsdauer: `11000` und `10001` zählen gleich. Ein Fenster entspricht hier etwa 19 Stunden beim Hospital, 5,6 Tagen bei Copenhagen und 470 Tagen bei MathOverflow. Normierung gleicht die mathematische Form an, nicht die soziale Bedeutung. Der bekannte „vollständige“ Zielgraph ist das bereinigte verfügbare Archiv, nicht die gesamte soziale Wirklichkeit.

Der H-Cap begrenzt Eventrecords. Fünf Sensorrecords können eine kurze zusammenhängende Begegnung darstellen, fünf Nachrichten dagegen weit auseinanderliegen. S und B hängen ebenfalls an der ursprünglichen Erfassungsgranularität. Diese Abhängigkeit ist keine Implementierungspanne, muss aber quellenbezogen beschrieben werden. Ein kompletter nachträglicher Umbau zu Kontakt-Episoden wäre eine neue Studie und ist für die jetzige Aussage nicht erforderlich.

**10. Die Korrektoren: brauchbare Referenzen mit unterschiedlichen Garantien**

R hat gleiche marginale Dyadeninklusion, aber ein Quotient mit zufälligem Nenner ist nicht allgemein exakt unverzerrt. Die uneingeschränkte Unverzerrtheitsaussage in `results/baseline_revision_20261001/CORRECTOR_DECISION.md`, Zeile 10, sollte in einer aktuellen Erläuterung korrigiert werden; das historische Entscheidungsprotokoll selbst bleibt als Historie erhalten. Es gibt keinen Anlass, deshalb den R-Plug-in auszutauschen.

S korrigiert stationäre Traversierungshäufigkeiten durch inverse Ereignisgewichte. Ein uniformer Start ohne Burn-in und ein endlicher Walk erfüllen diese Asymptotik nicht exakt; vollständiges Budgetmatching beweist keine Durchmischung. Ich habe zusätzlich die Zusammenhangskomponenten geprüft: Vier der sechs realen Graphen sind verbunden. In CollegeMsg und MathOverflow liegen jeweils mehr als 99,9 % der Dyaden in der größten Komponente; die asymptotische Verzerrung allein durch die zufällige Startkomponente beträgt für ρ₂ ungefähr −0,00030 beziehungsweise −0,00029. Die abstrakte Komponentenproblematik ist hier also kein großer nachgewiesener Fehler. Endliche Walks bleiben relevant, etwa L=111 im Hospital. [Konkrete Diagnose](walk_components.json), [Random-Walk-Grundlagen](https://arxiv.org/abs/1612.03281)

Die H-Schranken sind für die gezogenen Dyaden korrekt hergeleitet. Sie sind kein Konfidenzintervall für das Archiv. Eine Archivvorhersage außerhalb des Stichprobenintervalls darf deshalb nicht pauschal als unmöglich bezeichnet werden. Der bestehende Parser verwirft solche Vorhersagen richtigerweise nicht.

Die Beta–ZTP-Referenz in B ist mathematisch als Arbeitsmodell nachvollziehbar; die positive Berechnung der Zellwahrscheinlichkeiten und die gemeinsame Anpassung der Eventintensität sind sorgfältig umgesetzt. Sie modelliert aber nicht beliebige Burstiness, Eintrittsprozesse oder dyadenspezifische Eventintensitäten. Schwache Identifikation und Rückfälle müssen neben ihrer Güte berichtet werden. Die DAR-Eventschicht `1+Poisson` ist zudem keine ZTP-Verteilung: Der B-Korrektor ist dort bewusst misspezifiziert. Ich würde die Referenz beibehalten, nicht nach den nun bekannten Testresultaten eine besser passende auswählen.

**11. Synthetik und bisheriges Ergebnis richtig gewichten**

DAR und AD sind geeignete kontrollierte Beispiele. Zwei Replikate je Bedingung liefern allerdings keine präzise Aussage über die ganze Generatorfamilie. Die acht Instanzen entstehen aus vier unabhängigen Paaren; gekoppelte Varianten dürfen bei einer gemeinsamen Unsicherheitsangabe nicht als acht unabhängige Replikate behandelt werden. Die Bedingungen getrennt zu zeigen ist hier besser als eine einzige synthetische Rangliste mit vermeintlich hoher Präzision.

Bei DAR verändert α trotz gleicher marginaler Aktivität die Zahl jemals aktiver Dyaden und damit das auf aktive Dyaden konditionierte Zielprofil. Bei AD verändert Gedächtnis auch die aggregierte Topologie. Das sind legitime Folgen der Mechanismen, aber keine Isolation eines einzigen Effekts bei ansonsten identischen beobachtbaren Graphen. Außerdem belastet H die synthetischen Hauptinstanzen vergleichsweise wenig; die vorhandenen realen Quellen tragen den wesentlichen H-Stresstest. Das sollte als Abdeckungsgrenze benannt werden.

Die nachgerechnete Tabelle für die sechs realen Quellen lautet:

| Arm | Qwen Thinking MAE₂ | Primärer Korrektor MAE₂ | Differenz |
|---|---:|---:|---:|
| R | 0,01762 | 0,01752 | +0,00010 |
| S | 0,07990 | 0,03079 | +0,04911 |
| H | 0,13725 | 0,03449 | +0,10276 |
| B | 0,20088 | 0,07653 | +0,12434 |

Das bisherige Ergebnis ist inhaltlich deutlich: Qwen Thinking erreicht in R ungefähr Plug-in, korrigiert in S einen Teil des starken Auswahlfehlers, bleibt dort aber hinter dem analytischen Korrektor. In H und B liefert es auf diesem Panel keinen Vorteil gegenüber der primären Referenz. Non-thinking ist mit MAE₂ von etwa 0,43–0,55 sehr schwach, obwohl sämtliche Antworten formal gültig sind. Ein Parserfehler erklärt diesen Befund im aktuellen Design nicht.

Ein späteres gutes GPT-/DeepSeek-Ergebnis ist möglich, aber wissenschaftlich nicht nötig, um die Arbeit zu rechtfertigen. Eine überzeugende negative Aussage mit klarer Aufgabenabgrenzung ist stärker als weitere Designänderungen auf der Suche nach einem LLM-Sieg.

**Konkrete Reihenfolge, die ich empfehlen würde**

1. **Vor weiteren Hauptantworten:** Forschungsfrage, Entwicklungschronologie und Konfigurationsvergleich einschließlich unterschiedlicher Ausgabeunterstützung schriftlich festlegen. Bestehende Beobachtungen und Hauptreferenzen bleiben erhalten.
2. **Auswertung korrigieren:** MCSE-Darstellung konsistent machen und Systemscore auch bei 100 % Rückfall berichten; diese Fälle gezielt testen. Vorhandene Antworten neu auswerten, Originalauswertung behalten.
3. **Technische Freigabe herstellen:** Sol-/DeepSeek-Transport, Identitätsbindung, Versuchsspeicherung, Wiederaufnahme, Fehlerzustände und Budgetledger implementieren und prüfen. Modell-/Parameter-/Preisprüfung auf technischen Testeingaben abschließen.
4. **Dann Hauptläufe:** Dieselben 280 Beobachtungen und drei Antworten je Anbieter, ohne ergebnisabhängiges Prompt- oder Baseline-Tuning.
5. **Für die schriftliche Arbeit:** Tatsächliche Poolaufteilung, Zeitspannen/Eventdefinitionen, Grenzen von S/H/B und Quelle-/Paarstruktur sichtbar machen. Bestehende Diagnosen verwenden. Eine ausgewogene neue synthetische Entwicklungsdiagnose ist sinnvoll, aber als nachträgliche Sensitivität kein Grund, die Hauptläufe umzubauen.

Die beigefügte Einschätzung ist in vielen methodischen Punkten zutreffend. Diese Prüfung bestätigt insbesondere die Grenzen der Designhistorie, der Varianzzerlegung und der H-Schranken. Sie ergänzt jedoch überprüfbare Befunde zur tatsächlichen Poolaufteilung, zur unterschiedlichen Ausgabeunterstützung, zur Ausfallaggregation und zu unvollständigen Resume-Bindungen. Deshalb würde ich die verbleibende Arbeit nicht als reine Textkorrektur behandeln. Der notwendige Umfang ist trotzdem begrenzt: keine wissenschaftliche Kernsanierung, sondern klar benannte Korrekturen und eine belastbare Fertigstellung.

Im Rahmen dieser Prüfung wurden ausschließlich dieser Prüfbericht und seine Nachweise angelegt. Experimentcode, Word-Dokument, bestehende Antworten und bestehende Hauptauswertungen wurden nicht geändert. Es wurden keine neuen Modellanfragen oder Clusterjobs gestartet.
