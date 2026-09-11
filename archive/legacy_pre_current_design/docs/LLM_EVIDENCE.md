# Evidenzstand: LLM-Input und Modellvergleich

## Wozu dieses Dokument

Es sammelt, was über die Wahl des LLM-Inputs und über das Verhalten der
geprüften Modelle gemessen wurde, und mit welcher Belastbarkeit. Es ist kein
Regelwerk: die Zahlen sollen eine eigene Einschätzung ermöglichen, nicht
ersetzen. Wo die Evidenz dünn oder mehrdeutig ist, steht das dabei; wo eine
Deutung naheliegt, ist sie als Deutung gekennzeichnet.

Einige der weiter unten genannten Muster sind nachträgliche explorative
Auswertungen. Sie sind als mögliche Ansatzpunkte für eine unabhängige Prüfung
gedacht, nicht als vollständige Pattern-Liste oder als Vorgabe, wonach in den
Antworten gesucht werden soll. Insbesondere können andere sinnvolle
Stratifizierungen oder Lesarten zu einem anderen Gesamtbild führen.

Die Zieldefinitionen liegen getrennt in `docs/TARGET_EVALUATION_FREEZE.md`,
die Modellmatrix und die Cluster-Läufe in `docs/LLM_V21_RUNS.md`, die
Nicht-LLM-Vergleichsverfahren in `docs/DESIGN_BASELINES.md`.

## Zwei Panels, die man nicht verwechseln sollte

- **Historische Evidenz:** 84 Fälle, 420 eingefrorene Prompts unter
  `results/llm_v2/`. Alle Zahlen hier beziehen sich darauf.
- **Finales Hauptpanel:** 32 Graphinstanzen unter
  `results/final_target_panel/`, drei Walks, 96 Fälle. Prompts dafür sind noch
  nicht erzeugt.

Jede Übertragung vom ersten auf das zweite Panel ist eine begründete
Designentscheidung, kein gemessener Befund. Das historische Panel ist
budgetlastig auf 100 und enthält einen Renewal-Block, den das finale Panel
nicht hat.

Daneben existiert eine ältere **Phase-3-Vorstudie** (60 Fälle, andere Modelle,
Ausgabe nur eines Skalars statt eines Profils, `results/phase3/`). Ihre
absoluten Zahlen sind nicht mit den 84-Fall-Zahlen vergleichbar; die *Richtung*
von Bedingungseffekten lässt sich gegenüberstellen.

Integrität der historischen Evidenz: `results/llm_v2/prompts.jsonl` mit 420
eindeutigen Prompts, SHA-256
`12a32ecf43c54df38b5f08f43569cebe4dd37fdec21aa2a05b9fe0b4b8f50dc8`;
`llm_cases.csv` mit 84 Fällen; jede gepaarte Inputzelle mit 36 identischen
Fällen, 12 je Walkstrategie.

## Die Varianten

### Inputleiter

| input_kind | Inhalt | Promptlänge relativ zu `mask` |
|---|---|---|
| `nw` | `(n, w)`-Histogramm; nur Anzahl belegter Fenster, nicht deren Lage | 0,89× |
| `mask` | `(n, Fenstermaske)`-Histogramm; welche Fenster beobachtet wurden | 1,00× |
| `mask_crawl_full` | `mask` plus Crawl-Diagnostik | 1,19× |
| `mask_crawl_temporal` | zusätzlich aggregiertes Temporalprofil | 1,64× |
| `mask_crawl_temporal_recent` | zusätzlich jüngste anonymisierte Events | 2,30× |

`C_one_step` ist aus `nw` grundsätzlich nicht identifizierbar. Die vier
`disclosed`-Ablationszellen teilen sich exakt 36 `case_id`s und sind eine
Teilmenge der 84 `disclosed/mask`-Fälle; für einen Inputvergleich gehört `mask`
auf dieselben 36 eingeschränkt. Leaderboards zeigen `mask` sonst mit n=84.

### Kontextachse

| condition | Zusatz gegenüber der vorigen Stufe | Länge relativ zu `hidden` |
|---|---|---|
| `hidden` | — | 1,00× |
| `disclosed` | Beschreibung des jeweils verwendeten Sammelmechanismus und seiner möglichen Auswahlverzerrungen | 1,17× |
| `disclosed_examples` | zusätzlich drei gelöste Beispiele mit Eingabe und korrekten Antworten | 2,18× |

Die Beispiele stammen aus Gruppen, die von allen Evaluationsgruppen disjunkt
sind (im Code per `assert` abgesichert), also nach derselben Disziplin wie das
GroupKFold der ML-Baselines.

**Technischer Hinweis zur archivierten Mechanismusbeschreibung.** Der
V2.1-Prompt beschreibt `time_agnostic_t` an einer Stelle als nicht uniforme
Kantenstichprobe. Präziser gilt für den einfachen Random Walk auf dem
ungewichteten ungerichteten Graphen: Nach dem Mischen sind gerichtete
Kantentraversierungen und damit auch ungerichtete Kanten stationär uniform.
Bei endlichem Budget können Start-, Mixing- und Lokalitätseffekte die
realisierte Stichprobe trotzdem ungleich aussehen lassen. Die archivierten
Prompts werden dadurch nicht nachträglich verändert; die Formulierung sollte
bei der Interpretation ihrer Antworten aber nicht als exakt vorausgesetzt
werden.

## Ausgewertete Läufe

Rund 3.800 Prompt-Antworten aus zwölf Läufen einschließlich der abgeschlossenen
Claude-Examples-Versuche:

| Familie | Läufe | Umfang |
|---|---|---|
| API | Gemini-3.1-Lite (minimal/think), DeepSeek-V4-Pro nothink, Mistral-Small-4 (none/high) | je 420/420 vollständig |
| Open Weights | Qwen3.6-27B nothink 410/420, Qwen3.6-27B think 416/420, R1-Distill-32B 303/420 | `results/llm_v21/cluster_snapshot/`, Prompts SHA-identisch |
| Produkt-Screens | Codex GPT-5.6: `disclosed/mask` je 56 und `disclosed_examples/mask` je 30 vollständig; Claude Code Opus: Baseline NoTools 28/30 und Tools 30/30, Examples NoTools 27/28 und Tools 30/30 vollständig | `results/codex_screen_snapshot/`, `results/cc_screen_snapshot/` |

Bei den Open-Weights-Läufen sind viele offene Prompts tokenlimitbedingt
(`length`) und nicht inhaltlich gescheitert. Complete-Case- und
failure-penalisierte Zahlen laufen dort weit auseinander; beide zu zeigen ist
informativer als eine davon zu wählen. Nie versuchte Prompts sind etwas
anderes als versuchte und gescheiterte.

**Stand der Open-Weights-Nachläufe:** Die Budgetleiter (Primärlauf, dann
Wiederholung der offenen Fälle mit größerem Tokenbudget) ist in
`docs/LLM_V21_RUNS.md` dokumentiert. Der lokale Snapshot enthält inzwischen
den abgeschlossenen dritten Tier: NoThink 410/420 und Think 416/420. Die
verbleibenden 10 beziehungsweise 4 Fälle sind tokenlimitierte Nichtantworten;
für R1-Distill ist nach 303/420 kein weiterer Tier geplant.

Die Produkt-Screens tragen einen nicht entfernbaren Harness-Prompt und lassen
sich nicht auf eine Version pinnen (bei Codex nachgemessen: ~9,2 kB injizierte
Anweisungen, auch mit deaktivierten Werkzeugen). Das schließt sie nicht aus,
ändert aber, welche Aussage sie stützen können. Codex hat inzwischen auch die
gepaarte `disclosed_examples/mask`-Zelle in beiden Armen vollständig. Auch die
Claude-Versuche sind beendet; ein NoTools-Examples-Fall blieb nach einem
1.500-Sekunden-Timeout ohne vollständige Antwort. Die append-only Dateien enthalten daneben
Retry- und Usage-Limit-Records; die Auswertung lässt einen späteren Fehler
eine bereits vollständige Antwort desselben Prompts nicht überschreiben.

## Befunde

Gerechnet nach der Regel aus dem Target-Freeze: nur das finale JSON, kein
Clipping, ungültige Komponenten mit Strafe 1, ProfileMAE über k=2..5, Paarung
über `case_id`.

### Inputleiter: kein robuster Gewinner

Über zehn auswertbare Läufe hinweg gewinnt jede Variante bei einigen Modellen
und verliert bei anderen; die aktuellen mittleren Ränge liegen ungefähr
zwischen 2,70 (`+recent`) und 3,50 (`mask`) bei fünf Varianten.

Entscheidend für die Einordnung ist die Größenordnung. Betrachtet man nur die
Fälle, in denen ein Modell überhaupt geantwortet hat, beträgt die Spanne über
die fünf Varianten **innerhalb** eines Modells im Median 0,027, **zwischen**
den Modellen dagegen 0,120. Auf denselben zehn Läufen liegen die
Complete-Case-Spaltenmittel eng zwischen rund 0,141 und 0,148 (`nw` 0,145 ·
`mask` 0,145 · `+crawl` 0,148 · `+temporal` 0,141 · `+recent` 0,144). Ein
scheinbar niedrigeres `mask`-Mittel entsteht, wenn zusätzlich die beiden nur
für `mask` vorhandenen Claude-Screens einfließen; das wäre kein symmetrischer
Inputvergleich.

Die großen Unterschiede in den penalisierten Zahlen stammen fast vollständig
aus Nichtantworten, nicht aus schlechteren Antworten. Was die reicheren Inputs
messbar tun: Sie verlängern den Prompt und senken bei tokenlimitierten
Modellen die Antwortrate (Qwen3.6 nothink von 1,00 auf 0,92; R1-Distill bei
`+crawl` auf 0,67). Bei den API-Modellen bleibt die Antwortrate bei 1,00.

`mask_crawl_full` wirkt in dieser Kombination eher ungünstig: längerer Prompt,
im vergleichbaren Complete-Case-Mittel kein erkennbarer Gewinn und bei einigen
tokenlimitierten Läufen eine niedrigere Antwortrate. Die Fehlerunterschiede
zwischen den Inputs sind allerdings klein.

Die steigende Antwortrate für `C_one_step` bei reicheren Inputs ist
möglicherweise ein Artefakt der Prompt-Erlaubnis, `null` zu antworten, und
könnte sich verändern, wenn `C` verpflichtend abgefragt wird.

*Mögliche Lesart:* Das spricht für den kleinsten Input mit voller
Identifizierbarkeit, also `mask`. *Gegenargument, das man kennen sollte:*
`mask` hat den nominell schlechtesten mittleren Rang; die Wahl stützt sich
also auf Nebenkriterien (Länge, Antwortrate, Identifizierbarkeit) und nicht
auf einen gemessenen Genauigkeitsvorteil. `nw` bleibt als Informationsablation
interessant, weil `C` dort prinzipiell fehlt.

Explorativ ist auch die geringe Übereinstimmung zwischen Modellen: Die
modellübergreifenden Korrelationen der fallweisen Fehleränderung gegenüber
`mask` liegen für die vier Alternativen nur ungefähr zwischen 0,00 und 0,05.
Das ist damit verträglich, dass zusätzlicher Input stark modell- oder
promptabhängig genutzt wird. Es beweist weder, dass die Zusatzinformationen
wertlos sind, noch dass es keine Teilgruppen mit konsistentem Nutzen gibt.

Für das sekundäre Lifetime-Ziel fallen die reicheren temporalen Inputs etwas
günstiger aus als für das Persistenzprofil. Da Lifetime nicht zum zentralen
Target des Hauptvergleichs gehört, ergibt sich daraus allein kein Grund, den
umfangreicheren Input für die Hauptaufgabe zu bevorzugen.

### Kontextachse: großer Effekt auf das Niveau, uneinheitlich auf die Rangfolge

Der systematische Unterschätzungs-Bias verringert sich beim Übergang
`hidden` → `disclosed` → `disclosed_examples` bei **allen acht** daraufhin
geprüften Modellen, im aktuellen Stand im Mittel von ungefähr −0,255 über
−0,169 auf −0,111. Einzelne Modelle bewegen sich deutlich.

Der Fehler auf beantworteten Fällen sinkt ebenfalls monoton, aber schwach:
0,158 → 0,151 → 0,141 im Mittel über die acht Läufe. Diese Größenordnung ist
klein gegenüber vielen Unterschieden zwischen Modellen.

Die Rangkorrelation zur Wahrheit entwickelt sich uneinheitlich und im Mittel
nach unten: ungefähr 0,385 (`hidden`) → 0,251 (`disclosed`) → 0,243
(`disclosed_examples`). Beim direkten Schritt von `disclosed` zu den
Beispielen verlieren aktuell vier von acht Modelle Rangkorrelation; über den
gesamten Schritt von `hidden` zu den Beispielen sind es fünf von acht. Das
Mittel allein verdeckt erhebliche Unterschiede zwischen den Läufen.

Die Beispiele verlängern den Prompt auf das 2,18-Fache. Bei tokenlimitierten
Modellen kostet das Antworten: Qwen3.6 think fällt von 0,88 auf 0,54
Antwortrate, R1-Distill von 0,76 auf 0,64. Bei den API-Modellen ändert sich
nichts.

Eine explorative Aufteilung deutet darauf hin, dass Beispiele den `rho_k2`-
Fehler eher bei Fällen mit hohen Wahrheitswerten und bei den zuvor stärker
unterschätzenden Walkstrategien senken. In niedrigen Wahrheitsquartilen ist
der mittlere Unterschied klein. Das passt zu einer Kalibrierungswirkung, ist
aber keine eindeutige Trennung zwischen Kalibrierung und neu gelernter
Falldifferenzierung. Bei Codex verbessern sich auf den 30 gepaarten Fällen
zusätzlich die Rangkorrelationen. Bei Claude steigt die rho2-Rangkorrelation
in beiden Armen ebenfalls; zugleich wird der rho2-Bias negativer und der
rho2-MAE leicht schlechter. ProfileMAE bleibt bei NoTools praktisch gleich
und sinkt bei Tools leicht, während `C_one_step` in beiden Armen günstiger
ausfällt. Das ist kein einheitlicher Gewinn über Zielgrößen hinweg.

### Ankerkontrolle

Da die drei Beispiele bewusst über die Spannweite gestreut sind, liegt ihr
Mittelwert nahe am Gesamtmittelwert der Wahrheitswerte. Eine Konstantvorhersage
in Höhe dieses Mittelwerts erreicht auf den 84 Fällen einen rho2-Fehler von
0,187 und schlägt damit die API- und Open-Weights-Läufe (die CLI-Screens laufen
auf anderen Fallmengen und sind nicht direkt gegenzurechnen).

Direkte Diagnostiken zeigen, dass die Modelle **nicht** einfach diesen Wert
ausgeben: ihre Vorhersagen streuen weiterhin, nur ein Teil liegt in
Ankernähe, und die Kopplung an die beobachteten Daten steigt bei einigen
Modellen mit Beispielen sogar an.

*Einschränkend:* In der Bedingung mit Beispielen steht das Niveau des
Mean-Floors faktisch im Prompt. Vergleiche gegen genau diesen Floor sind dort
schwach interpretierbar, Vergleiche gegen die übrigen Baselines und zwischen
Modellen nicht.

### Woran hängt die Vorhersage?

Ein naiver Ablesewert aus den beobachteten Daten (`readoff_rho2` in
`llm_eval_frozen.py`) korreliert selbst mit 0,56 zur Wahrheit. Die
Rangkorrelation der Modellvorhersagen zu diesem Ablesewert:

| Lauf | `hidden` | `disclosed` | `disclosed_examples` |
|---|---|---|---|
| qwen3.6 think | 0,98 | 0,65 | 0,90 |
| qwen3.6 nothink | 0,94 | 0,61 | 0,60 |
| r1-distill-32b | 0,90 | 0,48 | 0,23 |
| mistral-s4 none | 0,83 | 0,61 | 0,42 |
| deepseek-v4 nothink | 0,73 | 0,39 | 0,29 |
| mistral-s4 high | 0,69 | 0,18 | 0,14 |
| codex-5.6 notools | – | 0,73 | – |
| claude-code notools | – | 0,27 | – |

In `hidden` liegen mehrere Läufe sehr nahe am Ablesewert; ihre Korrelation zur
Wahrheit (0,52–0,57) entspricht dort ungefähr dem, was der Ablesewert allein
erreicht. Mit Offenlegung sinkt die Kopplung an den Ablesewert deutlich — bei
den meisten Läufen sinkt gleichzeitig die Korrelation zur Wahrheit.

Claude Code fällt aus dem Muster: geringe Kopplung an den Ablesewert (0,27)
bei der höchsten Korrelation zur Wahrheit (0,76). Codex liegt dazwischen.

*Mögliche Lesart:* Die Offenlegung löst die Modelle von einer brauchbaren
Heuristik, ohne dass etwas Besseres an deren Stelle tritt. *Alternative
Lesart, die die Daten nicht ausschließen:* Der Ablesewert ist in `hidden` die
rationale Antwort, weil die Verzerrung dort nicht bekannt ist; der Rückgang
der Rangkorrelation unter Offenlegung könnte also auch eine Verschiebung des
Aufgabentyps abbilden statt eines Fähigkeitsdefizits.

### Reagieren die Läufe auf die Datenlage?

Fehler auf `rho_k2`, aufgeschlüsselt nach Abdeckung des Walks
(`disclosed`/`mask`; die Wahrheitsmittel steigen von 0,272 auf 0,444):

| Lauf | sehr niedrig | niedrig | mittel | hoch |
|---|---|---|---|---|
| gemini-lite think | 0,239 | 0,268 | 0,267 | 0,288 |
| deepseek-v4 nothink | 0,254 | 0,272 | 0,246 | 0,260 |
| qwen3.6 nothink | 0,238 | 0,273 | 0,259 | 0,239 |
| r1-distill-32b | 0,273 | 0,269 | 0,339 | 0,340 |
| codex-5.6 notools | 0,227 | 0,303 | 0,202 | 0,083 |
| claude-code notools | 0,170 | 0,114 | 0,111 | 0,040 |

Bei den meisten Läufen ist der Fehler über die Abdeckungsbänder weitgehend
flach; bei den beiden CLI-Screens fällt er um das Drei- bis Vierfache.

Diese rohe Aufteilung ist jedoch mit dem Wahrheitsniveau und der
Fallschwierigkeit konfundiert. Nach einer einfachen linearen Kontrolle für den
wahren `rho_k2` ist höhere Coverage bei allen betrachteten Läufen mit kleinerem
Fehler verbunden. Das ist keine kausale Schätzung, spricht aber dagegen, die
flachen Rohwerte unmittelbar als fehlende Nutzung der Coverage zu lesen.

Dieselbe Trennung zeigt sich bei der **Walkstrategie**. Die Wahrheitsmittel
sind über die drei Strategien praktisch gleich (0,369 / 0,381 / 0,383). Für
die CLI-Screens und Qwen3.6 think unterscheidet sich der Fehler zwischen
`time_agnostic_t` und `time_respecting` um einen Faktor 2–3 (Claude Code
0,065 gegen 0,157; Codex 0,106 gegen 0,279); für gemini-lite minimal und
deepseek-v4 nothink kaum.

Und bei der **Stabilität über die Inputformate**: Verhältnis aus der Streuung
der Vorhersage desselben Falls über die fünf Varianten zur Streuung zwischen
den Fällen — deepseek 1,23 · gemini think 1,18 · mistral none 1,17 ·
mistral high 1,15 · gemini minimal 0,99 · qwen think 0,54 · codex 0,33. Bei
Werten um 1 bewegt ein Formatwechsel die Antwort etwa so stark wie ein
Wechsel des Netzwerks.

*Zur Vorsicht:* Die Inputvarianten sind nicht informationsgleich (`nw` ist eine
echte Reduktion, die reicheren Varianten fügen Diagnostik hinzu). Ein Teil der
Bewegung ist also erwartbar. Wie groß dieser Teil ist, wurde nicht getrennt
geschätzt.

### Kalibrierung und Selbsteinschätzung

**Regressionssteigung** von Vorhersage auf Wahrheit (`rho_k2`; 1,0 wäre
kalibriert, 0,0 hieße, der Fall geht nicht ein):

| Lauf | `hidden` | `disclosed` | `+Beispiele` | sd(pred)/sd(truth) |
|---|---|---|---|---|
| gemini-lite minimal | 0,09 | −0,01 | 0,17 | 0,56 |
| gemini-lite think | 0,09 | 0,17 | 0,31 | 0,70 |
| deepseek-v4 nothink | 0,10 | 0,12 | −0,01 | 0,93 |
| mistral-s4 none | 0,28 | 0,18 | 0,23 | 0,98 |
| mistral-s4 high | −0,02 | 0,18 | 0,02 | 1,44 |
| qwen3.6 nothink | 0,14 | 0,28 | 0,33 | 0,87 |
| qwen3.6 think | 0,13 | 0,39 | 0,43 | 1,09 |
| r1-distill-32b | 0,14 | 0,04 | −0,03 | 0,67 |
| codex-5.6 | – | 0,67–0,71 | – | 1,11 |
| claude-code | – | 0,53–0,66 | – | 0,68–0,89 |

Die Streuungsspalte zeigt, dass die niedrigen Steigungen nicht aus Vorsicht
entstehen: die Vorhersagen variieren durchaus, bei Mistral-high sogar stärker
als die Wahrheit.

**90%-Intervalle:** Abdeckung 4–26 % bei den API- und Open-Weights-Läufen,
68–82 % bei den CLI-Screens. Die Rangkorrelation zwischen Intervallbreite und
tatsächlichem Fehler liegt bei den ersteren zwischen −0,19 und +0,12, bei
Codex und Claude Code zwischen 0,39 und 0,52. Formal sind die Intervalle
überall sauber (kein `lo > hi`, die Punktschätzung liegt praktisch immer
darin). Ihre Lage unterscheidet sich: die meisten Läufe setzen den Punkt
mittig (0,44–0,48 der Intervallbreite), Codex ins untere Drittel (0,29).

**Innere Kohärenz.** In den Wahrheitsdaten gilt auf Maschinengenauigkeit
`mean_occupancy = (1 + rho_k2 + rho_k3 + rho_k4 + rho_k5) / 5`. Ob die eigene
Antwort eines Laufs diese Identität erfüllt, ist ohne Wahrheitswerte prüfbar:
gemini-lite minimal 24 %, mistral-none 65 %, r1-distill 66 %, deepseek 75 %,
mistral-high 86 %, gemini-lite think 95 %, qwen3.6 nothink 95 %, qwen3.6
think / codex / claude-code 100 %.

Nebenbefund: Der Fehler der **genannten** `mean_occupancy` und der aus dem
eigenen Profil **abgeleiteten** ist bei allen zwölf Läufen praktisch gleich.
Das separate Feld trägt also keine zusätzliche Information — was die
Entscheidung im Target-Freeze, mean occupancy profilabgeleitet zu führen,
datenseitig stützt.

**Nichtidentifizierbarkeit.** Der Prompt erlaubt `null` für `C_one_step`. Bei
Input `nw` ist `C` beweisbar nicht bestimmbar. null-Rate `nw` gegen `mask`:
codex 100 % / 42 %, mistral-none 97 % / 69 %, r1-distill 93 % / 61 %,
qwen3.6 nothink 49 % / 14 %, mistral-high 44 % / 14 %, qwen3.6 think 38 % /
0 %, gemini-lite think 17 % / 3 %, gemini-lite minimal 8 % / 3 %,
deepseek-v4 nothink 86 % / 83 %. Codex trennt hier am schärfsten; DeepSeek
verweigert in beiden Zellen ähnlich oft, unterscheidet also nicht.

### Denkmodus und Werkzeuge

Gepaarte Läufe desselben Modells, `disclosed`/`mask`, gemeinsame Fälle:

| Paar | Fehler | Rangkorrelation | Steigung |
|---|---|---|---|
| Gemini minimal → think | 0,217 → 0,268 | −0,02 → 0,27 | −0,01 → 0,17 |
| Mistral none → high | 0,280 → 0,307 | 0,32 → 0,21 | 0,18 → 0,18 |
| Qwen3.6 nothink → think | 0,220 → 0,228 | 0,51 → 0,49 | 0,40 → 0,39 |
| Codex ohne → mit Werkzeug | 0,188 → 0,187 | 0,51 → 0,53 | 0,71 → 0,67 |
| Claude Code ohne → mit | 0,102 → 0,097 | 0,76 → 0,75 | 0,53 → 0,71 |

In zwei von drei Denkmodus-Paaren steigt der Fehler. Die innere Kohärenz
steigt in allen drei (24 % → 95 %, 65 % → 86 %, 95 % → 100 %).

Aggregiert ändert Werkzeugzugriff das Ergebnis der beiden CLI-Produkte wenig.
Nach Walkstrategie sind die Richtungen aber nicht identisch: Bei Codex und im
kleineren Claude-Sample scheinen Werkzeuge einigen Strategien zu helfen und
anderen kaum oder leicht zu schaden. Wegen der kleinen Teilmengen sollte das
eher als Prüffrage denn als stabiler Werkzeugeffekt gelesen werden. Lokale
Logs zeigen tatsächliche Tool-Aufrufe; deren Nutzen pro Antwort ist damit noch
nicht isoliert.

Daneben existiert eine kleinere `method/mask_crawl_temporal`-Zelle. Eine
generische Aufforderung, eine Schätzmethode zu entwerfen und kurz zu benennen,
verbessert dort das mittlere ProfileMAE nicht erkennbar; Bias und Rangfolge
bewegen sich in unterschiedliche Richtungen und die Effekte unterscheiden sich
zwischen Modellen. Die Teilmenge ist nicht repräsentativ für das Gesamtpanel.
Sie prüft außerdem keine konkrete verpflichtende Rechenregel, sondern nur eine
offene Methodenaufforderung.

### Modelle untereinander

Die mittlere Rangkorrelation zwischen den Läufen (0,36) liegt leicht über der
mittleren Korrelation der Läufe zur Wahrheit (0,31). Auffällig sind zwei
Extreme: qwen3.6 think und codex korrelieren mit 0,80–0,83 miteinander,
gemini-lite minimal mit keinem anderen Lauf (−0,04 bis −0,02).

### Fallstruktur

Der über alle Läufe gemittelte Fehler pro Fall korreliert mit 0,80 zum wahren
`rho_k2` und mit −0,12 zur Abdeckung. Die schwersten Fälle haben
Wahrheitswerte von 0,60–0,90 und liegen überwiegend in den mechanistischen
Datenblöcken; die leichtesten liegen bei 0,05–0,14.

*Hinweis zur Interpretation:* Ein Schätzer, der eine Konstante ausgibt, hätte
per Konstruktion einen Fehler, der stark mit der Wahrheit korreliert. Der Wert
0,80 ist mit diesem Muster verträglich, beweist es aber nicht — auch ein
teilweise informierter Schätzer mit Unterschätzungstendenz erzeugt ihn.

### Was in den Begründungstexten steht

Textmusteranalysen (`analysis/09` bis `11` im Evidenzpaket, bzw. die Skripte
in `analysis/scripts/`). Sie messen, was ein Lauf aufschreibt, nicht was er
tut; ein fehlender Treffer ist kein Beweis für Abwesenheit.

In V2.1 nennen die Läufe in `disclosed_examples` die Beispielwerte deutlich
häufiger als in `disclosed` (Grundrauschen 0–14 %): mistral-high 100 %,
qwen3.6 think 99 %, qwen3.6 nothink 79 %, r1-distill 75 %, mistral-none 55 %,
gemini-lite think 36 %, deepseek nothink 8 %, gemini-lite minimal 0 %.

Die Rechendichte (Divisionen pro Antwort) ist im Vergleich zwischen Läufen
auffällig **gegenläufig** zur Ergebnisqualität: claude-code 3,5 · codex 0,7 ·
gemini-lite minimal 0,2 · qwen3.6 nothink 21 · mistral-none 14 · r1-distill 34
· mistral-high 133 · qwen3.6 think 128. Die beiden besten Läufe schreiben die
kürzesten Begründungen. Innerhalb einzelner Läufe ist der Zusammenhang zwischen
Antwortlänge beziehungsweise sichtbarer Rechendichte und Fehler jedoch schwach
und uneinheitlich. Der Vergleich ist zudem durch Modell und Harness
konfundiert; er rechtfertigt keine Vorgabe, wie lange ein Modell nachdenken
sollte.

### Phase-3-Vorstudie

Dort enthielt `disclosed_calib` **ein** Kalibrierbeispiel mit dem wahren Wert
0,294; das Wahrheitsmittel der 60 Fälle liegt bei 0,311.

| Modell | Fehler `hidden` → `calib` | Bias | Rangkorrelation |
|---|---|---|---|
| deepseek-v4-flash | 0,258 → 0,099 | −0,201 → +0,021 | 0,466 → 0,749 |
| Qwen2.5-14B | 0,276 → 0,161 | −0,272 → −0,048 | 0,379 → 0,204 |
| Qwen2.5-32B | 0,285 → 0,194 | −0,285 → −0,032 | 0,538 → −0,097 |
| R1-Distill-32B | 0,222 → 0,160 | −0,183 → −0,027 | 0,171 → 0,225 |

Anteil der Vorhersagen im Umkreis ±0,05 um 0,294: Qwen2.5-14B 50 %,
R1-Distill 37 %, Qwen2.5-32B 32 %, deepseek-v4-flash 15 %. Zum Vergleich
liefern die Baselines auf denselben Fällen: `floor_lofo` 0,163 MAE bei
Rangkorrelation 0,123, `mle_uniform` 0,130 / 0,738, `mle_betacal_lofo`
0,066 / 0,853.

Automatische Einteilung der Begründungen (zitiert der Text Zahlen, die nur in
der Tabelle des Zielgraphen stehen?): deepseek-v4-flash 68 % rechnend,
r1-distill 75 %, Qwen2.5-14B 20 %, Qwen2.5-32B 8 %. Die beiden Qwen2.5-Läufe
enthalten fast keine Rechenoperationen (0,0–0,1 Divisionen pro Antwort).

*Mögliche Lesart:* Ein einzelnes Kalibrierbeispiel nahe am Mittelwert senkt
den Fehler stark, ohne die Unterscheidungsfähigkeit zu verbessern; bei einem
Lauf (deepseek-v4-flash) verbessern sich beide, und dessen Begründungen
verwenden den Beispielwert als Korrekturfaktor auf die eigenen Zahlen.
*Einschränkung:* R1-Distill rechnet ausweislich der Textanalyse häufig und
kommt trotzdem nicht über Floor-Niveau hinaus — Rechnen allein erklärt den
Unterschied also nicht.

## Grobe Leitplanken für weitere unabhängige Prüfung

Die folgenden Punkte grenzen vor allem offensichtliche Überinterpretationen
ein. Sie sollen weder einen Hauptprompt festlegen noch vorgeben, welche weiteren
Muster andere Reviewer oder Modelle suchen sollten.

- Die Evidenz ist verträglich mit einem kleinen Input plus offengelegtem
  Mechanismus als Arbeitsgrundlage; `mask` erfüllt das. Der Vorteil liegt
  nicht in gemessener Genauigkeit, sondern in Länge, Antwortrate und
  Identifizierbarkeit.
- Ob die Beispiele Teil der primären Bedingung sind oder als gleichrangige
  zweite Bedingung geführt werden, lässt sich aus den Daten nicht zwingend
  entscheiden. Beide Zellen existieren im Design, und ihr Kontrast ist selbst
  ein Ergebnis. Gegen die primäre Rolle sprechen die gesenkte Antwortrate bei
  tokenlimitierten Modellen und die über die historischen Läufe uneinheitliche
  Rangkorrelation. Dafür sprechen der konsistente Bias-Effekt, die auf 30
  Fällen günstigen gepaarten Codex-Ergebnisse und dass Few-Shot eine übliche
  Nutzungsform ist. Der abgeschlossene Claude-Stand ist je nach Zielgröße
  gemischt und stärkt daher keine der beiden Designoptionen eindeutig.
- Niveau und Rangfolge getrennt zu berichten macht mehrere der obigen Befunde
  erst sichtbar; ein einzelner Fehlerwert verdeckt sie. Eine
  Konstantvorhersage als Referenzzeile erleichtert die Interpretation.
- Der ProfileMAE mittelt über k=2..5. Die Komponenten k4 und k5 liegen
  wahrheitsseitig nahe null (Mittel 0,096 und 0,056) und sind entsprechend
  leicht zu treffen. Die `rho_k2`-Komponente daneben zu zeigen, trennt
  Profilmittel und Kopfgröße auf.
- Mehrere Messgrößen (Regressionssteigung, Abdeckungs-Skalierung,
  Strategie-Sensitivität, Ablesewert-Kopplung, Intervall-Informativität)
  ordnen die Läufe ähnlich. Ob sie dasselbe Konstrukt messen oder nur
  gemeinsam mit der Modellgröße variieren, wurde nicht getrennt.
- Explorative Schnitte nach Wahrheitsniveau, Strategie oder Coverage können
  aggregierte Effekte sichtbar verändern. Sie sind nützliche Hinweise, sollten
  aber zusammen mit Fallzahlen, Paarung und möglichen Konfundierungen gelesen
  werden.
- Aus sichtbarer Antwortlänge, Denkmodus oder Tool-Nutzung lässt sich bisher
  keine allgemeine optimale Bearbeitungsdauer ableiten. Im Hauptvergleich kann
  die Bearbeitung deshalb ohne zusätzliche Kürze- oder Reasoning-Vorgabe
  bleiben.

## Was offen ist

- **Panelwechsel:** Budgetprofil und Graphzusammensetzung unterscheiden sich.
  Der Beispiel-Effekt ist bei Codex auf je 30 exakt zum jeweiligen
  Baseline-Arm gepaarten Fällen gemessen. Bei Claude stehen 30 gepaarte
  Tools-Fälle und 27 vollständige gepaarte NoTools-Fälle zur Verfügung; ein
  weiterer NoTools-Fall timeoutete. Alle bleiben kleine Product-Screens auf
  dem historischen Panel und ersetzen keine Messung auf dem finalen
  Hauptpanel.
- **Beispieldesign:** Anzahl, Streuung und Nähe der Beispiele zum
  Gesamtmittelwert wurden nicht variiert. Der Unterschied zwischen der
  Phase-3-Zelle (ein Beispiel nahe am Mittel) und V2.1 (drei gestreute
  Beispiele) ist eine mögliche Erklärung für die unterschiedlich starken
  Effekte, aber nicht geprüft.
- **`C` verpflichtend:** Das null-Verhalten ist nur unter der aktuellen
  Prompt-Erlaubnis gemessen.
- **Konkrete Rechenregel:** Die offene `method`-Zelle ist kein Test einer
  fest vorgegebenen Verzerrungskorrektur. Eine solche neue Prompttechnik ist
  nicht separat geprüft und für den eingefrorenen Hauptvergleich nicht mehr
  als zusätzliche Zelle vorgesehen.
- **Qwen3.6 think:** Der dritte Tier ist lokal synchronisiert; 416/420 Prompts
  sind vollständig, vier bleiben am 126976-Tokenlimit offen.

## Rohdaten und Implementierung

- Prompts und Fälle: `results/llm_v2/prompts.jsonl`, `llm_cases.csv`,
  `llm_examples.csv`
- Antworten: `results/llm_v21/eval_input/`, `cluster_snapshot/`,
  `results/codex_screen_snapshot/`, `results/cc_screen_snapshot/`,
  `results/phase3/`
- Bestehende Auswertungen: `results/llm_v21/eval/` und die beiden
  `eval_*_screen_snapshot/`-Verzeichnisse
- Promptkonstruktion `src/make_llm_cases_v2.py`, Vollständigkeitskriterium
  `src/run_llm_v2.py:is_complete_record`

Die Kernzahlen erzeugt

```bash
PYTHONPATH=src python src/report_llm_evidence.py
```

über `src/llm_eval_frozen.py`, das die eingefrorene Bewertungsregel
implementiert und Parser sowie Vollständigkeitskriterium aus
`src/run_llm_v2.py` bezieht, damit Resume und Auswertung nicht auseinander
laufen. Einzelne Sektionen über `--section`; Invarianten in
`tests/test_llm_eval_frozen.py`.

Die vertiefenden Auswertungen (Steigung, Kohärenz, Abdeckungs-Skalierung,
Strategie, Stabilität, Ablesewert, Intervalle, Denkmodus, Textsignale) sind
bisher explorativ gerechnet und nicht Teil der getesteten Codebasis. Die
Skripte und ihre Ausgaben liegen im Evidenzpaket unter `analysis/`.

Die JSONL-Dateien sind zu groß für den Modellkontext. Wer sie selbst auswertet:
pro Prompt den letzten strukturell vollständigen Record wählen und über
`prompt_id` und `case_id` mit Prompt und Wahrheit verbinden. Einzelne Antworten
sind für die Plausibilitätsprüfung nützlich, nicht für Metriken.

Die bestehende Auswertung `src/eval_llm_v2.py` klippt Werte auf `[0, 1]` und
implementiert die Failure-Penalty nicht; ihre Tabellen sind eine bequeme
Übersicht, nicht die eingefrorene Bewertung.

Antwortdateien sind append-only. Rohvorhersagen werden nicht sortiert,
geklippt oder repariert — Modellinkonsistenz ist ein Ergebnis. Gemessen wurde
sie kaum: Monotonieverletzungen (`rho_k2 >= k3 >= k4 >= k5`) liegen bei allen
Läufen bei 0 %, außer r1-distill mit 2 %.
