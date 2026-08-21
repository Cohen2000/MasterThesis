# LLM-OFAT-Ablation: Startanleitung

## Tatsächlicher Umfang

Vier der sechs Zellen liegen bereits vor: `nw`, `mask`, `mask_crawl` und
`mask_all`. Neu fehlen `mask_temporal` und `mask_recent`, jeweils auf denselben
36 Fällen.

| Lauf | vorhandene Antworten | neu zu rechnen |
|---|---:|---:|
| Codex notools/high | 4 × 36 | 2 × 36 = 72 |
| Gemini 3.1 Flash Lite minimal | 4 × 36 | 2 × 36 × 3 Antworten = 216 |
| Qwen3.6-27B nothink | 4 × 36 | 2 × 36 = 72 |
| Qwen3.6-27B think | 4 × 36 | 2 × 36 = 72 |
| DeepSeek V4 Flash 0731 | nichts Wiederverwendbares | 6 × 36 = 216 |

Damit werden insgesamt **648 neue Generierungen** gestartet. Nur DeepSeek
wird vollständig neu gerechnet. Seine vorhandenen Zellen stammen von V4 Pro;
eine Mischung aus V4 Pro und V4 Flash wäre kein sauberer Faktorvergleich.

Die OFAT-Hauptauswertung verwendet bei allen Modellen Wiederholung 1. Die zwei
zusätzlichen Gemini-Antworten je neuer Zelle dienen als Stabilitätsdiagnose;
sie machen die vorhandenen vier Zellen nicht künstlich zu drei Replikaten.

## 1. Prompts vorbereiten

```bash
cd ~/Dokumente/MasterArbeit
bash scripts/run_llm_ofat.sh prepare
```

Erwartete Plangrößen:

```text
codex       72
gemini     216
deepseek   216
qwen        72
```

## 2. Lokale/API-Läufe starten

Schlüssel prüfen, ohne sie auszugeben:

```bash
test -n "${GEMINI_API_KEY:-}" && echo "Gemini-Key gesetzt" || echo "Gemini-Key fehlt"
test -n "${NVIDIA_API_KEY:-}" && echo "NVIDIA-Key gesetzt" || echo "NVIDIA-Key fehlt"
```

Danach:

```bash
bash scripts/run_llm_ofat.sh gemini
bash scripts/run_llm_ofat.sh deepseek
bash scripts/run_llm_ofat.sh codex
```

Gemini und DeepSeek laufen im Hintergrund. DeepSeek verteilt die 216 Prompts
standardmäßig auf acht Shards (27 pro Shard). Codex läuft im Vordergrund über
das ChatGPT-/Codex-Kontingent und verbraucht kein OpenAI-API-Budget.

Alle Befehle sind fortsetzbar: vollständige Prompt-IDs werden übersprungen.
DeepSeek führt bis zu drei Pässe für transiente Fehler oder unvollständige
Antworten aus. An einem unveränderten Tokenlimit abgeschnittene Antworten
werden nicht endlos wiederholt.

## 3. Lokalen Fortschritt prüfen

```bash
bash scripts/run_llm_ofat.sh status
```

Oder live:

```bash
watch -n 30 -c bash scripts/run_llm_ofat.sh status
```

Logs:

```bash
tail -f results/llm_v21_ofat/logs/gemini_minimal.shard0.log
tail -f results/llm_v21_ofat/logs/deepseek_flash.shard0.worker.log
```

Nach Codex die außerhalb des Repositories gespeicherte Datei kopieren:

```bash
cp ~/Dokumente/codex_ofat/answers_codex-gpt-5.6-sol_notools_high_ofat.jsonl \
   results/llm_v21_ofat/
```

## 4. Qwen zum Cluster hochladen

```bash
cd ~/Dokumente/MasterArbeit
WS=$(ssh uc3 'ws_find llm_pilot 2>/dev/null || echo $HOME/llm_pilot_ws')
ssh uc3 "mkdir -p '$WS/llm_v21' ~/MasterArbeit/slurm ~/MasterArbeit/scripts"

scp src/run_llm_v2.py src/select_llm_escalation.py \
    uc3:"$WS/llm_v21/"
scp results/llm_v21_ofat/prompts_ofat_qwen.jsonl \
    uc3:"$WS/llm_v21/"
scp slurm/llm_v21_common.sh slurm/llm_ofat_qwen36.sbatch \
    uc3:MasterArbeit/slurm/
scp scripts/submit_llm_ofat_qwen.sh scripts/cluster_llm_ofat_status.sh \
    uc3:MasterArbeit/scripts/
```

Die eingefrorene Datei `$WS/llm_v21/prompts.jsonl` wird dabei nicht
überschrieben.

## 5. Qwen starten und prüfen

```bash
ssh uc3 'cd ~/MasterArbeit && bash scripts/submit_llm_ofat_qwen.sh'
```

Es laufen acht Shards pro Modus. Die Einstellungen entsprechen den alten
Zellen, einschließlich derselben Eskalationsleiter:

- nothink: 8k → 16k → 32k
- think: 32k → 64k → 126976

Nur die 72 fehlenden Prompts je Modus werden verarbeitet.

Status:

```bash
ssh uc3 'cd ~/MasterArbeit && bash scripts/cluster_llm_ofat_status.sh'
```

Accounting:

```bash
ssh uc3 'sacct -S today --name=ofat_q36_primary,ofat_q36_n16,ofat_q36_n32,ofat_q36_t64,ofat_q36_t128 --format=JobID%24,JobName%22,State,Elapsed,MaxRSS'
```

## 6. Qwen herunterladen

```bash
cd ~/Dokumente/MasterArbeit
mkdir -p results/llm_v21_ofat

ssh uc3 bash -s <<'EOF' | tar -C results/llm_v21_ofat -xvf -
WS=$(ws_find llm_pilot 2>/dev/null || echo $HOME/llm_pilot_ws)
cd "$WS/llm_v21" || exit 1
tar cf - answers_ofat_qwen36_*.jsonl
EOF
```

## 7. Auswerten

```bash
bash scripts/run_llm_ofat.sh status
bash scripts/run_llm_ofat.sh evaluate
```

Ergebnisse:

- `results/llm_v21_ofat/eval/SUMMARY.md`
- `results/llm_v21_ofat/eval/ofat_metrics.csv`
- `results/llm_v21_ofat/eval/repeat_stability.csv`

Die Zellunterschiede werden auf denselben 36 Fällen gepaart. Bootstrap-
Intervalle resamplen Fälle beziehungsweise Graphen, nicht einzelne Antworten.
Fehlende Jobs werden nicht als Modellfehler gezählt; tatsächlich ungültige
Antworten erhalten wie eingefroren den Fehlerwert 1.
