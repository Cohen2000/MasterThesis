# Rekonstruktion des Ist-Zustands vor dem Qwen-Produktionslauf

Datum: 2026-09-18
Branch: `experiment/offline-freeze-20260916`
Lokaler HEAD / Remote HEAD: `778d0434f586520cbe99ba11122317d7d00b7245` ("protocol: finalize offline JSON revision before Qwen results")

## 1. Technischer Qwen-Smoke
- Status: **Vollständig und erfolgreich abgeschlossen** für beide Modi.
- Durchgeführte Jobs:
  - Job `7034830_0` (Non-thinking pass, 4 Requests in `answers_smoke/nonthinking_r1`, exit 0).
  - Job `7035216_3` (Thinking pass, 4 Requests in `answers_smoke/thinking_r1`, exit 0).
- Prüfpunkte:
  - Modell lädt fehlerfrei (`Qwen/Qwen3.6-35B-A3B`, safetensors, vLLM 0.29.0).
  - Thinking funktioniert (Reasoning sauber getrennt und geschlossen, `reasoning_closed: true`).
  - Non-thinking funktioniert (kein Reasoning im Text).
  - `StructuredOutputsParams(json_object=True)` funktioniert fehlerfrei.
  - JSON-Output strikt valide, Monotonie und Wertebereiche eingehalten (8/8 valid).
  - Tokenzählung und Hash-Bindungen (`engine_vs_own_input_tokens_mismatch: 0`, `prompt_hash_mismatch: 0`, `payload_hash_mismatch: 0`).
  - Keine dieser Smoke-Antworten wird in der Hauptauswertung verwendet.

## 2. Echte Hauptproduktion
- Status: **Noch nicht gestartet**.
- Verzeichnis `answers/` auf dem Cluster existiert noch nicht.
- Admitted Requests in der Hauptproduktion: 0
- Completed Requests in der Hauptproduktion: 0
- Requests mit nur `.attempt`: 0
- Ungestartete Requests: 1.680 (840 `qwen_thinking`, 840 `qwen_nonthinking`)

## 3. Runner und Cluster-Code
- Cluster-Verzeichnis: `/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot/cells10_json_20260918`
- Runner-Datei: `run_qwen_engine.py`
- SHA-256: `95b2ac5bf28c59e011d2fa4d7f3824f33e251db5b85b4fa1585e96564a67100b`
- Request-Manifest: `requests.jsonl`
- SHA-256: `12298bde7e5ec3a0eeee74e24b9a086a8579280dc409bf1968393d802b5592de`
- Exakte Übereinstimmung mit lokalem Stand `778d0434f586520cbe99ba11122317d7d00b7245`.

## 4. Offline-Abnahme
- `tests/frozen_main`: 125 Tests bestanden (0 Fehler).
- `scripts/verify_protocol_revision.py`: Erfolgreich verifiziert (600 unveränderte Blöcke, 500 Poolgraphen balanciert, 14 Modelle geprüft, 560 Vorhersagen nachgerechnet, 3.360 Requests disjunkt zu Altdaten, API release unauthorized).
